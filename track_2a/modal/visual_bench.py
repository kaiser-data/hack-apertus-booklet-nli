"""Page-level retrieval on the dev split: MiniLM (max over a page's passages) vs ColQwen2.

    uv run --group modal modal run modal/visual_bench.py

Gold pages of a claim: every page with a PyMuPDF passage that overlaps the annotated reference
(`overlaps_reference`). Claims with no gold page are dropped and counted. Hit@k: a gold page is
among the top-k pages of the claim's booklet.
"""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
VOL_PATH = "/vol"
CACHE_PATH = "/cache"
COLPALI_VERSION = "0.3.18"
base = (
    modal.Image.from_dockerfile(ROOT / "Dockerfile", context_dir=ROOT)
    .entrypoint([])
    .env({"BOOKLET_NLI_DATA": f"{VOL_PATH}/data", "HF_HUB_OFFLINE": "0",
          "BOOKLET_NLI_EMBEDDERS": f"{VOL_PATH}/embedders"})
)
gpu_image = base.run_commands(
    f"uv pip install --python /app/.venv/bin/python colpali-engine=={COLPALI_VERSION}"
).env({"HF_HOME": f"{CACHE_PATH}/hf"})
volume = modal.Volume.from_name("booklet-nli-data", create_if_missing=True)
cache = modal.Volume.from_name("booklet-nli-visual-cache", create_if_missing=True)
app = modal.App("booklet-nli-visual-bench", image=base)
KS = (1, 3, 5)
DPI = 110


def _setup():
    import sys

    sys.path[:0] = ["/app/src", "/app/.venv/lib/python3.12/site-packages"]
    from booklet_nli.booklet import extract_passages, fetch_pdf
    from booklet_nli.data import load_rows, select
    from booklet_nli.metrics import overlaps_reference

    rows = [r for r in select(load_rows(), "dev", None) if r["entailment_label"] != 1]
    books = {u: extract_passages(fetch_pdf(u)) for u in sorted({r["booklet_url"] for r in rows})}
    kept, dropped = [], 0
    for r in rows:
        gold = sorted({p.page for p in books[r["booklet_url"]]
                       if overlaps_reference(p.text, r["reference_string"])})
        if gold:
            kept.append({**r, "gold_pages": gold})
        else:
            dropped += 1
    return kept, dropped, books, fetch_pdf


def _hits(rows: list[dict], rankings: list[list[int]]) -> dict:
    """Hit@k overall and by same- vs cross-language; rankings hold 1-based pages, best first."""
    groups: dict[str, list[tuple[dict, list[int]]]] = {"all": [], "same_language": [],
                                                       "cross_language": []}
    for r, order in zip(rows, rankings):
        groups["all"].append((r, order))
        same = r["claim_language"] == r["reference_language"]
        groups["same_language" if same else "cross_language"].append((r, order))
    return {g: {"n": len(v), **{f"hit@{k}": round(sum(bool(set(o[:k]) & set(r["gold_pages"]))
                                                       for r, o in v) / max(len(v), 1), 4)
                                for k in KS}}
            for g, v in groups.items()}


@app.function(cpu=4, memory=8192, timeout=3600, volumes={VOL_PATH: volume})
def minilm() -> dict:
    import time

    rows, dropped, books, _ = _setup()
    import numpy as np

    from booklet_nli import retrieve

    start = time.perf_counter()
    mats = {u: retrieve.passage_matrix(ps, retrieve.DEFAULT_EMBEDDER) for u, ps in books.items()}
    embed_s = time.perf_counter() - start
    rankings = []
    for r in rows:
        ps = books[r["booklet_url"]]
        sims = mats[r["booklet_url"]] @ retrieve._embed(retrieve.DEFAULT_EMBEDDER, [r["claim"]],
                                                        "query")[0]
        best: dict[int, float] = {}
        for p, s in zip(ps, sims):
            best[p.page] = max(best.get(p.page, -np.inf), float(s))
        rankings.append(sorted(best, key=lambda pg: -best[pg]))
    volume.commit()
    return {"method": f"{retrieve.DEFAULT_EMBEDDER} (page = max cosine over its passages)",
            "claims": len(rows), "dropped_no_gold_page": dropped, "booklets": len(books),
            "passages": sum(len(p) for p in books.values()),
            "embed_s_total_cached_or_cpu": round(embed_s, 1),
            "hits": _hits(rows, rankings), "rankings": [o[:5] for o in rankings]}


@app.function(image=gpu_image, gpu="L40S", cpu=4, memory=32768, timeout=2 * 3600,
              volumes={VOL_PATH: volume, CACHE_PATH: cache},
              secrets=[modal.Secret.from_name("huggingface-secret")])
def colqwen(model_name: str = "vidore/colqwen2-v1.0", batch: int = 4) -> dict:
    import time

    rows, dropped, books, fetch_pdf = _setup()
    import colpali_engine
    import pymupdf
    import torch
    from colpali_engine.models import ColPali, ColPaliProcessor, ColQwen2, ColQwen2Processor
    from PIL import Image

    Model, Proc = (ColQwen2, ColQwen2Processor) if "colqwen" in model_name else (ColPali,
                                                                                 ColPaliProcessor)
    t0 = time.perf_counter()
    try:
        model = Model.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda:0").eval()
    except TypeError:
        model = Model.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                      device_map="cuda:0").eval()
    proc = Proc.from_pretrained(model_name)
    load_s = time.perf_counter() - t0
    cache.commit()

    render_s, gpu_s, page_emb, n_pages = 0.0, {}, {}, {}
    for url in books:
        t = time.perf_counter()
        doc = pymupdf.open(fetch_pdf(url))
        imgs = []
        for page in doc:
            pix = page.get_pixmap(dpi=DPI)
            imgs.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
        render_s += time.perf_counter() - t
        n_pages[url] = len(imgs)
        torch.cuda.synchronize()
        t = time.perf_counter()
        embs = []
        with torch.no_grad():
            for i in range(0, len(imgs), batch):
                inp = proc.process_images(imgs[i:i + batch]).to(model.device)
                embs.extend(e for e in model(**inp))  # padded rows score ~0 under MaxSim
        torch.cuda.synchronize()
        gpu_s[url] = time.perf_counter() - t
        page_emb[url] = embs

    t = time.perf_counter()
    rankings = []
    with torch.no_grad():
        for r in rows:
            q = model(**proc.process_queries([r["claim"]]).to(model.device))
            scores = proc.score_multi_vector(list(q), page_emb[r["booklet_url"]])[0]
            rankings.append([int(i) + 1 for i in torch.argsort(scores, descending=True)])
    query_s = time.perf_counter() - t
    return {"method": f"{model_name} MaxSim over page images ({DPI} dpi)",
            "colpali_engine": str(colpali_engine.__version__) if hasattr(colpali_engine, "__version__")
            else COLPALI_VERSION,
            "torch": str(torch.__version__), "gpu": str(torch.cuda.get_device_name(0)),
            "claims": len(rows), "dropped_no_gold_page": dropped, "booklets": len(books),
            "pages": sum(n_pages.values()), "model_load_s": round(load_s, 1),
            "render_s_total_cpu": round(render_s, 1),
            "gpu_page_embed_s_total": round(sum(gpu_s.values()), 1),
            "gpu_page_embed_s_per_booklet": round(sum(gpu_s.values()) / len(books), 2),
            "query_ms": round(query_s / len(rows) * 1000, 1),
            "hits": _hits(rows, rankings), "rankings": [o[:5] for o in rankings],
            "claim_ids": [r["id"] for r in rows], "gold_pages": [r["gold_pages"] for r in rows]}


@app.local_entrypoint()
def main(model: str = "vidore/colqwen2-v1.0", skip_minilm: bool = False):
    calls = {"colqwen": colqwen.spawn(model)}
    if not skip_minilm:
        calls["minilm"] = minilm.spawn()
    out = {}
    for name, call in calls.items():
        try:
            out[name] = call.get()
        except Exception as e:  # noqa: BLE001
            out[name] = {"error": repr(e)[:2000]}
        print(name, json.dumps({k: v for k, v in out[name].items()
                                if k not in ("rankings", "claim_ids", "gold_pages")}))
    path = ROOT / "runs" / "visual-bench-dev.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(out)
    path.write_text(json.dumps(old, indent=1))
