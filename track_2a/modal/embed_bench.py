"""Compare embedders for retrieval on the dev split, each on its own 4-vCPU Modal container.

    uv run --group modal modal run modal/embed_bench.py

For every embedder: recall@k of the annotated reference passage over all non-neutral dev claims
(the same measure as `booklet-nli retrieval`), and the CPU seconds to embed every passage of the
dev booklets once, which is the work the official time score can see.
"""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
VOL_PATH = "/vol"
image = (
    modal.Image.from_dockerfile(ROOT / "Dockerfile", context_dir=ROOT)
    .entrypoint([])
    .env({"BOOKLET_NLI_DATA": f"{VOL_PATH}/data", "HF_HUB_OFFLINE": "0",
          "BOOKLET_NLI_EMBEDDERS": f"{VOL_PATH}/embedders"})
)
volume = modal.Volume.from_name("booklet-nli-data", create_if_missing=True)
app = modal.App("booklet-nli-embed-bench", image=image)

EMBEDDERS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "google/embeddinggemma-300m",
    "intfloat/multilingual-e5-large",
    "Qwen/Qwen3-Embedding-0.6B-Q",
]
KS = (1, 3, 5, 8, 12, 16, 20)


def _materialize_symlinks(root: Path) -> None:
    import shutil

    for link in [p for p in root.rglob("*") if p.is_symlink()]:
        target = link.resolve()
        link.unlink()
        shutil.copy2(target, link)


@app.function(cpu=4, memory=16384, timeout=3 * 60 * 60, volumes={VOL_PATH: volume})
def bench(embedder: str) -> dict:
    import sys
    import time

    # The project and its dependencies live in /app/.venv (same Python 3.12 as Modal's runtime).
    sys.path[:0] = ["/app/src", "/app/.venv/lib/python3.12/site-packages"]
    from booklet_nli import retrieve
    from booklet_nli.booklet import extract_passages, fetch_pdf
    from booklet_nli.data import load_rows, select
    from booklet_nli.metrics import overlaps_reference

    rows = [r for r in select(load_rows(), "dev", None) if r["entailment_label"] != 1]
    books = {u: extract_passages(fetch_pdf(u)) for u in sorted({r["booklet_url"] for r in rows})}
    try:
        retrieve._embed(embedder, ["warm-up"], "query")  # download + load, not timed
    except Exception:  # noqa: BLE001
        # ONNX external weights behind HF-cache symlinks: "path escapes model directory".
        _materialize_symlinks(Path(f"{VOL_PATH}/embedders"))
        retrieve._load.cache_clear()
        retrieve._embed(embedder, ["warm-up"], "query")
    start = time.perf_counter()
    mats = {u: retrieve._embed(embedder, [f"{p.section}\n{p.text}" for p in ps], "passage")
            for u, ps in books.items()}
    embed_s = time.perf_counter() - start
    hits, hybrid = dict.fromkeys(KS, 0), dict.fromkeys(KS, 0)
    q_start = time.perf_counter()
    for r in rows:
        ps = books[r["booklet_url"]]
        dense = mats[r["booklet_url"]] @ retrieve._embed(embedder, [r["claim"]], "query")[0]
        lexical = retrieve.bm25_scores(r["claim"], [f"{p.section} {p.text}" for p in ps])
        for name, order in (("dense", dense.argsort()[::-1]),
                            ("hybrid", retrieve.rrf(dense, lexical))):
            tally = hits if name == "dense" else hybrid
            for k in KS:
                tally[k] += any(overlaps_reference(ps[i].text, r["reference_string"])
                                for i in order[:k])
    volume.commit()
    n_pass = sum(len(ps) for ps in books.values())
    return {"embedder": embedder, "claims": len(rows), "booklets": len(books), "passages": n_pass,
            "recall_at_k": {k: round(v / len(rows), 4) for k, v in hits.items()},
            "hybrid_bm25_rrf_recall_at_k": {k: round(v / len(rows), 4) for k, v in hybrid.items()},
            "embed_s_total": round(embed_s, 1), "embed_s_per_booklet": round(embed_s / len(books), 1),
            "query_ms": round((time.perf_counter() - q_start) / len(rows) * 1000, 1)}


@app.local_entrypoint()
def main(only: str = ""):
    names = [e for e in EMBEDDERS if not only or any(o in e for o in only.split(","))]
    results = list(bench.map(names, return_exceptions=True))
    out = []
    for e, r in zip(names, results):
        if isinstance(r, Exception):
            print(f"{e}: FAILED {r!r}")
            continue
        out.append(r)
        print(json.dumps(r))
    path = ROOT / "runs" / "embed-bench-dev.json"
    old = json.loads(path.read_text()) if path.exists() else []
    merged = {r["embedder"]: r for r in old + out}
    path.write_text(json.dumps(list(merged.values()), indent=2))
