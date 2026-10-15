"""Docling (layout-aware) chunks vs PyMuPDF passages for retrieval on the dev split.

    uv run --group modal modal run modal/docling_bench.py

Each dev booklet is converted on its own 4-vCPU container (no OCR: the booklets are born-digital).
Chunks: (a) "items": Docling body items in reading order, packed into <=900-character chunks that
never cross a page, headed by the last section header; (b) "hybrid": Docling's HybridChunker
(MiniLM tokenizer, 256 tokens), split further at word boundaries to <=900 characters.
All are ranked with the same MiniLM embedder; recall@k = some top-k chunk overlaps the reference.
"""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
VOL_PATH = "/vol"
DOCLING_VERSION = "2.135.0"
EMBEDDER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
image = (
    modal.Image.from_dockerfile(ROOT / "Dockerfile", context_dir=ROOT)
    .entrypoint([])
    .run_commands(f"uv pip install --python /app/.venv/bin/python docling=={DOCLING_VERSION}",
                  "HF_HUB_OFFLINE=0 /app/.venv/bin/docling-tools models download layout tableformer",
                  # Modal's runtime imports the system typing_extensions first; docling needs a newer one.
                  "pip install -U 'typing_extensions>=4.14'")
    .env({"BOOKLET_NLI_DATA": f"{VOL_PATH}/data", "HF_HUB_OFFLINE": "0",
          "BOOKLET_NLI_EMBEDDERS": f"{VOL_PATH}/embedders", "OMP_NUM_THREADS": "4"})
)
volume = modal.Volume.from_name("booklet-nli-data", create_if_missing=True)
app = modal.App("booklet-nli-docling-bench", image=image)
KS = (8, 16)
MAX_CHARS = 900


def _path():
    import sys

    sys.path[:0] = ["/app/src", "/app/.venv/lib/python3.12/site-packages"]


def _split(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    out, buf = [], ""
    for word in text.split():
        if buf and len(buf) + 1 + len(word) > max_chars:
            out.append(buf)
            buf = ""
        buf = f"{buf} {word}".strip()
    return out + ([buf] if buf else [])


@app.function(cpu=4, memory=16384, timeout=2 * 3600, volumes={VOL_PATH: volume})
def convert(url: str) -> dict:
    import time

    _path()
    from booklet_nli.booklet import fetch_pdf

    try:
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    except ImportError:
        from docling.datamodel.pipeline_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pdf = fetch_pdf(url)
    opts = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
    opts.accelerator_options = AcceleratorOptions(num_threads=4, device=AcceleratorDevice.CPU)
    conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    t = time.perf_counter()
    conv.initialize_pipeline(InputFormat.PDF)  # model load, not counted as conversion
    init_s = time.perf_counter() - t
    t = time.perf_counter()
    doc = conv.convert(pdf).document
    conv_s = time.perf_counter() - t

    items, section, sec, page, buf = [], "", "", None, ""
    for item, _lvl in doc.iterate_items():
        label = str(getattr(item, "label", ""))
        if not getattr(item, "prov", None):
            continue
        pg = item.prov[0].page_no
        if "table" in label:
            text = item.export_to_markdown(doc=doc)
        else:
            text = getattr(item, "text", "") or ""
        text = " ".join(text.split())
        if "section_header" in label or label.endswith("title"):
            section = text
        if len(text) < 25:
            continue
        if buf and (pg != page or len(buf) + 1 + len(text) > MAX_CHARS):
            items.extend({"page": page, "section": sec, "text": c} for c in _split(buf))
            buf = ""
        if not buf:
            page, sec = pg, section
        buf = f"{buf} {text}".strip()
    if buf:
        items.extend({"page": page, "section": sec, "text": c} for c in _split(buf))

    hybrid, hybrid_err = [], None
    try:
        from docling.chunking import HybridChunker
        from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
        from transformers import AutoTokenizer

        tok = HuggingFaceTokenizer(tokenizer=AutoTokenizer.from_pretrained(EMBEDDER), max_tokens=256)
        for ch in HybridChunker(tokenizer=tok).chunk(doc):
            pages = sorted({p.page_no for it in ch.meta.doc_items for p in it.prov})
            head = " › ".join(ch.meta.headings or [])
            hybrid.extend({"page": pages[0] if pages else None, "section": head, "text": c}
                          for c in _split(" ".join(ch.text.split())))
    except Exception as e:  # noqa: BLE001
        hybrid_err = repr(e)[:500]
    return {"url": url, "pages": doc.num_pages(), "init_s": round(init_s, 1),
            "convert_s": round(conv_s, 1), "items": items, "hybrid": hybrid,
            "hybrid_error": hybrid_err}


@app.function(cpu=4, memory=8192, timeout=3600, volumes={VOL_PATH: volume})
def score(converted: list[dict]) -> dict:
    import time

    _path()
    import docling

    from booklet_nli import retrieve
    from booklet_nli.booklet import extract_passages, fetch_pdf
    from booklet_nli.data import load_rows, select
    from booklet_nli.metrics import overlaps_reference

    rows = [r for r in select(load_rows(), "dev", None) if r["entailment_label"] != 1]
    conv = {c["url"]: c for c in converted}
    pym = {u: [{"page": p.page, "section": p.section, "text": p.text}
               for p in extract_passages(fetch_pdf(u))] for u in conv}
    out = {"embedder": EMBEDDER, "claims": len(rows), "booklets": len(conv),
           "docling_version": getattr(docling, "__version__", DOCLING_VERSION),
           "cpu": "Modal 4 vCPU, 16 GB, no OCR, table structure on",
           "convert_s_per_booklet": round(sum(c["convert_s"] for c in converted) / len(conv), 1),
           "convert_s_total": round(sum(c["convert_s"] for c in converted), 1),
           "pages": sum(c["pages"] for c in converted),
           "init_s_mean": round(sum(c["init_s"] for c in converted) / len(conv), 1),
           "hybrid_errors": [c["hybrid_error"] for c in converted if c["hybrid_error"]],
           "methods": {}}
    for name, books in (("pymupdf", pym), ("docling_items", {u: c["items"] for u, c in conv.items()}),
                        ("docling_hybrid", {u: c["hybrid"] for u, c in conv.items()})):
        if not all(books.values()):
            continue
        t = time.perf_counter()
        mats = {u: retrieve._embed(EMBEDDER, [f"{c['section']}\n{c['text']}" for c in cs], "passage")
                for u, cs in books.items()}
        embed_s = time.perf_counter() - t
        hits = dict.fromkeys(KS, 0)
        findable = 0  # claims where some chunk at all overlaps the reference (ceiling)
        for r in rows:
            cs = books[r["booklet_url"]]
            ok = [overlaps_reference(c["text"], r["reference_string"]) for c in cs]
            findable += any(ok)
            order = (mats[r["booklet_url"]] @ retrieve._embed(EMBEDDER, [r["claim"]], "query")[0]
                     ).argsort()[::-1]
            for k in KS:
                hits[k] += any(ok[i] for i in order[:k])
        n_chunks = sum(len(c) for c in books.values())
        out["methods"][name] = {
            "chunks": n_chunks,
            "mean_chars": round(sum(len(c["text"]) for cs in books.values() for c in cs) / n_chunks),
            "any_chunk_overlaps": round(findable / len(rows), 4),
            **{f"recall@{k}": round(v / len(rows), 4) for k, v in hits.items()},
            "embed_s_per_booklet": round(embed_s / len(books), 1)}
    volume.commit()
    return out


@app.local_entrypoint()
def main():
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    urls = score_urls.remote()
    converted, failed = [], []
    for u, r in zip(urls, convert.map(urls, return_exceptions=True)):
        if isinstance(r, Exception):
            failed.append({"url": u, "error": repr(r)[:1000]})
            print("FAILED", u, repr(r)[:300])
        else:
            converted.append(r)
            print(u, r["pages"], "pages", r["convert_s"], "s")
    result = score.remote(converted) if converted else {}
    result["failed"] = failed
    result["per_booklet"] = [{k: c[k] for k in ("url", "pages", "init_s", "convert_s")}
                             for c in converted]
    print(json.dumps({k: v for k, v in result.items() if k != "per_booklet"}, indent=1))
    (ROOT / "runs" / "docling-bench-dev.json").write_text(json.dumps(result, indent=1))


@app.function(cpu=1, volumes={VOL_PATH: volume})
def score_urls() -> list[str]:
    _path()
    from booklet_nli.data import load_rows, select

    return sorted({r["booklet_url"] for r in select(load_rows(), "dev", None)
                   if r["entailment_label"] != 1})
