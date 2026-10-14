"""Live demo: type a claim, pick a voting booklet, get Apertus's verdict with the cited passages.

A small CPU web app in front of the GPU server from `modal/serve_apertus.py`. It reuses `verify()`
exactly as the evaluation does (16 retrieved passages, same prompt), so a judge sees the system
that was measured. The API key stays in the Modal secret; the browser never sees it.

    modal serve modal/demo.py      # temporary URL while the command runs
    modal deploy modal/demo.py     # https://<workspace>--booklet-nli-demo-web.modal.run

Guards: only the 60 booklets of the benchmark (20 vote dates x de/fr/it), claims up to
MAX_CLAIM characters, at most PER_IP_PER_DAY requests per client and DAILY_CAP in total, one
container with a few requests at a time. The GPU server scales to zero when idle, so the first
request after a pause waits for it to start (about 5 minutes); the page shows that state.
"""

import datetime as dt
import os
import threading
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
VOL_PATH = "/vol"
MAX_CLAIM = 400
PER_IP_PER_DAY = 30
DAILY_CAP = 400

image = (
    modal.Image.debian_slim(python_version="3.12")
    # Same versions as uv.lock.
    .uv_pip_install("fastembed==0.9.0", "httpx==0.28.1", "numpy==2.5.4", "openai==3.28.0",
                    "pymupdf==1.28.2", "fastapi[standard]")
    .env({"BOOKLET_NLI_DATA": f"{VOL_PATH}/data", "LLM_NAME": "swiss-ai/Apertus-v1.5-8B"})
    .add_local_dir(ROOT / "src" / "booklet_nli", "/root/booklet_nli")
    .add_local_file(Path(__file__).with_name("demo.html"), "/root/demo.html")
)
# The evaluation volume already holds the dataset, booklets, passages, embeddings and embedder.
volume = modal.Volume.from_name("booklet-nli-data", create_if_missing=True)
quota = modal.Dict.from_name("booklet-nli-demo-quota", create_if_missing=True)

app = modal.App("booklet-nli-demo")


def booklet_catalog() -> dict:
    """{date: {lang: {"url": ..., "votes": [...]}}} for every booklet in the benchmark."""
    from booklet_nli.data import load_rows

    cat: dict = {}
    for r in load_rows():
        entry = cat.setdefault(r["booklet_publish_date"], {}).setdefault(
            r["reference_language"], {"url": r["booklet_url"], "votes": []})
        if r["vote"] not in entry["votes"]:
            entry["votes"].append(r["vote"])
    return dict(sorted(cat.items(), reverse=True))


@app.function(image=image, cpu=2, memory=4096, volumes={VOL_PATH: volume},
              secrets=[modal.Secret.from_name("apertus-nli-api-key")],
              max_containers=1, scaledown_window=600, timeout=600)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def web():
    import httpx
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel

    from booklet_nli.config import llm_config
    from booklet_nli.llm import LLM
    from booklet_nli.nli import verify

    server = modal.Function.from_name("apertus-nli", "serve").get_web_url().rstrip("/")
    os.environ["LLM_BASE_URL"] = f"{server}/v1"
    llm = LLM(llm_config(), timeout=420)
    catalog = booklet_catalog()
    urls = {lang_entry["url"] for date in catalog.values() for lang_entry in date.values()}
    page = Path("/root/demo.html").read_text()
    waking = threading.Event()

    def wake() -> None:
        # vLLM's /health needs no key; a long request keeps Modal starting the GPU container.
        try:
            httpx.get(f"{server}/health", timeout=30 * 60)
        finally:
            waking.clear()

    api = FastAPI(docs_url=None, redoc_url=None)

    @api.get("/", response_class=HTMLResponse)
    def index() -> str:
        return page

    @api.get("/api/booklets")
    def booklets() -> dict:
        return catalog

    @api.get("/api/health")
    def health() -> dict:
        try:
            ready = httpx.get(f"{server}/health", timeout=4).status_code == 200
        except httpx.HTTPError:
            ready = False
        if not ready and not waking.is_set():
            waking.set()
            threading.Thread(target=wake, daemon=True).start()
        return {"ready": ready}

    class Ask(BaseModel):
        booklet: str
        claim: str

    def spend(ip: str) -> None:
        day = dt.date.today().isoformat()
        total, mine = quota.get(f"all:{day}", 0), quota.get(f"ip:{day}:{ip}", 0)
        if total >= DAILY_CAP:
            raise HTTPException(429, "The demo's daily limit is reached. Please try again tomorrow.")
        if mine >= PER_IP_PER_DAY:
            raise HTTPException(429, f"Limit of {PER_IP_PER_DAY} claims per day reached.")
        quota[f"all:{day}"], quota[f"ip:{day}:{ip}"] = total + 1, mine + 1

    @api.post("/api/verify")
    def check(ask: Ask, request: Request) -> dict:
        claim = " ".join(ask.claim.split())
        if ask.booklet not in urls:
            raise HTTPException(400, "Unknown booklet.")
        if not 8 <= len(claim) <= MAX_CLAIM:
            raise HTTPException(400, f"The claim must be 8 to {MAX_CLAIM} characters long.")
        # The Cloudflare Pages front end forwards the visitor's address as x-demo-client.
        ip = (request.headers.get("x-demo-client") or request.headers.get("x-forwarded-for")
              or request.client.host).split(",")[0].strip()
        spend(ip)
        v = verify(llm, ask.booklet, claim)
        return {**v.to_dict(), "parsed": v.parsed}

    return api
