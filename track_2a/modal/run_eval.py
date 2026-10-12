"""Run `booklet-nli eval` on a Modal CPU container instead of the laptop.

The container is built from this project's own Dockerfile, so it also checks the image `make run`
uses. Modal's client goes into the image's system Python; the project stays in /app/.venv. The
container talks to the deployed Apertus server (`modal/serve_apertus.py`). Booklets, passages,
embeddings and run directories live on the Volume `booklet-nli-data`, so runs resume and the cache
survives between them.

    modal run modal/run_eval.py --split test --limit 976 --k 16 --name test-rag-k16-n976
    modal run modal/run_eval.py --split test --limit 60 --mode full --name test-full-n60

If `runs/<name>/predictions.jsonl` exists locally, it is uploaded first and those rows are
skipped. When the run ends, `predictions.jsonl` and `metrics.json` are written back to
`runs/<name>/`.
Requires the Modal secret `apertus-nli-api-key` (LLM_API_KEY).
"""

import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
MODEL = "swiss-ai/Apertus-v1.5-8B"
VOL_PATH = "/vol"

image = (
    modal.Image.from_dockerfile(ROOT / "Dockerfile", context_dir=ROOT)
    .entrypoint([])
    .env({"BOOKLET_NLI_DATA": f"{VOL_PATH}/data", "LLM_NAME": MODEL})
)
volume = modal.Volume.from_name("booklet-nli-data", create_if_missing=True)

app = modal.App("booklet-nli-eval")


@app.function(image=image, cpu=4, memory=8192, timeout=4 * 60 * 60,
              volumes={VOL_PATH: volume},
              secrets=[modal.Secret.from_name("apertus-nli-api-key")])
def run_eval(split: str, limit: int, mode: str, k: int, workers: int, name: str) -> dict:
    import os

    server = modal.Function.from_name("apertus-nli", "serve").get_web_url().rstrip("/")
    env = {**os.environ, "LLM_BASE_URL": f"{server}/v1"}
    out = Path(VOL_PATH) / "runs" / name
    cmd = ["/app/.venv/bin/booklet-nli", "eval", "--split", split, "--limit", str(limit),
           "--mode", mode, "--k", str(k), "--workers", str(workers), "--out", str(out)]
    print(" ".join(cmd), flush=True)
    code = subprocess.run(cmd, check=False, env=env).returncode  # keep partial results
    volume.commit()
    files = {p.name: p.read_text() for p in (out / "predictions.jsonl", out / "metrics.json")
             if p.exists()}
    return {"returncode": code, "files": files}


@app.local_entrypoint()
def main(split: str = "test", limit: int = 60, mode: str = "rag", k: int = 16,
         workers: int = 12, name: str = ""):
    name = name or f"{split}-{mode}-k{k}-n{limit}"
    local = ROOT / "runs" / name
    done = local / "predictions.jsonl"
    if done.exists() and done.stat().st_size:
        with volume.batch_upload(force=True) as batch:
            batch.put_file(str(done), f"/runs/{name}/predictions.jsonl")
        print(f"uploaded {sum(1 for _ in done.open())} finished rows from {done}")

    result = run_eval.remote(split, limit, mode, k, workers, name)
    local.mkdir(parents=True, exist_ok=True)
    for fname, text in result["files"].items():
        (local / fname).write_text(text)
    print(f"exit {result['returncode']}; wrote {sorted(result['files'])} to {local}")
