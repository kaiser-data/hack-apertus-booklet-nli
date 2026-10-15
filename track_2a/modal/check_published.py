"""Run the published submission image exactly as the organisers pull it (ghcr.io, linux/amd64).

    uv run --group modal modal run modal/check_published.py                       # examples/
    uv run --group modal modal run modal/check_published.py --cases <dir> --base-url <url>/v1

LLM_NAME is left unset, as in the official run, so the image picks the model itself.
"""

import json
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REF = "ghcr.io/kaiser-data/hack-apertus-booklet-nli:latest"
image = modal.Image.from_registry(REF).entrypoint([])
app = modal.App("booklet-nli-published-check", image=image)


@app.function(secrets=[modal.Secret.from_name("apertus-nli-api-key")], cpu=4, memory=4096,
              timeout=3600)
def run(files: dict[str, bytes], base_url: str) -> tuple[str, str]:
    import os

    for name, blob in files.items():
        (Path("/data") / name).parent.mkdir(parents=True, exist_ok=True)
        (Path("/data") / name).write_bytes(blob)
    Path("/output").mkdir(exist_ok=True)
    env = {"PATH": "/usr/bin:/bin", "BASE_URL": base_url, "API_KEY": os.environ["LLM_API_KEY"],
           "BOOKLET_NLI_DATA": "/tmp/booklet-nli", "BOOKLET_NLI_EMBEDDERS": "/app/models",
           "HF_HUB_OFFLINE": "1"}  # the image's own ENV, which Modal does not apply to subprocesses
    proc = subprocess.run(["/app/.venv/bin/booklet-nli-cases", "--input", "/data/cases.jsonl",
                           "--output", "/output/predictions.jsonl"],
                          env=env, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"exit {proc.returncode}\n{proc.stderr[-3000:]}")
    return Path("/output/predictions.jsonl").read_text(), proc.stderr[-1500:]


@app.local_entrypoint()
def main(cases: str = str(ROOT / "examples"),
         base_url: str = "https://kaiser-data--apertus-nli-70b-serve.us-east.modal.run/v1",
         out: str = "published-check.jsonl"):
    root = Path(cases)
    names = ["cases.jsonl"] + [f"booklets/{p.name}" for p in (root / "booklets").glob("*.pdf")]
    preds, err = run.remote({n: (root / n).read_bytes() for n in names}, base_url)
    rows = [json.loads(line) for line in preds.splitlines()]
    print(f"{REF}: {len(rows)} predictions")
    print("\n".join(line for line in err.splitlines() if "attempt" not in line))
    for r in rows[:4]:
        print(" ", r["id"], r["label_name"], [e["page"] for e in r["evidence"]], r["metrics"])
    (ROOT / "runs" / out).write_text(preds)
