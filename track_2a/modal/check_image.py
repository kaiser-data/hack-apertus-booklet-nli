"""Build the submission Dockerfile on Modal (linux/amd64, CPU) and run the evaluation contract.

    uv run --group modal modal run modal/check_image.py                 # examples/, live endpoint
    uv run --group modal modal run modal/check_image.py --cases <dir> --skip-offline

Two runs: one against the Apertus server, one with the network blocked, which must still parse
and embed every booklet (the embedder is baked into the image) and answer every case.
"""

import json
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
image = modal.Image.from_dockerfile(ROOT / "Dockerfile", context_dir=ROOT).entrypoint([])
app = modal.App("booklet-nli-image-check", image=image)
ENTRY = "/app/.venv/bin/booklet-nli-cases"


def _run(files: dict[str, bytes], env: dict[str, str], k: int = 16,
         extra: tuple[str, ...] = ()) -> tuple[str, str]:
    data = Path("/data")
    for name, blob in files.items():
        (data / name).parent.mkdir(parents=True, exist_ok=True)
        (data / name).write_bytes(blob)
    Path("/output").mkdir(exist_ok=True)
    proc = subprocess.run([ENTRY, "--input", "/data/cases.jsonl",
                           "--output", "/output/predictions.jsonl", "--k", str(k), *extra],
                          env={**env, "PATH": "/usr/bin:/bin"}, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"exit {proc.returncode}\n{proc.stderr[-3000:]}")
    return Path("/output/predictions.jsonl").read_text(), proc.stderr[-3000:]


@app.function(secrets=[modal.Secret.from_name("apertus-nli-api-key")], cpu=4, memory=4096,
              timeout=3600)
def online(files: dict[str, bytes], base_url: str, model: str, k: int = 16,
           extra: tuple[str, ...] = ()) -> tuple[str, str]:
    import os
    # model "" leaves LLM_NAME unset, as in the official run: the entrypoint picks the model.
    return _run(files, {"BASE_URL": base_url, "API_KEY": os.environ["LLM_API_KEY"],
                        **({"LLM_NAME": model} if model else {}),
                        "BOOKLET_NLI_DATA": "/tmp/booklet-nli", "BOOKLET_NLI_EMBEDDERS": "/app/models",
                        "HF_HUB_OFFLINE": "1"}, k, extra)


@app.function(block_network=True, cpu=4, memory=4096, timeout=1800)
def offline(files: dict[str, bytes]) -> tuple[str, str]:
    return _run(files, {"BASE_URL": "http://127.0.0.1:9/v1", "API_KEY": "x",
                        "BOOKLET_NLI_DATA": "/tmp/booklet-nli", "BOOKLET_NLI_EMBEDDERS": "/app/models",
                        "HF_HUB_OFFLINE": "1"})


@app.local_entrypoint()
def main(cases: str = str(ROOT / "examples"),
         base_url: str = "https://kaiser-data--apertus-nli-serve.modal.run/v1",
         skip_offline: bool = False, model: str = "",
         out: str = "image-check-predictions.jsonl", k: int = 16, glossary: bool = False):
    root = Path(cases)
    names = ["cases.jsonl"] + [f"booklets/{p.name}" for p in (root / "booklets").glob("*.pdf")]
    files = {n: (root / n).read_bytes() for n in names}
    n_cases = len(files["cases.jsonl"].decode().splitlines())
    runs = [("online", lambda: online.remote(files, base_url, model, k,
                                                   ("--actor-glossary",) if glossary else ()))]
    if not skip_offline:
        runs.insert(0, ("offline", lambda: offline.remote(files)))
    for name, call in runs:
        result, err = call()
        preds = [json.loads(line) for line in result.splitlines()]
        print(f"== {name}: {len(preds)}/{n_cases} predictions")
        print("\n".join(line for line in err.splitlines() if "attempt" not in line)[-1500:])
        for p in preds[:4]:
            print(" ", p["id"], p["label_name"], [e["page"] for e in p["evidence"]], p["metrics"])
        if name == "online":
            (ROOT / "runs" / out).write_text(result)
