"""Serve the official swiss-ai/Apertus-v1.5-70B with vLLM on Modal (2x H100, tensor parallel).

A copy of serve_apertus.py for the 70B model, as a separate app so the 8B server stays deployed.

Weights:  modal run modal/serve_apertus_70b.py::download   (CPU container, ~140 GB, once)
Deploy:   modal deploy modal/serve_apertus_70b.py
Then:     BASE_URL=https://<workspace>--apertus-nli-70b-serve.modal.run/v1  LLM_NAME=swiss-ai/Apertus-v1.5-70B

The container scales to zero after SCALEDOWN seconds idle, so you pay only while evaluating.
Requires two Modal secrets: `huggingface-secret` (HF_TOKEN with access to the gated model)
and `apertus-nli-api-key` (LLM_API_KEY, the bearer token clients must send).
"""

import os
import subprocess

import modal

MODEL = "swiss-ai/Apertus-v1.5-70B"
# swiss-ai's own vLLM build for Apertus 1.5 (upstream support is pending), pinned by digest.
IMAGE = ("ghcr.io/swiss-ai/vllm_apertus_1.5_release@"
         "sha256:d85533f251c8272218ae2e519510836376e5a335b44f2204f193018e37753bc6")
GPU = "H100:2"
TENSOR_PARALLEL = 2
# Retrieved context is ~3.5k tokens and the longest task B reference ~8k; 32k leaves KV room.
MAX_MODEL_LEN = 32768
SCALEDOWN = 300
PORT = 8000

# The image's Python lives in /opt/venv without Modal's runtime dependencies; install the
# client there explicitly (uv if the image has it, pip otherwise).
image = (
    modal.Image.from_registry(IMAGE)
    .entrypoint([])
    # The image's hf_xet fails on this repo ("Unable to parse string as hex hash value").
    .env({"HF_HUB_DISABLE_XET": "1"})
    .run_commands(
        "(command -v uv >/dev/null && uv pip install --python /opt/venv/bin/python 'modal==1.5.3')"
        " || /opt/venv/bin/python -m pip install 'modal==1.5.3'"
    )
)
hf_cache = modal.Volume.from_name("apertus-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("apertus-vllm-cache", create_if_missing=True)

app = modal.App("apertus-nli-70b")


@app.function(
    image=image,
    gpu=GPU,
    scaledown_window=SCALEDOWN,
    timeout=2 * 60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[modal.Secret.from_name("huggingface-secret"),
             modal.Secret.from_name("apertus-nli-api-key")],
)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=PORT, startup_timeout=30 * 60)
def serve() -> None:
    cmd = [
        "vllm", "serve", MODEL,
        "--served-model-name", MODEL,
        "--host", "0.0.0.0", "--port", str(PORT),
        "--chat-template-content-format", "string",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--tensor-parallel-size", str(TENSOR_PARALLEL),
        "--gpu-memory-utilization", "0.90",
        "--enable-prefix-caching",  # the system prompt and a booklet repeat across claims
        "--api-key", os.environ["LLM_API_KEY"],
    ]
    subprocess.Popen(cmd)


@app.function(image=image, timeout=2 * 60 * 60, cpu=8, volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[modal.Secret.from_name("huggingface-secret")])
def download() -> None:
    """Fetch the weights on a CPU container so GPU time is not spent downloading."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(MODEL, token=os.environ.get("HF_TOKEN"))
    hf_cache.commit()
    print("weights at", path)


@app.local_entrypoint()
def smoke() -> None:
    """One chat request against the deployed (or ephemeral) server."""
    import json
    import time
    import urllib.request

    url = serve.get_web_url().rstrip("/") + "/v1/chat/completions"
    body = json.dumps({
        "model": MODEL, "max_tokens": 60, "temperature": 0,
        "messages": [{"role": "user", "content": "Antworte in einem Satz: Was ist ein Referendum?"}],
    }).encode()
    req = urllib.request.Request(url, body, {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['LLM_API_KEY']}",
    })
    start = time.time()
    with urllib.request.urlopen(req, timeout=40 * 60) as resp:
        out = json.load(resp)
    print(f"{time.time() - start:.1f}s", out["choices"][0]["message"]["content"], out["usage"])
