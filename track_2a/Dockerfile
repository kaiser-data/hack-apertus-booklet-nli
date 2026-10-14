# Built for the official evaluation: linux/amd64, CPU only, no downloads at runtime.
#   docker build --platform linux/amd64 -t booklet-nli .
#   docker run --rm -e BASE_URL -e API_KEY -v "$PWD/cases:/data:ro" -v "$PWD/out:/output" \
#     booklet-nli --input /data/cases.jsonl --output /output/predictions.jsonl
FROM --platform=linux/amd64 python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_PYTHON_DOWNLOADS=never

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

# Bake the retrieval embedder into the image; at runtime it is read offline.
ENV BOOKLET_NLI_EMBEDDERS=/app/models
RUN /app/.venv/bin/python -c "from booklet_nli.retrieve import _embed, DEFAULT_EMBEDDER; \
_embed(DEFAULT_EMBEDDER, ['warm-up'], 'query')"
ENV HF_HUB_OFFLINE=1

# /data is read-only in the evaluation; parsed passages and embeddings are cached in /tmp.
ENV BOOKLET_NLI_DATA=/tmp/booklet-nli
ENTRYPOINT ["/app/.venv/bin/booklet-nli-cases"]
