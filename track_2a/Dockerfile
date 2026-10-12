FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /usr/local/bin/uv
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

# Booklets, parsed passages, embeddings and the embedder are cached under /app/data/cache;
# mount a volume there to keep them across runs.
ENV BOOKLET_NLI_DATA=/app/data
ENTRYPOINT ["/app/.venv/bin/booklet-nli"]
CMD ["--help"]
