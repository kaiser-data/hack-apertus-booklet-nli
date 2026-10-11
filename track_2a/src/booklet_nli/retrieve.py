"""Cross-lingual passage retrieval with a small multilingual embedding model (CPU, ONNX)."""

import hashlib
from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding

from .booklet import Passage
from .config import CACHE_DIR

DEFAULT_EMBEDDER = "intfloat/multilingual-e5-large"


@lru_cache(maxsize=2)
def _model(name: str) -> TextEmbedding:
    return TextEmbedding(name, cache_dir=str(CACHE_DIR / "embedders"))


def _prefix(name: str, kind: str) -> str:
    # e5 models are trained with "query: " / "passage: " prefixes.
    return f"{kind}: " if "e5" in name else ""


def _embed(name: str, texts: list[str], kind: str) -> np.ndarray:
    vecs = np.array(list(_model(name).embed([_prefix(name, kind) + t for t in texts])))
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def passage_matrix(passages: list[Passage], embedder: str) -> np.ndarray:
    key = hashlib.sha1(("\n".join(p.text for p in passages) + embedder).encode()).hexdigest()[:16]
    cache = CACHE_DIR / "embeddings" / f"{key}.npy"
    if cache.exists():
        return np.load(cache)
    # The section title disambiguates booklets that cover several votes.
    mat = _embed(embedder, [f"{p.section}\n{p.text}" for p in passages], "passage")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, mat)
    return mat


def rank(claim: str, passages: list[Passage], embedder: str = DEFAULT_EMBEDDER) -> list[Passage]:
    """All passages, most similar to the claim first."""
    scores = passage_matrix(passages, embedder) @ _embed(embedder, [claim], "query")[0]
    return [passages[i] for i in np.argsort(-scores)]


def top_k(claim: str, passages: list[Passage], k: int, embedder: str = DEFAULT_EMBEDDER) -> list[Passage]:
    """The k passages most similar to the claim, in document order."""
    return sorted(rank(claim, passages, embedder)[:k], key=lambda p: p.id)
