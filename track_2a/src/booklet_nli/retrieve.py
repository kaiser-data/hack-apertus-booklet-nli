"""Cross-lingual passage retrieval with a small multilingual embedding model (CPU, ONNX)."""

import hashlib
import re
import threading
from collections import Counter
from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding

from .booklet import Passage
from .config import CACHE_DIR, EMBEDDER_DIR

DEFAULT_EMBEDDER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


_MODEL_LOCK = threading.Lock()


@lru_cache(maxsize=2)
def _load(name: str) -> TextEmbedding:
    return TextEmbedding(name, cache_dir=str(EMBEDDER_DIR))


def _model(name: str) -> TextEmbedding:
    with _MODEL_LOCK:  # lru_cache alone lets concurrent first calls load several copies
        return _load(name)


_QWEN_TASK = "Given a claim about a Swiss vote, retrieve booklet passages that support or refute it"


def _prefix(name: str, kind: str) -> str:
    """The query/document prompts each model was trained with."""
    if "e5" in name:
        return f"{kind}: "
    if "embeddinggemma" in name:
        return "task: fact checking | query: " if kind == "query" else "title: none | text: "
    if "Qwen3-Embedding" in name:
        return f"Instruct: {_QWEN_TASK}\nQuery: " if kind == "query" else ""
    return ""


def _embed(name: str, texts: list[str], kind: str) -> np.ndarray:
    vecs = np.array(list(_model(name).embed([_prefix(name, kind) + t for t in texts])))
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def passage_matrix(passages: list[Passage], embedder: str) -> np.ndarray:
    # The section title disambiguates booklets that cover several votes.
    texts = [f"{p.section}\n{p.text}" for p in passages]
    key = hashlib.sha1(("\x00".join(texts) + embedder).encode()).hexdigest()[:16]
    cache = CACHE_DIR / "embeddings" / f"{key}.npy"
    if cache.exists():
        return np.load(cache)
    mat = _embed(embedder, texts, "passage")
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(f".{threading.get_ident()}.tmp.npy")
    np.save(tmp, mat)
    tmp.replace(cache)  # atomic: a concurrent reader never sees a half-written file
    return mat


def embed_scores(claim: str, texts: list[str], embedder: str = DEFAULT_EMBEDDER) -> np.ndarray:
    """Cosine similarity of each text to the claim (texts embedded as passages)."""
    return _embed(embedder, texts, "passage") @ _embed(embedder, [claim], "query")[0]


def rank(claim: str, passages: list[Passage], embedder: str = DEFAULT_EMBEDDER) -> list[Passage]:
    """All passages, most similar to the claim first."""
    scores = passage_matrix(passages, embedder) @ _embed(embedder, [claim], "query")[0]
    return [passages[i] for i in np.argsort(-scores)]


def top_k(claim: str, passages: list[Passage], k: int, embedder: str = DEFAULT_EMBEDDER) -> list[Passage]:
    """The k passages most similar to the claim, in document order."""
    return sorted(rank(claim, passages, embedder)[:k], key=lambda p: p.id)


_TOKEN = re.compile(r"\w+")
RRF_K = 60  # standard reciprocal-rank-fusion constant (Cormack et al., 2009)


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def bm25_scores(query: str, texts: list[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """Okapi BM25 of the query against each text. Catches exact numbers, dates and names, which
    the dense embedder may blur; across languages it still matches numbers and proper nouns."""
    docs = [Counter(_tokens(t)) for t in texts]
    lengths = np.array([sum(d.values()) for d in docs], dtype=float)
    avg = lengths.mean() if len(docs) else 1.0
    scores = np.zeros(len(docs))
    for term in set(_tokens(query)):
        df = sum(1 for d in docs if term in d)
        if not df:
            continue
        idf = np.log(1 + (len(docs) - df + 0.5) / (df + 0.5))
        tf = np.array([d.get(term, 0) for d in docs], dtype=float)
        scores += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lengths / avg))
    return scores


def rrf(*rankings: np.ndarray) -> np.ndarray:
    """Fuse score arrays by reciprocal rank; returns indices, best first."""
    fused = np.zeros(len(rankings[0]))
    for scores in rankings:
        ranks = np.empty(len(scores), dtype=int)
        ranks[np.argsort(-scores, kind="stable")] = np.arange(len(scores))
        fused += 1.0 / (RRF_K + ranks + 1)
    return np.argsort(-fused, kind="stable")
