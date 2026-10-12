"""The OST benchmark (OSTswiss/MNLIoverSwissVotingBooklets) and a leakage-free dev/test split."""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import httpx

from .config import CACHE_DIR

# Pinned to one dataset revision so results stay reproducible if the repo changes.
DATASET_REVISION = "fc2b27600310778da6bbf445651ddbca22d86269"
DATASET_URL = ("https://huggingface.co/datasets/OSTswiss/MNLIoverSwissVotingBooklets"
               f"/resolve/{DATASET_REVISION}/v1.1.jsonl")
DATASET_SHA256 = "87ee4afd9c0ee8861106c50f05884481402ca668aecf927df68fc52d915b02fd"
DEV_SHARE = 0.25


def load_rows(path: str | None = None) -> list[dict]:
    if path is None:
        target = CACHE_DIR / "MNLIoverSwissVotingBooklets-v1.1.jsonl"
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            resp = httpx.get(DATASET_URL, follow_redirects=True, timeout=120)
            resp.raise_for_status()
            if hashlib.sha256(resp.content).hexdigest() != DATASET_SHA256:
                raise ValueError(f"dataset checksum mismatch for {DATASET_URL}")
            target.write_bytes(resp.content)
        path = str(target)
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    for i, row in enumerate(rows):
        row.setdefault("id", i)
    return rows


def split_of(row: dict) -> str:
    """Split by vote date: the de/fr/it booklets of one vote never straddle dev and test."""
    digest = hashlib.sha1(row["booklet_publish_date"].encode()).digest()[0] / 255
    return "dev" if digest < DEV_SHARE else "test"


def select(rows: list[dict], split: str, limit: int | None, seed: int = 0) -> list[dict]:
    """Pick a split, then a sample stratified over (label, claim language, booklet language)."""
    rows = rows if split == "all" else [r for r in rows if split_of(r) == split]
    if limit is None or limit >= len(rows):
        return rows
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        strata[(r["entailment_label"], r["claim_language"], r["reference_language"])].append(r)
    rng = random.Random(seed)
    for bucket in strata.values():
        rng.shuffle(bucket)
    picked: list[dict] = []
    while len(picked) < limit:  # round-robin over strata keeps the sample balanced
        for key in sorted(strata):
            if strata[key] and len(picked) < limit:
                picked.append(strata[key].pop())
    return sorted(picked, key=lambda r: r["id"])
