"""Macro-F1, per-language breakdowns, evidence overlap and efficiency numbers."""

import re
from collections import defaultdict
from statistics import mean, median

from . import LABELS

_WORD = re.compile(r"\w+")


def macro_f1(gold: list[int], pred: list[int]) -> float:
    f1s = []
    for c in range(len(LABELS)):
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(f1s) / len(f1s)


def confusion(gold: list[int], pred: list[int]) -> list[list[int]]:
    m = [[0] * len(LABELS) for _ in LABELS]
    for g, p in zip(gold, pred):
        m[g][p] += 1
    return m


def _ngrams(text: str, n: int = 5) -> set[tuple[str, ...]]:
    words = _WORD.findall(text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def overlaps_reference(passage_text: str, reference: str, threshold: float = 0.5) -> bool:
    """A passage 'hits' if most of its word 5-grams occur in the annotated reference snippet."""
    grams = _ngrams(passage_text)
    return bool(grams) and len(grams & _ngrams(reference)) / len(grams) >= threshold


def summarize(records: list[dict]) -> dict:
    gold = [r["gold"] for r in records]
    pred = [r["pred"] for r in records]
    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_group[f"claim_{r['claim_language']}"].append(r)
        by_group[f"booklet_{r['reference_language']}"].append(r)
        same = r["claim_language"] == r["reference_language"]
        by_group["monolingual" if same else "cross_lingual"].append(r)
    with_gold_evidence = [r for r in records if r["gold"] != 1]
    return {
        "n": len(records),
        "macro_f1": round(macro_f1(gold, pred), 4),
        "accuracy": round(sum(g == p for g, p in zip(gold, pred)) / len(records), 4),
        "confusion": {"rows_gold_cols_pred": list(LABELS), "matrix": confusion(gold, pred)},
        "macro_f1_by_group": {
            k: {"n": len(v), "macro_f1": round(macro_f1([r["gold"] for r in v],
                                                        [r["pred"] for r in v]), 4)}
            for k, v in sorted(by_group.items())},
        "evidence_hit_rate": round(mean(r["evidence_hit"] for r in with_gold_evidence), 4)
        if with_gold_evidence else None,
        "context_recall": round(mean(r["context_hit"] for r in with_gold_evidence), 4)
        if with_gold_evidence else None,
        "prompt_tokens": {"mean": round(mean(r["prompt_tokens"] for r in records)),
                          "median": median(r["prompt_tokens"] for r in records)},
        "seconds": {"mean": round(mean(r["seconds"] for r in records), 2),
                    "median": round(median(r["seconds"] for r in records), 2)},
        "parse_failures": sum(r["parse_failed"] for r in records),
    }
