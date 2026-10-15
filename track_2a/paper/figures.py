"""Figures for paper/report.tex, computed from the committed run files only.

    uv run --with matplotlib python paper/figures.py      # writes paper/fig-*.pdf, prints the numbers

Palette: the dataviz reference categorical slots 1-3 (blue, orange, aqua), which pass the
all-pairs colour-vision checks; every mark is also directly labelled, so print in grey works.
"""

import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from booklet_nli.metrics import bootstrap_ci, macro_f1

RUNS = ROOT / "runs"
OUT = Path(__file__).resolve().parent
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "pdf.fonttype": 42, "legend.frameon": False,
})


def rows(path: Path) -> dict:
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}


def r3(x: float) -> str:
    """Round like the report: 4 stored digits, then half-up to 3."""
    return str(Decimal(str(round(x, 4))).quantize(Decimal("0.001"), ROUND_HALF_UP))


def fig_context():
    """Whole booklet vs 16 retrieved passages, the same 60 held-out test claims (8B)."""
    full, rag = rows(RUNS / "test-full-n60/predictions.jsonl"), rows(RUNS / "test-rag-k16-n976/predictions.jsonl")
    ids = sorted(full)
    gold = [full[i]["gold"] for i in ids]
    pts = []
    for name, run, color in (("Whole booklet", full, ORANGE), ("16 retrieved passages", rag, BLUE)):
        pred = [run[i]["pred"] for i in ids]
        tok = sum(run[i]["prompt_tokens"] for i in ids) / len(ids)
        f1, (lo, hi) = macro_f1(gold, pred), bootstrap_ci(gold, pred)
        pts.append((name, tok, f1, lo, hi, color))
        print(f"context: {name}: tokens {tok:.0f}, F1 {r3(f1)} [{r3(lo)}, {r3(hi)}], n={len(ids)}")
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for name, tok, f1, lo, hi, color in pts:
        ax.errorbar(tok, f1, yerr=[[f1 - lo], [hi - f1]], fmt="o", ms=6, color=color,
                    ecolor=color, elinewidth=1.4, capsize=3, mec="white", mew=1)
        right = tok < 10000
        ax.annotate(f"{name}\nF1 {r3(f1)}, {tok:,.0f} tokens", (tok, f1), textcoords="offset points",
                    xytext=(10 if right else -10, -4), ha="left" if right else "right", va="center",
                    color=INK, fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlim(1500, 80000)
    ax.xaxis.set_major_locator(FixedLocator([2000, 5000, 10000, 20000, 50000]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("Prompt tokens per claim (log scale)")
    ax.set_ylabel("Macro-F1 (95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(OUT / "fig-context.pdf")


def score(path: Path) -> dict:
    return json.loads(path.read_text())["tasks"]


def fig_evidence_and_embedders():
    """(a) evidence granularity under the official matcher; (b) embedder recall vs CPU cost."""

    variants = [("Whole pages", "whole-pages"), ("~900-char passages", "passages"),
                ("Sentence quotes", "sentences")]
    hits = []
    for label, name in variants:
        s = json.loads((RUNS / f"contract-dev-n150/evidence-variants/score-{name}.json").read_text())
        hits.append((label, s["tasks"]["A"]["evidence_score"], s["tasks"]["A"]["evidence"]["found"]))
    bench = json.loads((RUNS / "embed-bench-dev.json").read_text())
    DY = {"Qwen3-Embedding-0.6B-Q": 10, "multilingual-e5-large": -12}  # keep the two labels apart
    names = {"paraphrase-multilingual-MiniLM-L12-v2": "MiniLM (used)",
             "multilingual-e5-large": "e5-large", "Qwen3-Embedding-0.6B-Q": "Qwen3-Emb.-0.6B"}

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 2.15), gridspec_kw={"width_ratios": [1, 1.05]})
    y = range(len(hits))
    a.barh(list(y), [h for _, h, _ in hits], color=[MUTED, MUTED, BLUE], height=0.55)
    for i, (label, h, found) in enumerate(hits):
        a.text(h + 0.01, i, f"{r3(h)}  ({found}/99)", va="center", fontsize=7.5, color=INK)
        print(f"evidence: {label}: Hit@5 {r3(h)} ({found}/99)")
    a.set_yticks(list(y), [label for label, _, _ in hits])
    a.set_xlim(0, 1)
    a.set_xlabel("Evidence Hit@5 (organisers' matcher)")
    a.set_title("(a) Three evidence configurations", loc="left")
    a.grid(axis="y", visible=False)

    for r in bench:
        short = r["embedder"].split("/")[-1]
        x, rec = r["embed_s_per_booklet"], r["recall_at_k"]["16"]
        color = BLUE if "MiniLM" in short else AQUA
        b.scatter(x, rec, s=36, color=color, edgecolor="white", linewidth=1, zorder=3)
        left = "MiniLM" not in short
        b.annotate(f"{names.get(short, short)}\n{r3(rec)}, {x:.0f} s", (x, rec),
                   textcoords="offset points",
                   xytext=(-8 if left else 8, DY.get(short, 0)),
                   ha="right" if left else "left", va="center", fontsize=7.5, color=INK)
        print(f"embedder: {short}: recall@16 {r3(rec)}, {x:.1f} s/booklet")
    b.set_xscale("log")
    b.set_xlim(1.5, 500)
    b.set_ylim(0.89, 0.985)
    b.xaxis.set_major_locator(FixedLocator([3, 10, 30, 100, 300]))
    b.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    b.xaxis.set_minor_locator(NullLocator())
    b.set_xlabel("CPU seconds per booklet (4 vCPU, log)")
    b.set_ylabel("Recall@16")
    b.set_title("(b) Retrieval quality vs CPU cost", loc="left")
    fig.tight_layout(w_pad=2.5)
    fig.savefig(OUT / "fig-evidence-embedders.pdf")


ERROR_CATEGORIES = ["retrieval miss", "actor-mentioning claim", "claim with a number", "negation",
                    "other reasoning", "over-inference (gold neutral)"]


def error_category(p: dict, claim: str) -> str:
    """First matching rule; transparent so the taxonomy can be audited (no model judges here)."""
    import re
    if p["gold"] != 1 and not p["context_hit"]:
        return "retrieval miss"
    if p["gold"] == 1:
        return "over-inference (gold neutral)"
    if re.search(r"(?i)komitee|comit[ée]|comitato|bundesrat|conseil f[ée]d[ée]ral|consiglio federale"
                 r"|parlament|parlement|parlamento", claim):
        return "actor-mentioning claim"
    if re.search(r"\d", claim):
        return "claim with a number"
    if re.search(r"(?i)\b(nicht|kein\w*|nie|ne|pas|aucun\w*|jamais|non|nessun\w*|mai)\b", claim):
        return "negation"
    return "other reasoning"


def fig_errors():
    """Where the 8B fails on the 976 test claims, per 100 claims of each language group."""
    from collections import Counter

    from booklet_nli.data import load_rows

    claims = {r["id"]: r["claim"] for r in load_rows()}
    preds = list(rows(RUNS / "test-rag-k16-n976/predictions.jsonl").values())
    cross = lambda p: p["claim_language"] != p["reference_language"]
    size = Counter(cross(p) for p in preds)
    errs = Counter((error_category(p, claims[p["id"]]), cross(p)) for p in preds if p["gold"] != p["pred"])
    fig, ax = plt.subplots(figsize=(6.9, 1.95))
    y = list(range(len(ERROR_CATEGORIES)))[::-1]
    h = 0.36
    for k, (is_cross, color, label) in enumerate(((False, BLUE, f"same language (n = {size[False]})"),
                                                   (True, ORANGE, f"cross-lingual (n = {size[True]})"))):
        vals = [100 * errs[(c, is_cross)] / size[is_cross] for c in ERROR_CATEGORIES]
        pos = [yy + (h / 2 if k == 0 else -h / 2) for yy in y]
        ax.barh(pos, vals, height=h, color=color, label=label, edgecolor="white", linewidth=1)
        for yy, v, c in zip(pos, vals, ERROR_CATEGORIES):
            ax.text(v + 0.05, yy, f"{v:.1f}  ({errs[(c, is_cross)]})", va="center", fontsize=7, color=INK)
            print(f"errors: {c} {'cross' if is_cross else 'same'}: {errs[(c, is_cross)]} = {v:.1f}/100")
    ax.set_yticks(y, ERROR_CATEGORIES)
    ax.set_xlabel("Errors per 100 claims of the group (8B, test split; count in brackets)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 4.6)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig-errors.pdf")


if __name__ == "__main__":
    fig_context()
    fig_evidence_and_embedders()
    fig_errors()
