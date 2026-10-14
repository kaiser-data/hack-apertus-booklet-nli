"""The Hack Apertus evaluation contract: `--input cases.jsonl --output predictions.jsonl`.

Each case has `id`, `vote`, `claim: {text, language}` and exactly one of
`booklet: {path, language}` (task A; path relative to the input file's directory, /data in the
container) or `reference: {text, language}` (task B). Every case gets one response line, also
when it fails, and the process exits 0.
"""

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import LABELS
from .booklet import Passage, extract_passages
from .config import llm_config
from .llm import LLM
from .nli import judge
from .retrieve import DEFAULT_EMBEDDER, embed_scores, passage_matrix, rank

MAX_EVIDENCE = 5  # only the first five items are scored
MAX_EVIDENCE_CHARS = 5000  # longer items do not count
FROM_CITED = 3  # quotes from the passages Apertus cited, then from the rest of its context
_SENTENCE_END = re.compile(r"(?<=[.!?;:»])\s+(?=[A-ZÄÖÜÉÈÀ«\"(0-9])")
MIN_QUOTE_CHARS = 30


def _sentences(p: Passage) -> list[tuple[int, str]]:
    parts = [s for s in _SENTENCE_END.split(p.text) if len(s) >= MIN_QUOTE_CHARS]
    return [(p.page, s) for s in parts or [p.text]]


def quotes(claim: str, cited: list[Passage], context: list[Passage], embedder: str) -> list[dict]:
    """Verbatim sentences as evidence: the ones closest to the claim from the cited passages,
    then from the other passages in context. Short quotes match a gold passage more reliably
    than whole passages, which can straddle its boundary."""
    def best(passages: list[Passage], n: int) -> list[tuple[int, str]]:
        sents = [s for p in passages for s in _sentences(p)]
        if not sents:
            return []
        scores = embed_scores(claim, [t for _, t in sents], embedder)
        return [sents[i] for i in sorted(range(len(sents)), key=lambda i: -scores[i])[:n]]

    picked = best(cited, FROM_CITED)
    rest = [p for p in context if p not in cited]
    picked += [s for s in best(rest, MAX_EVIDENCE) if s not in picked][:MAX_EVIDENCE - len(picked)]
    return [{"page": page, "text": text[:MAX_EVIDENCE_CHARS]} for page, text in picked]


def _booklet_path(case: dict, root: Path) -> Path:
    return root / case["booklet"]["path"]


def answer(llm: LLM, case: dict, root: Path, k: int, embedder: str, use_vote: bool) -> dict:
    start = time.perf_counter()
    claim = case["claim"]["text"]
    vote = case.get("vote") if use_vote else None
    if "booklet" in case:
        ranked = rank(claim, extract_passages(_booklet_path(case, root)), embedder)
        context = sorted(ranked[:k], key=lambda p: p.id)
    else:
        ranked = context = [Passage(0, 0, "", case["reference"]["text"])]
    v = judge(llm, context, claim, vote)
    if "booklet" not in case:
        items = [{"page": None, "text": p.text[:MAX_EVIDENCE_CHARS]} for p in v.evidence[:1]]
    elif v.label == 1 and not v.evidence:
        items = []
    else:  # a verdict needs a citation: without one, start from the best-ranked passage
        items = quotes(claim, v.evidence or ranked[:1], context, embedder)
    return _response(case["id"], v.label, items, v.prompt_tokens, v.completion_tokens, start)


def _response(case_id: str, label: int, evidence: list[dict], input_tokens: int,
              output_tokens: int, start: float) -> dict:
    return {"id": case_id, "label": label, "label_name": LABELS[label], "evidence": evidence,
            "metrics": {"input_tokens": input_tokens, "output_tokens": output_tokens,
                        "inference_time_ms": round((time.perf_counter() - start) * 1000)}}


def _safe_answer(llm: LLM, case: dict, root: Path, k: int, embedder: str, use_vote: bool) -> dict:
    start = time.perf_counter()
    for attempt in range(3):
        try:
            return answer(llm, case, root, k, embedder, use_vote)
        except Exception as exc:  # noqa: BLE001 - one bad case must not lose the others
            print(f"{case.get('id')}: attempt {attempt + 1} failed: {exc!r}", file=sys.stderr)
            time.sleep(2 ** attempt)
    return _response(str(case.get("id")), 1, [], 0, 0, start)  # abstain as neutral


def _prepare(cases: list[dict], root: Path, embedder: str) -> None:
    """Parse and embed each booklet once, before worker threads share the caches."""
    for path in sorted({_booklet_path(c, root) for c in cases if "booklet" in c}):
        try:
            passage_matrix(extract_passages(path), embedder)
        except Exception as exc:  # noqa: BLE001 - reported again per case
            print(f"{path}: {exc!r}", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="booklet-nli-cases", description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path,
                        help="directory booklet paths are relative to (default: the input's)")
    parser.add_argument("--k", type=int, default=16, help="passages retrieved per claim")
    parser.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-vote", action="store_true", help="leave the vote name out")
    args = parser.parse_args(argv)

    cases = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    root = args.data_root or args.input.resolve().parent
    llm = LLM(llm_config(), timeout=300)
    print(f"{len(cases)} cases, model {llm.model}", file=sys.stderr)
    _prepare(cases, root, args.embedder)

    def run(case: dict) -> dict:
        return _safe_answer(llm, case, root, args.k, args.embedder, not args.no_vote)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.workers) as pool, args.output.open("w", encoding="utf-8") as out:
        for n, result in enumerate(pool.map(run, cases), 1):
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            if n % 20 == 0:
                print(f"  {n}/{len(cases)}", file=sys.stderr)
    print(f"wrote {len(cases)} predictions to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
