"""The Hack Apertus evaluation contract: `--input cases.jsonl --output predictions.jsonl`.

Each case has `id`, `vote`, `claim: {text, language}` and exactly one of
`booklet: {path, language}` (task A; path relative to the input file's directory, /data in the
container) or `reference: {text, language}` (task B). Every case gets one response line, also
when it fails, and the process exits 0.
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import LABELS
from .booklet import Passage, extract_passages
from .config import llm_config
from .llm import LLM, Completion
from .nli import judge
from .retrieve import DEFAULT_EMBEDDER, embed_scores, passage_matrix, rank

MAX_EVIDENCE = 5  # only the first five items are scored
MAX_EVIDENCE_CHARS = 5000  # longer items do not count
FROM_CITED = 3  # quotes from the passages Apertus cited, then from the rest of its context
_SENTENCE_END = re.compile(r"(?<=[.!?;:»])\s+(?=[A-ZÄÖÜÉÈÀ«\"(0-9])")
MIN_QUOTE_CHARS = 30
ATTEMPTS = 2
# Dev, 150 claims per task: 70B 0.925 / 0.979 vs 8B 0.857 / 0.930 (task A / B), same tokens.
PREFERRED_MODELS = ("swiss-ai/Apertus-v1.5-70B", "swiss-ai/Apertus-v1.5-8B")


def _sentences(p: Passage) -> list[tuple[int, str]]:
    parts = [s for s in _SENTENCE_END.split(p.text) if len(s) >= MIN_QUOTE_CHARS]
    return [(p.page, s) for s in parts or [p.text]]


def quotes(claim: str, cited: list[Passage], context: list[Passage], embedder: str) -> list[dict]:
    """Verbatim sentences as evidence: the ones closest to the claim from the cited passages,
    then from the other passages in context. Short quotes match a gold passage more reliably
    than whole passages, which can straddle its boundary."""
    def best(passages: list[Passage], n: int) -> list[tuple[int, str]]:
        sents = list(dict.fromkeys(s for p in dict.fromkeys(passages) for s in _sentences(p)))
        if not sents:
            return []
        scores = embed_scores(claim, [t for _, t in sents], embedder)
        return [sents[i] for i in sorted(range(len(sents)), key=lambda i: -scores[i])[:n]]

    picked = best(cited, FROM_CITED)
    rest = [p for p in context if p not in cited]
    picked += [s for s in best(rest, MAX_EVIDENCE) if s not in picked][:MAX_EVIDENCE - len(picked)]
    if len(picked) < MAX_EVIDENCE:  # few sentences outside the cited passages: use theirs
        picked += [s for s in best(cited, MAX_EVIDENCE) if s not in picked][:MAX_EVIDENCE - len(picked)]
    return [{"page": page, "text": text[:MAX_EVIDENCE_CHARS]} for page, text in picked]


def _booklet_path(case: dict, root: Path) -> Path:
    return root / case["booklet"]["path"]


class Booklets:
    """Parse and embed each booklet once, inside the worker threads, so that this work overlaps
    with other cases' Apertus calls instead of running before the first one. The official time
    score counts only wall-clock time with no LLM request in flight."""

    def __init__(self, embedder: str):
        self.embedder, self.guard, self.locks, self.done = embedder, threading.Lock(), {}, {}
        self.parse_lock = threading.Lock()  # PyMuPDF is not thread-safe: one document at a time

    def passages(self, path: Path) -> list[Passage]:
        with self.guard:
            lock = self.locks.setdefault(path, threading.Lock())
        with lock:
            if path not in self.done:
                with self.parse_lock:
                    ps = extract_passages(path)
                passage_matrix(ps, self.embedder)  # warm the embedding cache
                self.done[path] = ps
        return self.done[path]


class TimedLLM:
    """Wraps the LLM and records how long at least one request was in flight."""

    def __init__(self, llm: LLM):
        self.llm = llm
        self.lock, self.active, self.since, self.busy = threading.Lock(), 0, 0.0, 0.0

    @property
    def model(self) -> str:
        return self.llm.model

    def complete(self, *args, **kwargs) -> Completion:
        with self.lock:
            if self.active == 0:
                self.since = time.perf_counter()
            self.active += 1
        try:
            return self.llm.complete(*args, **kwargs)
        finally:
            with self.lock:
                self.active -= 1
                if self.active == 0:
                    self.busy += time.perf_counter() - self.since


def answer(llm: LLM, case: dict, root: Path, k: int, embedder: str, use_vote: bool,
           booklets: Booklets | None = None, glossary: bool = False) -> dict:
    start = time.perf_counter()
    claim = case["claim"]["text"]
    vote = case.get("vote") if use_vote else None
    if "booklet" in case:
        path = _booklet_path(case, root)
        passages = booklets.passages(path) if booklets else extract_passages(path)
        ranked = rank(claim, passages, embedder)
        context = sorted(ranked[:k], key=lambda p: p.id)
    else:
        ranked = context = [Passage(0, 0, "", case["reference"]["text"])]
    v = judge(llm, context, claim, vote, glossary)
    if "booklet" not in case:
        items = [{"page": None, "text": p.text[:MAX_EVIDENCE_CHARS]} for p in v.evidence[:1]]
    else:  # always quote: a wrong "neutral" can still cite the right sentence, and only the
        # first five items are scored. Without citations, start from the best-ranked passage.
        items = quotes(claim, v.evidence or ranked[:1], context, embedder)
    return _response(case["id"], v.label, items, v.prompt_tokens, v.completion_tokens, start)


def _response(case_id: str, label: int, evidence: list[dict], input_tokens: int,
              output_tokens: int, start: float) -> dict:
    return {"id": case_id, "label": label, "label_name": LABELS[label], "evidence": evidence,
            "metrics": {"input_tokens": input_tokens, "output_tokens": output_tokens,
                        "inference_time_ms": round((time.perf_counter() - start) * 1000)}}


def _safe_answer(llm: LLM, case: dict, root: Path, k: int, embedder: str, use_vote: bool,
                 booklets: Booklets, glossary: bool = False) -> dict:
    start = time.perf_counter()
    for attempt in range(ATTEMPTS):  # the OpenAI client itself retries 429/5xx with backoff
        try:
            return answer(llm, case, root, k, embedder, use_vote, booklets, glossary)
        except Exception as exc:  # noqa: BLE001 - one bad case must not lose the others
            print(f"{case.get('id')}: attempt {attempt + 1} failed: {exc!r}", file=sys.stderr)
            time.sleep(5)
    return _response(str(case.get("id")), 1, [], 0, 0, start)  # abstain as neutral


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="booklet-nli-cases", description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path,
                        help="directory booklet paths are relative to (default: the input's)")
    parser.add_argument("--k", type=int, default=16, help="passages retrieved per claim")
    parser.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--no-vote", action="store_true", help="leave the vote name out")
    parser.add_argument("--actor-glossary", action="store_true",
                        help="add the cross-lingual actor glossary to the prompt (experimental)")
    args = parser.parse_args(argv)

    # Split on "\n" only: str.splitlines() also breaks at U+2028 inside JSON strings.
    cases = [json.loads(line) for line in args.input.read_text(encoding="utf-8").split("\n")
             if line.strip()]
    root = args.data_root or args.input.resolve().parent
    started = time.perf_counter()
    base = LLM(llm_config(), timeout=120)
    if not os.environ.get("LLM_NAME"):  # default: the 70B if the endpoint serves it (§5.5)
        base.pick(list(PREFERRED_MODELS))
    llm = TimedLLM(base)
    booklets = Booklets(args.embedder)
    print(f"{len(cases)} cases, model {llm.model}", file=sys.stderr)

    def run(case: dict) -> dict:
        return _safe_answer(llm, case, root, args.k, args.embedder, not args.no_vote, booklets,
                            args.actor_glossary)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.workers) as pool, args.output.open("w", encoding="utf-8") as out:
        for n, fut in enumerate(as_completed([pool.submit(run, c) for c in cases]), 1):
            out.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
            out.flush()
            if n % 20 == 0:
                print(f"  {n}/{len(cases)}", file=sys.stderr)
    wall = time.perf_counter() - started
    print(f"wrote {len(cases)} predictions to {args.output}; {wall:.1f} s wall, of which "
          f"~{wall - llm.busy:.1f} s with no LLM request in flight (approximate) "
          f"({(wall - llm.busy) / max(len(cases), 1) * 1000:.0f} ms per case)", file=sys.stderr)


if __name__ == "__main__":
    main()
