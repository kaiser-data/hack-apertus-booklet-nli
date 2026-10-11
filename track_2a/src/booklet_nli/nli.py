"""Document-grounded NLI: booklet + claim -> entailment / neutral / contradiction, with evidence."""

import json
import re
from dataclasses import dataclass, field

from . import LABELS
from .booklet import Passage, extract_passages, fetch_pdf
from .llm import LLM
from .retrieve import DEFAULT_EMBEDDER, top_k

SYSTEM = """You verify claims against an official Swiss federal voting booklet \
(Abstimmungsbüchlein / explications du Conseil fédéral / spiegazioni del Consiglio federale).

You get numbered passages from the booklet and one claim. The claim may be in a different \
language than the booklet. Judge ONLY the relationship between the claim and the passages, \
not whether the claim is true in the real world.

- entailment: the passages state or directly imply the claim, including who says it \
(Federal Council, Parliament, initiative committee, referendum committee).
- contradiction: the passages state something incompatible with the claim: a different \
number, date, recommendation, actor or consequence, or a negation.
- neutral: the passages do not settle the claim either way.

Attributing a statement to the wrong actor is a contradiction. Watch for negations and \
changed numbers.

Answer with JSON only: {"reason": "<one or two sentences>", "evidence": [<passage numbers>], \
"label": "entailment" | "neutral" | "contradiction"}. Evidence lists the passages your verdict \
rests on; use [] only for neutral claims with no related passage."""

SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "integer"}},
        "label": {"type": "string", "enum": list(LABELS)},
    },
    "required": ["reason", "evidence", "label"],
    "additionalProperties": False,
}

_JSON = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class Verdict:
    label: int  # dataset id: 0 entailment, 1 neutral, 2 contradiction
    reason: str
    evidence: list[Passage]
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    context_passages: int
    parsed: bool = True
    raw: str = field(repr=False, default="")

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "label_name": LABELS[self.label],
            "reason": self.reason,
            "evidence": [{"passage": p.id, "page": p.page, "section": p.section, "text": p.text}
                         for p in self.evidence],
            "usage": {"prompt_tokens": self.prompt_tokens,
                      "completion_tokens": self.completion_tokens,
                      "seconds": round(self.seconds, 3),
                      "context_passages": self.context_passages},
        }


def format_passages(passages: list[Passage]) -> str:
    return "\n\n".join(f"[{p.id}] (p. {p.page}{', ' + p.section if p.section else ''})\n{p.text}"
                       for p in passages)


def parse_verdict(text: str) -> tuple[int | None, str, list[int]]:
    """Parse the model's JSON; the label is None when the output has no valid label."""
    match = _JSON.search(text)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    label = str(data.get("label", "")).strip().lower()
    label_id = LABELS.index(label) if label in LABELS else None
    evidence = [int(e) for e in data.get("evidence", []) if str(e).lstrip("-").isdigit()]
    return label_id, str(data.get("reason", "")), evidence


def select_context(claim: str, passages: list[Passage], mode: str, k: int,
                   embedder: str) -> list[Passage]:
    if mode == "full":
        return passages
    if mode == "rag":
        return top_k(claim, passages, k, embedder)
    raise ValueError(f"unknown mode {mode!r}")


def verify(llm: LLM, booklet: str, claim: str, mode: str = "rag", k: int = 8,
           embedder: str = DEFAULT_EMBEDDER) -> Verdict:
    passages = extract_passages(fetch_pdf(booklet))
    context = select_context(claim, passages, mode, k, embedder)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Passages:\n\n{format_passages(context)}\n\nClaim: {claim}"},
    ]
    out = llm.complete(messages, max_tokens=300, json_schema=SCHEMA)
    label, reason, ids = parse_verdict(out.text)
    by_id = {p.id: p for p in context}
    # Unparseable output abstains as neutral and is counted as a parse failure.
    return Verdict(1 if label is None else label, reason, [by_id[i] for i in ids if i in by_id],
                   out.prompt_tokens, out.completion_tokens, out.seconds, len(context),
                   parsed=label is not None, raw=out.text)
