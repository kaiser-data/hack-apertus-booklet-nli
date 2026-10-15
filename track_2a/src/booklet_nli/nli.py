"""Document-grounded NLI: booklet + claim -> entailment / neutral / contradiction, with evidence."""

import json
import re
from dataclasses import dataclass, field

from . import LABELS
from .booklet import Passage, extract_passages, fetch_pdf
from .llm import LLM
from .retrieve import DEFAULT_EMBEDDER, top_k

# Output cap, used for all reported runs. Raising it to 600 did not help: answers that hit it are
# runaway evidence lists (see technical_report.md §5.1).
MAX_TOKENS = 300

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

# Optional (--actor-glossary): the main cross-lingual error was a claim naming "the committee" or
# the Federal Council in one language while the booklet names the body in another (report §6).
ACTOR_GLOSSARY = """

Actors are named differently in German, French and Italian; these are the same actor:
- Federal Council = Bundesrat = Conseil fédéral = Consiglio federale
- Parliament = Parlament / Bundesversammlung = Parlement / Assemblée fédérale = Parlamento / Assemblea federale
- initiative committee = Initiativkomitee = comité d'initiative = comitato d'iniziativa
- referendum committee = Referendumskomitee = comité référendaire = comitato referendario
A claim about "the committee" (das Komitee, le comité, il comitato) means the committee whose \
arguments the booklet prints for this vote. Decide who holds a position before deciding the label."""

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
    def where(p: Passage) -> str:  # page 0: a reference text given without a page
        parts = ([f"p. {p.page}"] if p.page else []) + ([p.section] if p.section else [])
        return f" ({', '.join(parts)})" if parts else ""

    return "\n\n".join(f"[{p.id}]{where(p)}\n{p.text}" for p in passages)


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


def verify(llm: LLM, booklet: str, claim: str, mode: str = "rag", k: int = 16,
           embedder: str = DEFAULT_EMBEDDER, vote: str | None = None) -> Verdict:
    passages = extract_passages(fetch_pdf(booklet))
    return judge(llm, select_context(claim, passages, mode, k, embedder), claim, vote)


def judge(llm: LLM, context: list[Passage], claim: str, vote: str | None = None,
          glossary: bool = False) -> Verdict:
    """One Apertus call: the claim against the given passages. `vote` names the proposal."""
    about = f"Vote: {vote}\n" if vote else ""
    messages = [
        {"role": "system", "content": SYSTEM + (ACTOR_GLOSSARY if glossary else "")},
        {"role": "user",
         "content": f"Passages:\n\n{format_passages(context)}\n\n{about}Claim: {claim}"},
    ]
    out = llm.complete(messages, max_tokens=MAX_TOKENS, json_schema=SCHEMA)
    label, reason, ids = parse_verdict(out.text)
    by_id = {p.id: p for p in context}
    # Unparseable output abstains as neutral and is counted as a parse failure.
    return Verdict(1 if label is None else label, reason, [by_id[i] for i in ids if i in by_id],
                   out.prompt_tokens, out.completion_tokens, out.seconds, len(context),
                   parsed=label is not None, raw=out.text)
