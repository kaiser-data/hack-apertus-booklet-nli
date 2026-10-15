# Official-contract runs on dev (report §5.5)

150 dev claims (seed 0, the §5.2 sample) as task A and task B cases (`scripts/make_cases.py`),
scored with `evaluate.py` from https://gitlab.com/ifsoftware/hackapertus-starter (commit 5e69577).

| File | Run |
|---|---|
| `predictions-novote.jsonl` | entrypoint with `--no-vote`, passage-level evidence |
| `predictions-vote.jsonl` | with `vote`, passage-level evidence |
| `predictions-image.jsonl` | Docker image on Modal (amd64): `vote` + sentence quotes |
| `predictions-image-8b-final.jsonl` | after the review fixes (lazy booklet preparation, deduplicated quotes, bounded retries) |
| `predictions-image-70b.jsonl` | same image, `LLM_NAME=swiss-ai/Apertus-v1.5-70B` (Modal, 2× H200) |
| `predictions-image-auto.jsonl` | same image, `LLM_NAME` unset against the 70B server: the entrypoint picked the 70B |
| `fallback-8b-examples.jsonl` | `examples/` against the 8B-only server, `LLM_NAME` unset: fell back to the 8B |

Booklet PDFs are not included; `make_cases.py` downloads them into `booklets/`.

The starter's `evaluate.py` prints a task B minimum of 0.70; the organisers' slides (8 Oct) say 0.75.
