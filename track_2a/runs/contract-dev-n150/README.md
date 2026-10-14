# Official-contract runs on dev (report §5.5)

150 dev claims (seed 0, the §5.2 sample) as task A and task B cases (`scripts/make_cases.py`),
scored with `evaluate.py` from https://gitlab.com/ifsoftware/hackapertus-starter (commit 5e69577).

| File | Run |
|---|---|
| `predictions-novote.jsonl` | entrypoint with `--no-vote`, passage-level evidence |
| `predictions-vote.jsonl` | with `vote`, passage-level evidence |
| `predictions-image.jsonl` | submitted Docker image on Modal (amd64): `vote` + sentence quotes |

Booklet PDFs are not included; `make_cases.py` downloads them into `booklets/`.
