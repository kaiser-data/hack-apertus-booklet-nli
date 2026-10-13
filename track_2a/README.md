# Booklet NLI: claim checking over Swiss voting booklets with Apertus

**Hack Apertus Online 2026, Track 2A, OST challenge:** *Multilingual Natural Language Inference
over Swiss Official Voting Booklets.*

Give it an official federal voting booklet (PDF, German, French or Italian) and a claim in any of
the three languages. It answers **entailment (0)**, **neutral (1)** or **contradiction (2)**, and
shows the booklet passages (with page numbers) the answer rests on.

Apertus is the only model that makes the decision. A small multilingual embedding model picks
which passages Apertus gets to read (see [Models](#models)).

## Results

On the held-out test split (976 claims, 14 votes not used during development): **macro-F1 0.891**,
about 3,400 prompt tokens and a median 4.5 s Apertus response per claim. Giving Apertus the whole booklet instead
scores 0.735 on the same 60-claim subsample (vs 0.861 with retrieval) at 9× the tokens; part of
that gap comes from whole-booklet answers that never finish (a runaway evidence list; report §5.1).

- Visual summary: [`docs/results.html`](docs/results.html) (open locally in a browser)
- Technical report: [`kaiser-data_Report.pdf`](kaiser-data_Report.pdf), source [`technical_report.md`](technical_report.md)
- Raw predictions and metrics behind every number: [`runs/`](runs/)

## How it works

```
booklet PDF ──► passages (page + outline section) ──► top-k retrieval (cross-lingual) ──┐
claim ─────────────────────────────────────────────────────────────────────────────────┴─► Apertus ──► {label, evidence, reason}
```

1. **Passages.** PyMuPDF splits the booklet into passages of at most ~900 characters. Each one
   keeps its page number and the PDF outline path (proposal › part › subsection). Words broken
   across lines are rejoined.
2. **Context.** `--mode rag` (default) gives Apertus only the `k` passages closest to the claim.
   `--mode full` gives it the whole booklet, which is the baseline the challenge asks for.
3. **Verdict.** Apertus returns JSON (`reason`, `evidence` passage ids, `label`). Where the
   server supports it, the JSON shape is enforced with structured output.

Every prediction reports prompt tokens, completion tokens and wall time, so accuracy and cost can
be compared directly.

## Quick start

```bash
cp .env.example .env   # fill in LLM_BASE_URL and LLM_API_KEY
set -a; source .env; set +a   # the CLI reads the environment, not .env
uv sync
uv run booklet-nli predict \
  --booklet https://www.bk.admin.ch/dam/de/sd-web/UxUxfYD5ma0h/2021-06-13_erlaeuterungen_des_bundesrates.pdf \
  --claim "Le Conseil fédéral recommande de rejeter l'initiative sur l'eau potable."
```

### CLI

| Command | What it does |
|---|---|
| `booklet-nli predict --booklet <pdf/url> --claim "..."` | One verdict as JSON |
| `booklet-nli batch --input claims.jsonl --output preds.jsonl` | Rows with `booklet_url` and `claim` |
| `booklet-nli eval --split test --limit 60 --out runs/x` | Macro-F1, per-language scores, evidence hits, tokens, latency. Resumable |
| `booklet-nli retrieval --split dev` | Retrieval recall@k against the annotated reference passage. CPU only, no LLM |
| `booklet-nli compare runs/a runs/b` | Macro-F1 of two runs on their shared claims, with a paired bootstrap 95% interval |
| `booklet-nli rescore runs/a ...` | Recompute `metrics.json` (incl. bootstrap interval) from `predictions.jsonl` |

Options: `--mode rag|full`, `--k 16`, `--embedder <fastembed model>`.

### With Docker (what judges run)

```bash
export LLM_NAME=swiss-ai/Apertus-v1.5-8B LLM_BASE_URL=... LLM_API_KEY=...
make run                      # 60-claim sample of the test split (seed 0)
make run LIMIT=976 RUN=runs/test-fresh   # the whole test split, run fresh
make predict BOOKLET=<url> CLAIM="..."
make batch INPUT=data/claims.jsonl OUTPUT=runs/preds.jsonl
```

- `make run` needs a running Apertus endpoint (see below); it only runs the client.
- The same targets work from the repository root (`make run` forwards to `track_2a/`).
- Runs write to `runs/<split>-<mode>-k<K>-n<LIMIT>/`. An existing directory is resumed: rows
  that already have a prediction are skipped, and a directory made with different settings is
  refused. Pass `RUN=runs/<new name>` for a fresh run: `make run LIMIT=976` alone resumes the
  committed `runs/test-rag-k16-n976/`, finds every claim done and only recomputes its metrics.
- The CLI uses 8 parallel requests by default; the reported runs used 12 (`modal/run_eval.py`).

## Serving Apertus

Any OpenAI-compatible endpoint works, configured through `LLM_NAME`, `LLM_BASE_URL` and
`LLM_API_KEY`.

- **Modal (used for our results).** `modal/serve_apertus.py` runs the official
  `swiss-ai/Apertus-v1.5-8B` weights in swiss-ai's own vLLM image on one L40S. It scales to zero
  after 5 idle minutes. Needs the Modal secrets `huggingface-secret` (`HF_TOKEN`) and
  `apertus-nli-api-key` (`LLM_API_KEY`).
  ```bash
  make modal-deploy   # prints https://<workspace>--apertus-nli-serve.modal.run
  export LLM_BASE_URL=https://<workspace>--apertus-nli-serve.modal.run/v1
  ```
- **Local (laptop).** A community 4-bit GGUF of the same model runs in Ollama. From inside Docker,
  use `LLM_BASE_URL=http://host.docker.internal:11434/v1`.

## Models

| Role | Model | Where |
|---|---|---|
| NLI decision, evidence selection | `swiss-ai/Apertus-v1.5-8B` | GPU (vLLM) |
| Passage retrieval | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (via fastembed, ONNX) | CPU |

## Data

[`OSTswiss/MNLIoverSwissVotingBooklets`](https://huggingface.co/datasets/OSTswiss/MNLIoverSwissVotingBooklets)
(MIT): 1,488 claim/booklet pairs, balanced over labels and over all nine claim/booklet language
pairs. It ships without a split, so we split **by vote date**: the German, French and Italian
booklets of one vote are always on the same side. That gives a dev split (6 vote dates, 512
pairs) for prompt and retrieval choices, and a test split (14 vote dates, 976 pairs) for
reporting only. Booklet PDFs are downloaded from bk.admin.ch on first use and cached in
`data/cache/` (not committed).

## Development

```bash
uv sync --group dev
make test lint
```

## License

Apache-2.0, see [LICENSE](../LICENSE).
