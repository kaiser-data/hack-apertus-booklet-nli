# What this project took, and from whom

One entry per debt. `paper/refs.bib` is the bibliography; this file says what each source gave us.

### OST (2026): MNLI over Swiss Voting Booklets v1.1, the benchmark

> https://huggingface.co/datasets/OSTswiss/MNLIoverSwissVotingBooklets, revision `fc2b276`, MIT.

**Used in:** `src/booklet_nli/data.py`, every run in `runs/`, report §2 and §4.
**Taken:** claims, labels, reference passages, booklet URLs, vote names.
**Not taken:** the dataset ships no split; the dev/test split by vote date is ours (`data.py`).
The `baseline_score` column is never used.

### OST AI Applications & Deployment Lab (2026): hackapertus-starter

> https://gitlab.com/ifsoftware/hackapertus-starter, commit `5e69577`.

**Used in:** `src/booklet_nli/cases.py` (input/output contract), `scripts/make_cases.py` (case format
and ids mirror `prepare_cases.py`), all "official" dev scores (`evaluate.py`, run unmodified),
`examples/cases.jsonl` (copied from the starter's examples).
**Not taken:** no starter code is in this repository; `make_cases.py` is a reimplementation that reads
the pinned dataset revision and our split.

### Purandare & Kuganathan (8 Oct 2026): Track 2A Q&A slides

**Used in:** report §2 (scoring weights, validity thresholds, the processing-time definition) and
the design choices that depend on them (lazy preparation, 70B default, keeping MiniLM).

### Swiss AI Initiative: Apertus-v1.5-8B and Apertus-v1.5-70B

> https://huggingface.co/swiss-ai/Apertus-v1.5-8B · https://huggingface.co/swiss-ai/Apertus-v1.5-70B
> (snapshot `94888dd`); served with swiss-ai's vLLM image (`modal/serve_apertus*.py`, digest pinned).

**Used in:** every label and citation the system produces. No fine-tuning.

### Reimers & Gurevych (2020): paraphrase-multilingual-MiniLM-L12-v2

**Used in:** `src/booklet_nli/retrieve.py` (retrieval and sentence ranking for evidence).

### Wang et al. (2024) multilingual-e5-large; Zhang et al. (2025) Qwen3-Embedding-0.6B

**Used in:** `modal/embed_bench.py` only (benchmarked, not used by the submission). Query/document
prefixes follow the model cards. The arXiv identifiers in `refs.bib` (2402.05672, 2506.05176) were
not re-opened in this session; verify before archival publication.

### Robertson & Zaragoza (2009) BM25; Cormack et al. (2009) reciprocal rank fusion

**Used in:** `retrieve.py::bm25_scores`, `retrieve.py::rrf` (k1 = 1.5, b = 0.75, RRF constant 60, the
standard values), benchmark only; not used by the submission.

### Kwon et al. (2023): vLLM

**Used in:** the Apertus servers on Modal.

### Swiss Federal Chancellery: voting booklets

> https://www.bk.admin.ch/de/sammlung-der-abstimmungsbuechlein-seit-1978

**Used in:** every booklet the system reads; `examples/booklets/2024_09_22_de.pdf` is a copy of an
official booklet, for the image's example run.

## AI assistance during development

| Tool | What it did | How its output was checked |
|---|---|---|
| Claude Code (Anthropic, Claude Opus 5.5) | wrote most code, scripts and documentation drafts under the author's direction; ran experiments on Modal | tests and lint; every reported number recomputed from committed run files |
| Codex CLI (OpenAI) | five read-only reviews: 12 Oct (data; results), 13 Oct (overclaims), 15 Oct (contract; final) | each finding checked against the code before acting; rejected findings stated in commit messages |
| Grok (xAI) | two read-only reviews, 12 Oct (data; results and improvements) | as above; the date-clustered bootstrap it suggested is in the archival report |
| *(author to confirm)* | an external review pasted on 15 Oct | |
