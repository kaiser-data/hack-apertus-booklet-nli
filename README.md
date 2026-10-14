# Booklet NLI — checking claims against Swiss voting booklets with Apertus

**Hack Apertus · Track 2A submission.** Give it an official federal voting booklet (German, French
or Italian) and a claim in any of the three languages. Apertus-v1.5-8B says whether the booklet
**supports**, **contradicts** or **doesn't settle** the claim, and cites the pages it relied on.

- **Held-out test (976 claims, 14 unseen vote dates): macro-F1 0.891** (95% CI 0.871–0.909),
  about 3,400 prompt tokens per claim, 9× fewer than reading the whole booklet.
- **Container image (amd64, public):** `ghcr.io/kaiser-data/hack-apertus-booklet-nli:latest`, built
  from `track_2a/Dockerfile` by [CI](.github/workflows/image.yml), which also runs the contract
  offline before every push. `docker run --platform linux/amd64 -e BASE_URL -e API_KEY -v
  <cases>:/data:ro -v <out>:/output ghcr.io/kaiser-data/hack-apertus-booklet-nli:latest --input
  /data/cases.jsonl --output /output/predictions.jsonl`
- **Official evaluation contract:** the Docker image takes `--input /data/cases.jsonl --output
  /output/predictions.jsonl` (tasks A and B, `BASE_URL`/`API_KEY`, amd64, CPU, offline). On 150 dev
  claims per task with the organisers' `evaluate.py`: task A 0.857, task B 0.944, evidence score 0.76.
  How to run it: [`track_2a/README.md`](track_2a/README.md#official-evaluation-contract-what-the-organisers-run).
- **Live:** [booklet-nli.pages.dev](https://booklet-nli.pages.dev/) (every test answer) ·
  [try a claim](https://booklet-nli.pages.dev/demo/) (Apertus on a GPU that sleeps when idle; the
  first claim after a pause waits about 5 minutes for it to start).
- **Explore every test answer:** [`track_2a/docs/results.html`](track_2a/docs/results.html) —
  search and filter all 976 claims, read the booklet passage behind each one, click the confusion matrix.
- **Write-up:** [`track_2a/kaiser-data_Report.pdf`](track_2a/kaiser-data_Report.pdf)
  ([markdown source](track_2a/technical_report.md)) · **Code and how to run it:** [`track_2a/README.md`](track_2a/README.md)

```bash
export BASE_URL=https://api.inference.cscs.ch/v1 API_KEY=...
make cases      # Docker: the evaluation contract on track_2a/examples/ (forwards to track_2a/)
make run        # Docker: 60-claim held-out sample of our test split, prints macro-F1
```

---

*The rest of this file is the Hack Apertus template's description of the repository layout.*

## Hack Apertus template

Template repository for [Hack Apertus](https://hackapertus.ch/) submissions.
Every project keeps almost the same layout, so organizers and judges find the
same things in the same place.

## Select your track

This repository holds one example project per track:

- `track_1a/`
- `track_1b/`
- `track_2a/`
- `track_2b/`

Keep the directory for the track you are competing in **exactly as it is** —
don't rename it or move its files — and delete the other track directories.
That directory is your project root. Keep the files and directories as shown
below.

## The structure

| Path | What it is |
| --- | --- |
| `README.md` | The challenge description and submission requirements for your track |
| `technical_report.md` | The deeper write-up: architecture, evaluation, limitations |
| `Makefile` | `make run` must spin up your project |
| `src/` | Your code |
| `data/` | Datasets — `track_1a`, `track_2a` and `track_2b` only; max. 100 MB |
| `findings/` | Issue files — `track_1a` only |
| `docs/` | Diagrams, notes, longer write-ups |

## Run it

Judges run `make run` from the root of the project, on a clean checkout:

```bash
make run
```

`make run` is expected to run the project using Docker, since that
is how the judges will run it.
- If you used other local open-weight models, include instructions for running the project in your technical report.
- Use the following environment variables:
```
LLM_NAME — name and version of the model
LLM_BASE_URL — endpoint base URL
LLM_API_KEY — your API key
```

## Getting started

1. Click **Use this template** to create your own repository.
2. Delete the other track directories. Don't rename or restructure yours.
3. Read its `README.md` and fill in `technical_report.md`.
4. Make `make run` work from the root of the project, on a clean checkout.

## License

All Hack Apertus projects are open-sourced. Please check our Terms & Conditions for specific licensing details (6. What you build is open source): https://hackapertus.ch/terms-and-conditions
