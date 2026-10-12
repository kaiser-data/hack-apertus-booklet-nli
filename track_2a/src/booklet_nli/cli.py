"""Command line: predict one claim, run a batch file, or evaluate on the OST benchmark."""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .booklet import extract_passages, fetch_pdf
from .config import llm_config
from .data import load_rows, select
from .llm import LLM
from .metrics import macro_f1, overlaps_reference, paired_diff, summarize
from .nli import select_context, verify
from .retrieve import DEFAULT_EMBEDDER, rank


def _add_context_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mode", choices=["rag", "full"], default="rag",
                   help="rag: top-k retrieved passages; full: the whole booklet")
    p.add_argument("--k", type=int, default=16, help="passages retrieved in rag mode")
    p.add_argument("--embedder", default=DEFAULT_EMBEDDER)


def cmd_predict(args: argparse.Namespace) -> None:
    verdict = verify(LLM(llm_config()), args.booklet, args.claim, args.mode, args.k, args.embedder)
    print(json.dumps(verdict.to_dict(), ensure_ascii=False, indent=2))


def cmd_batch(args: argparse.Namespace) -> None:
    """Input JSONL rows need `booklet_url` (or `booklet`) and `claim`; output adds the verdict."""
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    llm = LLM(llm_config())

    def run(row: dict) -> dict:
        v = verify(llm, row.get("booklet_url") or row["booklet"], row["claim"], args.mode, args.k,
                   args.embedder)
        return {**row, "prediction": v.to_dict()}

    _prefetch({r.get("booklet_url") or r["booklet"] for r in rows})
    with ThreadPoolExecutor(args.workers) as pool, open(args.output, "w") as out:
        out.writelines(json.dumps(result, ensure_ascii=False) + "\n" for result in pool.map(run, rows))
    print(f"wrote {len(rows)} predictions to {args.output}", file=sys.stderr)


def _prefetch(booklets: set[str]) -> None:
    """Download and parse every booklet once, before worker threads race for the cache."""
    for b in sorted(booklets):
        extract_passages(fetch_pdf(b))


def cmd_eval(args: argparse.Namespace) -> None:
    rows = select(load_rows(args.dataset), args.split, args.limit, args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    done = {}
    if pred_path.exists():  # resumable: skip rows that already have a prediction
        done = {r["id"]: r for r in map(json.loads, pred_path.read_text().splitlines())}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} rows, {len(done)} already done, {len(todo)} to run", file=sys.stderr)
    _prefetch({r["booklet_url"] for r in todo})
    llm = LLM(llm_config())

    def run(row: dict) -> dict:
        v = verify(llm, row["booklet_url"], row["claim"], args.mode, args.k, args.embedder)
        context = select_context(row["claim"], extract_passages(fetch_pdf(row["booklet_url"])),
                                 args.mode, args.k, args.embedder)
        return {
            "id": row["id"], "gold": row["entailment_label"], "pred": v.label,
            "claim_language": row["claim_language"],
            "reference_language": row["reference_language"],
            "evidence_hit": any(overlaps_reference(p.text, row["reference_string"])
                                for p in v.evidence),
            "context_hit": any(overlaps_reference(p.text, row["reference_string"])
                               for p in context),
            "prompt_tokens": v.prompt_tokens, "completion_tokens": v.completion_tokens,
            "seconds": v.seconds, "parse_failed": not v.parsed, "reason": v.reason,
            "evidence_pages": [p.page for p in v.evidence],
        }

    with ThreadPoolExecutor(args.workers) as pool, open(pred_path, "a") as out:
        futures = [pool.submit(run, r) for r in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            done[rec["id"]] = rec
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if n % 10 == 0:
                print(f"  {n}/{len(todo)}", file=sys.stderr)

    wanted = {r["id"] for r in rows}
    metrics = summarize([done[i] for i in sorted(wanted)])
    metrics["config"] = {"model": llm.model, "mode": args.mode, "k": args.k,
                         "embedder": args.embedder, "split": args.split, "limit": args.limit,
                         "seed": args.seed}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


def cmd_retrieval(args: argparse.Namespace) -> None:
    """CPU-only: how often the annotated reference passage is among the top-k (no LLM calls)."""
    rows = [r for r in select(load_rows(args.dataset), args.split, args.limit, args.seed)
            if r["entailment_label"] != 1]
    _prefetch({r["booklet_url"] for r in rows})
    ks = sorted(int(k) for k in args.ks.split(","))
    hits = {k: 0 for k in ks}
    for row in rows:
        order = rank(row["claim"], extract_passages(fetch_pdf(row["booklet_url"])), args.embedder)
        for k in ks:
            hits[k] += any(overlaps_reference(p.text, row["reference_string"]) for p in order[:k])
    result = {"embedder": args.embedder, "n": len(rows),
              "recall_at_k": {k: round(hits[k] / len(rows), 4) for k in ks}}
    print(json.dumps(result, indent=2))


def _predictions(run: str) -> dict[int, dict]:
    path = Path(run) / "predictions.jsonl"
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}


def cmd_compare(args: argparse.Namespace) -> None:
    """Macro-F1 of two runs on the claims they share, with a paired bootstrap interval."""
    a, b = _predictions(args.run_a), _predictions(args.run_b)
    ids = sorted(a.keys() & b.keys())
    gold = [a[i]["gold"] for i in ids]
    pa, pb = [a[i]["pred"] for i in ids], [b[i]["pred"] for i in ids]
    result = {"n_shared": len(ids), "a": args.run_a, "b": args.run_b,
              "macro_f1_a": round(macro_f1(gold, pa), 4),
              "macro_f1_b": round(macro_f1(gold, pb), 4),
              "a_minus_b": paired_diff(gold, pa, pb),
              "only_a_correct": sum(x == g != y for g, x, y in zip(gold, pa, pb)),
              "only_b_correct": sum(y == g != x for g, x, y in zip(gold, pa, pb))}
    print(json.dumps(result, indent=2))


def cmd_rescore(args: argparse.Namespace) -> None:
    """Recompute metrics.json from predictions.jsonl, keeping the recorded config."""
    for run in args.runs:
        path = Path(run) / "metrics.json"
        config = json.loads(path.read_text()).get("config") if path.exists() else None
        metrics = summarize([r for _, r in sorted(_predictions(run).items())])
        if config:
            metrics["config"] = config
        path.write_text(json.dumps(metrics, indent=2))
        print(f"{run}: macro-F1 {metrics['macro_f1']} CI95 {metrics['macro_f1_ci95']}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="booklet-nli", description=__doc__)
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("predict", help="classify one claim against one booklet")
    p.add_argument("--booklet", required=True, help="PDF path or URL")
    p.add_argument("--claim", required=True)
    _add_context_args(p)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("batch", help="classify a JSONL file of {booklet_url, claim} rows")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=8)
    _add_context_args(p)
    p.set_defaults(func=cmd_batch)

    for name, func in (("eval", cmd_eval), ("retrieval", cmd_retrieval)):
        p = sub.add_parser(name, help="evaluate on the OST benchmark" if name == "eval"
                           else "retrieval recall@k on the OST benchmark (CPU only)")
        p.add_argument("--dataset", help="local JSONL; default downloads it from Hugging Face")
        p.add_argument("--split", choices=["dev", "test", "all"], default="dev")
        p.add_argument("--limit", type=int)
        p.add_argument("--seed", type=int, default=0)
        p.set_defaults(func=func)
        if name == "eval":
            p.add_argument("--out", required=True, help="run directory (resumable)")
            p.add_argument("--workers", type=int, default=8)
            _add_context_args(p)
        else:
            p.add_argument("--ks", default="1,3,5,8,12,20")
            p.add_argument("--embedder", default=DEFAULT_EMBEDDER)

    p = sub.add_parser("compare", help="paired macro-F1 comparison of two run directories")
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("rescore", help="recompute metrics.json of run directories")
    p.add_argument("runs", nargs="+")
    p.set_defaults(func=cmd_rescore)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
