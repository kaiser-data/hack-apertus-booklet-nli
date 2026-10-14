"""Write official-format cases (tasks A and B) for one split, as the starter's prepare_cases.py does.

Same ids, fields and file names as https://gitlab.com/ifsoftware/hackapertus-starter, but from the
pinned dataset revision and restricted to our dev or test split, with a balanced sample.
Gold labels go to a separate expected-labels.jsonl that the predictor never reads.

    uv run python scripts/make_cases.py --split dev --limit 150 --out runs/cases-dev-n150
"""

import argparse
import json
import shutil
from pathlib import Path

from booklet_nli.booklet import fetch_pdf
from booklet_nli.data import load_rows, select

VERSION = "v1.1"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    p.add_argument("--limit", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--task", choices=["A", "B", "both"], default="both")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    rows = select(load_rows(), args.split, args.limit, args.seed)
    tasks = ["A", "B"] if args.task == "both" else [args.task]
    cases, expected = [], []
    for row in rows:
        name = f"{row['booklet_publish_date'].replace('-', '_')}_{row['reference_language']}.pdf"
        if "A" in tasks:
            target = args.out / "booklets" / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(fetch_pdf(row["booklet_url"]), target)
        for task in tasks:
            case = {"id": f"{VERSION}-row-{row['id']}-{task}", "vote": row["vote"],
                    "claim": {"text": row["claim"], "language": row["claim_language"]}}
            if task == "A":
                case["booklet"] = {"path": f"booklets/{name}", "language": row["reference_language"]}
            else:
                case["reference"] = {"text": row["reference_string"],
                                     "language": row["reference_language"]}
            cases.append(case)
            gold = {"id": case["id"], "label": row["entailment_label"]}
            if row["entailment_label"] != 1:
                gold["reference"] = row["reference_string"]
            expected.append(gold)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, records in (("cases.jsonl", cases), ("expected-labels.jsonl", expected)):
        with (args.out / name).open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    print(f"wrote {len(cases)} cases from {len(rows)} {args.split} rows to {args.out}")


if __name__ == "__main__":
    main()
