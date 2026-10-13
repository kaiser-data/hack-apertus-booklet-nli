"""Embed every test prediction into docs/results.html for the claim explorer.

    uv run python docs/build_explorer.py

Reads runs/test-rag-k16-n976/predictions.jsonl and the dataset (claim text, vote, booklet URL),
and replaces the JSON inside <script id="claims" type="application/json"> in results.html.
"""

import json
import re
from pathlib import Path

from booklet_nli.data import load_rows

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs" / "test-rag-k16-n976" / "predictions.jsonl"
PAGE = ROOT / "docs" / "results.html"


def main() -> None:
    rows = {r["id"]: r for r in load_rows()}
    preds = sorted(map(json.loads, RUN.read_text().splitlines()), key=lambda p: p["id"])
    votes, booklets = [], []

    def index(table: list, value: str) -> int:
        if value not in table:
            table.append(value)
        return table.index(value)

    claims = []
    for p in preds:
        r = rows[p["id"]]
        claims.append([
            p["id"], r["claim"], p["claim_language"], p["reference_language"],
            index(votes, f'{r["booklet_publish_date"]} · {r["vote"]}'),
            index(booklets, r["booklet_url"]),
            p["gold"], p["pred"], p["reason"], sorted(set(p["evidence_pages"])),
            int(p["parse_failed"]),
        ])
    data = json.dumps({"votes": votes, "booklets": booklets, "claims": claims},
                      ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = PAGE.read_text()
    html, n = re.subn(r'(<script id="claims" type="application/json">).*?(</script>)',
                      lambda m: m.group(1) + data + m.group(2), html, flags=re.DOTALL)
    if n != 1:
        raise SystemExit("results.html has no <script id=\"claims\"> block")
    PAGE.write_text(html)
    print(f"embedded {len(claims)} claims, {len(votes)} votes, {len(booklets)} booklets")


if __name__ == "__main__":
    main()
