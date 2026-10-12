from booklet_nli.booklet import _clean
from booklet_nli.data import select, split_of
from booklet_nli.metrics import bootstrap_ci, macro_f1, overlaps_reference, paired_diff
from booklet_nli.nli import parse_verdict


def test_clean_rejoins_words_broken_across_lines():
    assert _clean("eine landwirt\nschaftliche Produktion \nim Land") == (
        "eine landwirtschaftliche Produktion im Land")
    assert _clean("Produk-\ntionsmenge") == "Produktionsmenge"


def test_parse_verdict_reads_json_inside_noise():
    label, reason, ev = parse_verdict('ok: {"reason": "r", "evidence": [3, "4"], '
                                      '"label": "Contradiction"}')
    assert (label, reason, ev) == (2, "r", [3, 4])


def test_parse_verdict_flags_garbage():
    assert parse_verdict("no idea")[0] is None
    assert parse_verdict('{"label": "maybe"}')[0] is None


def test_macro_f1():
    assert macro_f1([0, 1, 2], [0, 1, 2]) == 1.0
    assert abs(macro_f1([0, 0, 1, 2], [0, 1, 1, 1]) - (2 / 3 + 1 / 2 + 0) / 3) < 1e-9


def test_overlap():
    ref = "Der Bundesrat empfiehlt die Initiative abzulehnen weil sie zu weit geht"
    assert overlaps_reference("Bundesrat empfiehlt die Initiative abzulehnen", ref)
    assert not overlaps_reference("Das Parlament hat die Vorlage angenommen und so", ref)


def _row(i, date, label=0, cl="de", rl="de"):
    return {"id": i, "booklet_publish_date": date, "entailment_label": label,
            "claim_language": cl, "reference_language": rl}


def test_split_keeps_one_vote_on_one_side():
    rows = [_row(i, "2021-06-13", cl=c, rl=r) for i, (c, r) in enumerate(
        [("de", "fr"), ("fr", "it"), ("it", "de")])]
    assert len({split_of(r) for r in rows}) == 1


def test_select_is_balanced_and_deterministic():
    rows = [_row(i, "x", label=i % 3) for i in range(30)]
    a = select(rows, "all", 9)
    assert [r["entailment_label"] for r in a].count(0) == 3
    assert a == select(rows, "all", 9)


def test_bootstrap_ci_brackets_the_score():
    gold = [0, 1, 2] * 20
    pred = [0, 1, 2] * 15 + [1, 2, 0] * 5
    lo, hi = bootstrap_ci(gold, pred, n=500)
    assert lo <= macro_f1(gold, pred) <= hi < 1.0
    assert bootstrap_ci(gold, gold, n=200) == [1.0, 1.0]


def test_paired_diff_is_zero_for_identical_runs():
    gold = [0, 1, 2] * 10
    pred = [0, 1, 1] * 10
    assert paired_diff(gold, pred, pred, n=200) == {"diff": 0.0, "ci95": [0.0, 0.0]}
