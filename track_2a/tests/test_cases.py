import json

from booklet_nli import cases
from booklet_nli.booklet import Passage
from booklet_nli.llm import Completion


class StubLLM:
    model = "stub"

    def __init__(self, text: str, fail: bool = False):
        self.text, self.fail, self.messages = text, fail, []

    def pick(self, preferred):
        return self.model

    def complete(self, messages, max_tokens, json_schema=None):
        if self.fail:
            raise ConnectionError("down")
        self.messages.append(messages)
        return Completion(self.text, 100, 10, 0.1)


def _case(**source):
    return {"id": "c1", "vote": "Ehe für alle", "claim": {"text": "X", "language": "de"}, **source}


def _check_shape(r):
    assert set(r) == {"id", "label", "label_name", "evidence", "metrics"}
    assert set(r["metrics"]) == {"input_tokens", "output_tokens", "inference_time_ms"}


def test_task_b_uses_the_reference_and_a_null_page(tmp_path):
    llm = StubLLM('{"reason": "r", "evidence": [0], "label": "contradiction"}')
    r = cases.answer(llm, _case(reference={"text": "Der Text.", "language": "de"}), tmp_path,
                     16, "e", use_vote=True)
    _check_shape(r)
    assert (r["label"], r["label_name"]) == (2, "contradiction")
    assert r["evidence"] == [{"page": None, "text": "Der Text."}]
    assert r["metrics"]["input_tokens"] == 100
    assert "Vote: Ehe für alle" in llm.messages[0][1]["content"]


def test_task_a_cites_the_best_passage_when_the_model_cites_none(tmp_path, monkeypatch):
    passages = [Passage(0, 3, "", "a"), Passage(1, 7, "", "b")]
    monkeypatch.setattr(cases, "extract_passages", lambda path: passages)
    monkeypatch.setattr(cases, "rank", lambda claim, ps, emb: [ps[1], ps[0]])
    monkeypatch.setattr(cases, "embed_scores", lambda claim, texts, emb: [1.0] * len(texts))
    llm = StubLLM('{"reason": "r", "evidence": [], "label": "entailment"}')
    r = cases.answer(llm, _case(booklet={"path": "b.pdf", "language": "de"}), tmp_path, 16, "e",
                     use_vote=False)
    assert r["evidence"] == [{"page": 7, "text": "b"}, {"page": 3, "text": "a"}]
    assert "Vote:" not in llm.messages[0][1]["content"]


def test_a_failing_case_abstains_and_the_run_finishes(tmp_path, monkeypatch):
    monkeypatch.setattr(cases.time, "sleep", lambda s: None)
    monkeypatch.setattr(cases, "LLM", lambda cfg, timeout: StubLLM("", fail=True))
    inp, out = tmp_path / "cases.jsonl", tmp_path / "out" / "predictions.jsonl"
    inp.write_text(json.dumps(_case(reference={"text": "t", "language": "fr"})) + "\n")
    cases.main(["--input", str(inp), "--output", str(out)])
    [r] = [json.loads(line) for line in out.read_text().splitlines()]
    _check_shape(r)
    assert (r["id"], r["label"], r["evidence"]) == ("c1", 1, [])


def test_quotes_are_sentences_from_cited_passages_first(monkeypatch):
    cited = Passage(0, 4, "", "Der Bundesrat empfiehlt ein Nein zur Vorlage. Das Parlament hat "
                              "die Vorlage im Juni mit grosser Mehrheit angenommen.")
    other = Passage(1, 9, "", "Die Kosten betragen nach Schätzung des Bundes rund 50 Millionen.")
    monkeypatch.setattr(cases, "embed_scores",
                        lambda claim, texts, emb: [("Parlament" in t) * 1.0 for t in texts])
    q = cases.quotes("claim", [cited], [cited, other], "e")
    assert [i["page"] for i in q] == [4, 4, 9]
    assert q[0]["text"].startswith("Das Parlament hat")


def test_input_with_a_line_separator_inside_a_string_is_one_case(tmp_path, monkeypatch):
    monkeypatch.setattr(cases, "LLM", lambda cfg, timeout: StubLLM(
        '{"reason": "r", "evidence": [], "label": "neutral"}'))
    inp, out = tmp_path / "cases.jsonl", tmp_path / "predictions.jsonl"
    inp.write_text(json.dumps(_case(reference={"text": "a b", "language": "de"}),
                              ensure_ascii=False) + "\n")
    cases.main(["--input", str(inp), "--output", str(out)])
    assert [json.loads(line)["id"] for line in out.read_text().splitlines()] == ["c1"]


def test_quotes_do_not_repeat_when_a_passage_is_cited_twice(monkeypatch):
    p = Passage(0, 4, "", "Der Bundesrat empfiehlt ein Nein zur Vorlage. Das Parlament hat "
                          "die Vorlage im Juni mit grosser Mehrheit angenommen.")
    monkeypatch.setattr(cases, "embed_scores", lambda claim, texts, emb: [1.0] * len(texts))
    q = cases.quotes("claim", [p, p], [p], "e")
    assert len(q) == len({i["text"] for i in q}) == 2


def test_pick_prefers_the_70b_when_served_and_falls_back_on_404():
    import httpx
    from openai import NotFoundError

    from booklet_nli.config import LLMConfig
    from booklet_nli.llm import LLM

    llm = LLM(LLMConfig("x", "http://127.0.0.1:9/v1", "k"))
    llm.served_models = lambda: ["swiss-ai/Apertus-v1.5-8B", "Apertus-v1.5-70B"]
    assert llm.pick(["swiss-ai/Apertus-v1.5-70B", "swiss-ai/Apertus-v1.5-8B"]) == "Apertus-v1.5-70B"

    llm.served_models = list  # no listing: try the first, keep the rest as fallbacks
    assert llm.pick(["a/70B", "a/8B"]) == "a/70B"
    calls = []

    def create(**kw):
        calls.append(kw["model"])
        if kw["model"] == "a/70B":
            raise NotFoundError("no such model", response=httpx.Response(
                404, request=httpx.Request("POST", "http://x")), body=None)
        return "ok"
    llm.client.chat.completions.create = create
    assert llm._create({"model": "ignored"}) == "ok" and calls == ["a/70B", "a/8B"]
    assert llm.model == "a/8B"


def test_a_400_naming_the_model_also_falls_back():
    import httpx
    from openai import BadRequestError

    from booklet_nli.config import LLMConfig
    from booklet_nli.llm import LLM

    llm = LLM(LLMConfig("x", "http://127.0.0.1:9/v1", "k"))
    llm.served_models = list
    llm.pick(["a/70B", "a/8B"])

    def create(**kw):
        if kw["model"] == "a/70B":
            raise BadRequestError("The model `a/70B` does not exist.", response=httpx.Response(
                400, request=httpx.Request("POST", "http://x")), body=None)
        return "ok"
    llm.client.chat.completions.create = create
    assert llm._create({}) == "ok" and llm.model == "a/8B"


def test_a_context_length_400_does_not_switch_the_model():
    import httpx
    import pytest
    from openai import BadRequestError

    from booklet_nli.config import LLMConfig
    from booklet_nli.llm import LLM

    llm = LLM(LLMConfig("x", "http://127.0.0.1:9/v1", "k"))
    llm.served_models = list
    llm.pick(["a/70B", "a/8B"])

    def create(**kw):
        raise BadRequestError("This model's maximum context length is 32768 tokens.",
                              response=httpx.Response(400, request=httpx.Request("POST", "http://x")),
                              body=None)
    llm.client.chat.completions.create = create
    with pytest.raises(BadRequestError):
        llm._create({})
    assert llm.model == "a/70B"
