import pytest

from leaklens.base import Benchmark, BenchmarkItem, ModelInterface, TokenLogprob
from leaklens.detectors.paraphrase_gap import ParaphraseGapDetector


class FakeBenchmark(Benchmark):
    name = "fake-bench"

    def __init__(self, items):
        self._items = items

    def items(self):
        return self._items


class ScriptedLogprobModel(ModelInterface):
    supports_logprobs = True

    def __init__(self, name, logprobs_by_text):
        self.name = name
        self._logprobs_by_text = logprobs_by_text

    def generate(self, prompt, max_tokens):
        return "unused"

    def token_logprobs(self, text):
        return self._logprobs_by_text[text]


def test_run_skips_items_without_paraphrase_field():
    items = [BenchmarkItem(item_id="i1", text="original text", order_index=0, fields={})]
    model = ScriptedLogprobModel("m", {})
    detector = ParaphraseGapDetector()
    result = detector.run(model, FakeBenchmark(items))
    assert result.per_item_scores == {}
    assert result.metadata["n_items_no_paraphrase_available"] == 1
    assert result.aggregate_score is None


def test_run_computes_gap_when_paraphrase_present():
    items = [
        BenchmarkItem(
            item_id="i1", text="original text here", order_index=0, fields={"paraphrase": "a rephrased version"}
        )
    ]
    original_lps = [TokenLogprob(token="a", logprob=lp) for lp in [-1, -1, -1, -1, -1]]  # CE = 1.0
    paraphrase_lps = [TokenLogprob(token="a", logprob=lp) for lp in [-3, -3, -3, -3, -3]]  # CE = 3.0
    model = ScriptedLogprobModel(
        "m", {"original text here": original_lps, "a rephrased version": paraphrase_lps}
    )
    detector = ParaphraseGapDetector()
    result = detector.run(model, FakeBenchmark(items))

    # paraphrase_ce - original_ce = 3.0 - 1.0 = 2.0 (original much easier -> contamination-suggestive)
    assert result.per_item_scores["i1"] == pytest.approx(2.0)
    assert result.aggregate_score == pytest.approx(2.0)


def test_run_skips_items_with_too_few_tokens_in_either_text():
    items = [
        BenchmarkItem(item_id="i1", text="hi", order_index=0, fields={"paraphrase": "a full paraphrase sentence"})
    ]
    model = ScriptedLogprobModel(
        "m",
        {
            "hi": [TokenLogprob(token="a", logprob=-1.0)],
            "a full paraphrase sentence": [TokenLogprob(token="a", logprob=lp) for lp in [-1, -1, -1, -1, -1]],
        },
    )
    detector = ParaphraseGapDetector()
    result = detector.run(model, FakeBenchmark(items))
    assert result.per_item_scores == {}
    assert result.metadata["n_items_skipped_too_short"] == 1


def test_run_reports_both_skip_reasons_independently():
    items = [
        BenchmarkItem(item_id="no_paraphrase", text="text", order_index=0, fields={}),
        BenchmarkItem(item_id="has_it", text="original text here", order_index=1, fields={"paraphrase": "rephrased version here"}),
    ]
    lps = [TokenLogprob(token="a", logprob=-1.0) for _ in range(5)]
    model = ScriptedLogprobModel("m", {"original text here": lps, "rephrased version here": lps})
    detector = ParaphraseGapDetector()
    result = detector.run(model, FakeBenchmark(items))
    assert result.metadata["n_items_no_paraphrase_available"] == 1
    assert "has_it" in result.per_item_scores
    assert result.per_item_scores["has_it"] == pytest.approx(0.0)
