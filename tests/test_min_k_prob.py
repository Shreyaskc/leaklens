import pytest

from leaklens.base import Benchmark, BenchmarkItem, ModelInterface, TokenLogprob
from leaklens.detectors.min_k_prob import MinKProbDetector, min_k_percent_logprob


def test_min_k_percent_logprob_basic():
    # 10 logprobs, k=20% -> bottom 2 (most negative) averaged.
    logprobs = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0]
    result = min_k_percent_logprob(logprobs, k_percent=20.0)
    assert result == pytest.approx((-10.0 + -9.0) / 2)


def test_min_k_percent_logprob_rounds_up_to_at_least_one():
    result = min_k_percent_logprob([-1.0, -5.0], k_percent=1.0)  # round(2*0.01)=0 -> clamped to 1
    assert result == -5.0


def test_min_k_percent_logprob_100_percent_is_plain_mean():
    logprobs = [-1.0, -2.0, -3.0]
    assert min_k_percent_logprob(logprobs, k_percent=100.0) == pytest.approx(-2.0)


def test_min_k_percent_logprob_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        min_k_percent_logprob([], k_percent=20.0)


def test_min_k_percent_logprob_rejects_bad_k():
    with pytest.raises(ValueError, match="k_percent"):
        min_k_percent_logprob([-1.0], k_percent=0.0)
    with pytest.raises(ValueError, match="k_percent"):
        min_k_percent_logprob([-1.0], k_percent=150.0)


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


def test_run_scores_each_item_and_averages():
    items = [
        BenchmarkItem(item_id="i1", text="text one", order_index=0),
        BenchmarkItem(item_id="i2", text="text two", order_index=1),
    ]
    logprobs_by_text = {
        "text one": [TokenLogprob(token="a", logprob=lp) for lp in [-1, -2, -3, -4, -5]],
        "text two": [TokenLogprob(token="a", logprob=lp) for lp in [-10, -20, -30, -40, -50]],
    }
    model = ScriptedLogprobModel("m", logprobs_by_text)
    detector = MinKProbDetector(k_percent=20.0)
    result = detector.run(model, FakeBenchmark(items))

    assert result.per_item_scores["i1"] == pytest.approx(-5.0)
    assert result.per_item_scores["i2"] == pytest.approx(-50.0)
    assert result.aggregate_score == pytest.approx((-5.0 + -50.0) / 2)


def test_run_skips_items_with_too_few_tokens():
    items = [BenchmarkItem(item_id="i1", text="hi", order_index=0)]
    model = ScriptedLogprobModel("m", {"hi": [TokenLogprob(token="a", logprob=-1.0)]})
    detector = MinKProbDetector()
    result = detector.run(model, FakeBenchmark(items))
    assert result.per_item_scores == {}
    assert result.metadata["n_items_skipped_too_short"] == 1
    assert result.aggregate_score is None
