import zlib

import pytest

from leaklens.base import Benchmark, BenchmarkItem, ModelInterface, TokenLogprob
from leaklens.detectors.perplexity_gap import (
    PerplexityGapDetector,
    cross_entropy_nats,
    perplexity_gap_score,
    zlib_compressed_length,
)


def test_cross_entropy_nats_is_mean_negative_logprob():
    assert cross_entropy_nats([-1.0, -2.0, -3.0]) == pytest.approx(2.0)


def test_cross_entropy_nats_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        cross_entropy_nats([])


def test_zlib_compressed_length_matches_stdlib():
    text = "hello world hello world hello world"
    assert zlib_compressed_length(text) == len(zlib.compress(text.encode("utf-8")))


def test_perplexity_gap_score_formula():
    logprobs = [-1.0, -2.0, -3.0]  # cross-entropy = 2.0
    text = "some text"
    expected = 2.0 / zlib_compressed_length(text)
    assert perplexity_gap_score(logprobs, text) == pytest.approx(expected)


def test_perplexity_gap_score_rejects_empty_text():
    with pytest.raises(ValueError, match="non-empty"):
        perplexity_gap_score([-1.0], "")


def test_zlib_compressed_length_of_empty_string_is_nonzero():
    # zlib always emits a header+footer, even for empty input -- this is
    # WHY perplexity_gap_score guards on `text` directly rather than on
    # compressed_len == 0, which can never actually happen.
    assert zlib_compressed_length("") == 8


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


def test_run_scores_items_and_averages():
    items = [BenchmarkItem(item_id="i1", text="some benchmark text here", order_index=0)]
    lps = [TokenLogprob(token="a", logprob=lp) for lp in [-1, -2, -3, -4, -5]]
    model = ScriptedLogprobModel("m", {"some benchmark text here": lps})
    detector = PerplexityGapDetector()
    result = detector.run(model, FakeBenchmark(items))

    expected = 3.0 / zlib_compressed_length("some benchmark text here")  # mean of [-1,-2,-3,-4,-5] negated = 3.0
    assert result.per_item_scores["i1"] == pytest.approx(expected)
    assert result.aggregate_score == pytest.approx(expected)


def test_run_skips_short_items():
    items = [BenchmarkItem(item_id="i1", text="hi", order_index=0)]
    model = ScriptedLogprobModel("m", {"hi": [TokenLogprob(token="a", logprob=-1.0)]})
    detector = PerplexityGapDetector()
    result = detector.run(model, FakeBenchmark(items))
    assert result.per_item_scores == {}
    assert result.metadata["n_items_skipped_too_short"] == 1
