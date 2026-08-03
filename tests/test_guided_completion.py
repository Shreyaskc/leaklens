import pytest

from leaklens.base import Benchmark, BenchmarkItem, ModelInterface
from leaklens.detectors.guided_completion import (
    GuidedCompletionDetector,
    similarity_ratio,
    split_prefix_continuation,
)


def test_split_prefix_continuation_basic():
    prefix, cont = split_prefix_continuation("abcdefghij", 0.5)
    assert prefix == "abcde"
    assert cont == "fghij"


def test_split_prefix_continuation_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        split_prefix_continuation("text", 0.0)
    with pytest.raises(ValueError):
        split_prefix_continuation("text", 1.0)


def test_similarity_ratio_identical_is_one():
    assert similarity_ratio("hello world", "hello world") == 1.0


def test_similarity_ratio_disjoint_is_low():
    assert similarity_ratio("aaaaaaaa", "zzzzzzzz") == 0.0


class ScriptedModel(ModelInterface):
    supports_logprobs = False

    def __init__(self, name, response_fn):
        self.name = name
        self._response_fn = response_fn

    def generate(self, prompt, max_tokens):
        return self._response_fn(prompt, max_tokens)


class FakeBenchmark(Benchmark):
    name = "fake-bench"

    def __init__(self, items):
        self._items = items

    def items(self):
        return self._items


def test_run_scores_verbatim_memorization_as_similarity_one():
    full_text = "The quick brown fox jumps over the lazy dog and keeps running fast."
    item = BenchmarkItem(item_id="i1", text=full_text, order_index=0)

    def perfect_recall(prompt, max_tokens):
        # Model reproduces the exact true continuation verbatim.
        _, true_continuation = split_prefix_continuation(full_text, 0.7)
        return true_continuation

    model = ScriptedModel("memorizer", perfect_recall)
    detector = GuidedCompletionDetector(prefix_fraction=0.7)
    result = detector.run(model, FakeBenchmark([item]))

    assert result.per_item_scores["i1"] == pytest.approx(1.0)
    assert result.aggregate_score == pytest.approx(1.0)


def test_run_scores_unrelated_completion_as_low_similarity():
    full_text = "The quick brown fox jumps over the lazy dog and keeps running fast."
    item = BenchmarkItem(item_id="i1", text=full_text, order_index=0)
    model = ScriptedModel("clean", lambda p, t: "zzz qqq xxx unrelated garbage 12345")
    detector = GuidedCompletionDetector(prefix_fraction=0.7)
    result = detector.run(model, FakeBenchmark([item]))
    assert result.per_item_scores["i1"] < 0.3


def test_run_skips_items_too_short():
    item = BenchmarkItem(item_id="i1", text="short", order_index=0)
    model = ScriptedModel("m", lambda p, t: "x")
    detector = GuidedCompletionDetector()
    result = detector.run(model, FakeBenchmark([item]))
    assert result.per_item_scores == {}
    assert result.aggregate_score is None
    assert result.metadata["n_items_skipped_too_short"] == 1


def test_run_compares_only_up_to_true_continuation_length():
    # Model keeps generating well past the true continuation's length; only
    # the matching-length prefix of the completion should be compared.
    # prefix_fraction=0.5 is exactly representable in binary float, so the
    # split point (20/40 chars) has no floating-point rounding surprise.
    full_text = "A" * 20 + "B" * 20  # 40 chars total
    assert split_prefix_continuation(full_text, 0.5) == ("A" * 20, "B" * 20)
    item = BenchmarkItem(item_id="i1", text=full_text, order_index=0)
    model = ScriptedModel("m", lambda p, t: "B" * 20 + "extra garbage that should be ignored")
    detector = GuidedCompletionDetector(prefix_fraction=0.5)
    result = detector.run(model, FakeBenchmark([item]))
    assert result.per_item_scores["i1"] == pytest.approx(1.0)
