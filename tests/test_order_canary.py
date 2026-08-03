import pytest

from leaklens.base import Benchmark, BenchmarkItem, ModelInterface
from leaklens.detectors.order_canary import OrderCanaryDetector


class FakeBenchmark(Benchmark):
    name = "fake-bench"

    def __init__(self, items):
        self._items = items

    def items(self):
        return self._items


class ScriptedModel(ModelInterface):
    supports_logprobs = False

    def __init__(self, name, response_fn):
        self.name = name
        self._response_fn = response_fn

    def generate(self, prompt, max_tokens):
        return self._response_fn(prompt, max_tokens)


def make_items(n, tail_chars=200, head_chars=100):
    # Each item's HEAD (first head_chars) and TAIL (last tail_chars) use a
    # distinct repeated character per item -- not shared filler. This
    # matters twice over: (1) item_by_tail below keys a lookup dict on each
    # item's tail slice, which must be unique per item or items collide and
    # overwrite each other; (2) similarity_ratio between two items' heads
    # must be near 0 when they're DIFFERENT items and 1.0 when they're the
    # SAME item, so the real-vs-shuffled-control gap actually measures
    # something -- shared filler characters would inflate both scores
    # equally and erase the signal the test is trying to observe.
    items = []
    for i in range(n):
        head_char = chr(65 + i)  # 'A', 'B', 'C', ...
        tail_char = chr(97 + i)  # 'a', 'b', 'c', ...
        text = (head_char * head_chars) + "-mid-" + (tail_char * tail_chars)
        items.append(BenchmarkItem(item_id=f"i{i}", text=text, order_index=i))
    return items


def test_run_not_applicable_with_fewer_than_three_items():
    detector = OrderCanaryDetector()
    result = detector.run(ScriptedModel("m", lambda p, t: ""), FakeBenchmark(make_items(2)))
    assert result.applicable is False
    assert "fewer than 3 items" in result.skip_reason


def test_memorizing_model_scores_higher_than_shuffled_control():
    items = make_items(6)
    item_by_tail = {it.text[-200:]: it for it in items}

    def memorizing_generate(prompt, max_tokens):
        # "Memorized" model: given item i's tail, it outputs the TRUE next
        # item's head verbatim -- simulating a model that has memorized the
        # benchmark's exact published ordering.
        current = item_by_tail.get(prompt)
        if current is None:
            return "no idea"
        next_item = next((it for it in items if it.order_index == current.order_index + 1), None)
        return next_item.text[:100] if next_item else "no idea"

    model = ScriptedModel("memorizer", memorizing_generate)
    detector = OrderCanaryDetector(shuffle_seed=0)
    result = detector.run(model, FakeBenchmark(items))

    assert result.applicable is True
    assert result.aggregate_score > 0.5  # real-pair similarity should dominate
    assert result.metadata["real_pair_mean_similarity"] > result.metadata["shuffled_control_mean_similarity"]


def test_non_memorizing_model_scores_near_zero_gap():
    items = make_items(6)

    model = ScriptedModel("clean", lambda p, t: "totally unrelated filler output")
    detector = OrderCanaryDetector(shuffle_seed=0)
    result = detector.run(model, FakeBenchmark(items))

    assert result.applicable is True
    # Neither real nor control pairing should score well for a model that
    # ignores the prompt entirely -- the gap should be small in both directions.
    assert abs(result.aggregate_score) < 0.3


def test_run_skips_items_below_length_thresholds():
    short_items = [BenchmarkItem(item_id=f"i{i}", text="short", order_index=i) for i in range(4)]
    detector = OrderCanaryDetector()
    result = detector.run(ScriptedModel("m", lambda p, t: "x"), FakeBenchmark(short_items))
    assert result.applicable is False
    assert "minimum length thresholds" in result.skip_reason


def test_shuffle_is_deterministic_given_seed():
    items = make_items(8)
    model = ScriptedModel("m", lambda p, t: "x")
    detector_a = OrderCanaryDetector(shuffle_seed=42)
    detector_b = OrderCanaryDetector(shuffle_seed=42)
    result_a = detector_a.run(model, FakeBenchmark(items))
    result_b = detector_b.run(model, FakeBenchmark(items))
    assert result_a.metadata["shuffled_control_mean_similarity"] == result_b.metadata["shuffled_control_mean_similarity"]
