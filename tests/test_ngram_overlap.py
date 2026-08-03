import pytest
import responses

from leaklens.base import Benchmark, BenchmarkItem
from leaklens.detectors.ngram_overlap import API_URL, NgramOverlapDetector


class FakeBenchmark(Benchmark):
    name = "fake-bench"

    def __init__(self, items):
        self._items = items

    def items(self):
        return self._items


def test_unknown_index_raises_at_construction():
    with pytest.raises(ValueError, match="Unknown infini-gram index"):
        NgramOverlapDetector(index="not_a_real_index")


@responses.activate
def test_query_count_parses_successful_response():
    responses.add(
        responses.POST,
        API_URL,
        json={"approx": False, "count": 42, "latency": 1.2, "token_ids": [1, 2], "tokens": ["a", "b"]},
        status=200,
    )
    detector = NgramOverlapDetector()
    assert detector.query_count("some text") == 42


@responses.activate
def test_query_count_raises_on_api_error_payload():
    responses.add(responses.POST, API_URL, json={"error": "Invalid index: xyz"}, status=200)
    detector = NgramOverlapDetector()
    with pytest.raises(RuntimeError, match="Invalid index"):
        detector.query_count("some text")


@responses.activate
def test_run_scores_contaminated_and_clean_items():
    # Item "contaminated" gets count>0, item "clean" gets count==0.
    def response_callback(request):
        import json

        body = json.loads(request.body)
        count = 100 if "contaminated" in body["query"] else 0
        return (200, {}, json.dumps({"approx": False, "count": count, "token_ids": [], "tokens": []}))

    responses.add_callback(responses.POST, API_URL, callback=response_callback, content_type="application/json")

    items = [
        BenchmarkItem(item_id="i1", text="this text is contaminated", order_index=0),
        BenchmarkItem(item_id="i2", text="this text is clean", order_index=1),
    ]
    detector = NgramOverlapDetector()
    result = detector.run(model=None, benchmark=FakeBenchmark(items))

    assert result.per_item_scores["i1"] == 100.0
    assert result.per_item_scores["i2"] == 0.0
    assert result.aggregate_score == 0.5  # 1 of 2 items contaminated
    assert result.metadata["n_items_query_failed"] == 0


@responses.activate
def test_run_records_query_failures_without_crashing():
    responses.add(responses.POST, API_URL, json={"error": "server error"}, status=200)
    items = [BenchmarkItem(item_id="i1", text="text", order_index=0)]
    detector = NgramOverlapDetector()
    result = detector.run(model=None, benchmark=FakeBenchmark(items))

    assert result.per_item_scores == {}
    assert result.metadata["n_items_query_failed"] == 1
    assert "i1" in result.metadata["query_errors"]
    assert result.aggregate_score is None  # no successfully-queried items


def test_run_attaches_scope_caveat_metadata():
    detector = NgramOverlapDetector()
    with responses.RequestsMock() as rsps:
        rsps.add(responses.POST, API_URL, json={"count": 0, "token_ids": [], "tokens": []})
        result = detector.run(model=None, benchmark=FakeBenchmark([BenchmarkItem(item_id="i", text="t", order_index=0)]))
    assert "not proof" in result.metadata["scope_caveat"]
