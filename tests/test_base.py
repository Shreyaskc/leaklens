import json

from leaklens.base import BenchmarkItem, DetectorResult, ModelInterface, ReportCard, TokenLogprob, now_iso


class FakeModel(ModelInterface):
    name = "fake-model"
    supports_logprobs = False

    def generate(self, prompt: str, max_tokens: int) -> str:
        return "generated"


def test_model_interface_token_logprobs_raises_by_default():
    import pytest

    m = FakeModel()
    with pytest.raises(NotImplementedError, match="supports_logprobs=False"):
        m.token_logprobs("some text")


def test_benchmark_item_defaults():
    item = BenchmarkItem(item_id="x1", text="hello", order_index=0)
    assert item.fields == {}


def test_detector_result_defaults():
    r = DetectorResult(detector_name="d", applicable=True, skip_reason=None, aggregate_score=0.5)
    assert r.per_item_scores == {}
    assert r.metadata == {}


def test_report_card_to_json_roundtrips():
    r1 = DetectorResult(detector_name="d1", applicable=True, skip_reason=None, aggregate_score=0.3, per_item_scores={"a": 0.3})
    r2 = DetectorResult(detector_name="d2", applicable=False, skip_reason="not applicable", aggregate_score=None)
    card = ReportCard(
        model_name="m", benchmark_name="b", generated_at=now_iso(), detector_results=[r1, r2], leaklens_version="0.1.0"
    )
    parsed = json.loads(card.to_json())
    assert parsed["model_name"] == "m"
    assert len(parsed["detector_results"]) == 2
    assert parsed["detector_results"][0]["aggregate_score"] == 0.3


def test_report_card_to_markdown_includes_both_applicable_and_skipped():
    r1 = DetectorResult(detector_name="d1", applicable=True, skip_reason=None, aggregate_score=0.123456)
    r2 = DetectorResult(detector_name="d2", applicable=False, skip_reason="no logprobs", aggregate_score=None)
    card = ReportCard(model_name="m", benchmark_name="b", generated_at=now_iso(), detector_results=[r1, r2], leaklens_version="0.1.0")
    md = card.to_markdown()
    assert "d1" in md and "d2" in md
    assert "0.123" in md
    assert "no logprobs" in md
    assert "—" in md  # placeholder for d2's missing score


def test_token_logprob_is_frozen():
    import dataclasses
    import pytest

    tlp = TokenLogprob(token="a", logprob=-1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        tlp.token = "b"
