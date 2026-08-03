from leaklens.audit import audit
from leaklens.base import Benchmark, BenchmarkItem, DetectorResult, ModelInterface
from leaklens.detectors.base import Detector


class FakeBenchmark(Benchmark):
    name = "fake-bench"

    def __init__(self, items):
        self._items = items

    def items(self):
        return self._items


class FakeModel(ModelInterface):
    supports_logprobs = False

    def __init__(self, name):
        self.name = name

    def generate(self, prompt, max_tokens):
        return "x"


class AlwaysApplicableDetector(Detector):
    name = "always"
    requires_logprobs = False

    def run(self, model, benchmark, **kwargs):
        return DetectorResult(detector_name=self.name, applicable=True, skip_reason=None, aggregate_score=0.7)


class LogprobOnlyDetector(Detector):
    name = "logprob_only"
    requires_logprobs = True

    def run(self, model, benchmark, **kwargs):
        return DetectorResult(detector_name=self.name, applicable=True, skip_reason=None, aggregate_score=0.1)


def test_audit_runs_applicable_detectors():
    model = FakeModel("m")
    benchmark = FakeBenchmark([BenchmarkItem(item_id="i1", text="text", order_index=0)])
    card = audit(model, benchmark, detectors=[AlwaysApplicableDetector()])
    assert card.model_name == "m"
    assert card.benchmark_name == "fake-bench"
    assert len(card.detector_results) == 1
    assert card.detector_results[0].aggregate_score == 0.7


def test_audit_marks_inapplicable_detectors_with_skip_reason_not_omits_them():
    model = FakeModel("m")  # supports_logprobs=False
    benchmark = FakeBenchmark([])
    card = audit(model, benchmark, detectors=[LogprobOnlyDetector()])
    assert len(card.detector_results) == 1
    r = card.detector_results[0]
    assert r.applicable is False
    assert r.aggregate_score is None
    assert "does not expose token logprobs" in r.skip_reason


def test_audit_default_detectors_uses_all_registered():
    from leaklens.detectors import ALL_DETECTORS

    model = FakeModel("m")
    benchmark = FakeBenchmark([])
    card = audit(model, benchmark)
    assert len(card.detector_results) == len(ALL_DETECTORS)
    result_names = {r.detector_name for r in card.detector_results}
    assert result_names == set(ALL_DETECTORS)


def test_audit_report_card_has_version_and_timestamp():
    card = audit(FakeModel("m"), FakeBenchmark([]), detectors=[AlwaysApplicableDetector()])
    assert card.leaklens_version
    assert card.generated_at
