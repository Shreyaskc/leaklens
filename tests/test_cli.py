import json

import pytest

from leaklens.cli import build_benchmark, build_model, main


def test_build_model_mlx_prefix_dispatches(monkeypatch):
    created = {}

    class FakeMLXModel:
        def __init__(self, repo):
            created["repo"] = repo

    monkeypatch.setattr("leaklens.cli.MLXModelInterface", FakeMLXModel)
    build_model("mlx:mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    assert created["repo"] == "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


def test_build_model_unrecognized_prefix_raises():
    with pytest.raises(ValueError, match="Unrecognized --model"):
        build_model("gpt-4o")


def test_build_benchmark_unknown_raises():
    with pytest.raises(ValueError, match="Unknown --benchmark"):
        build_benchmark("not-a-benchmark")


def test_build_benchmark_known_dispatches(monkeypatch):
    called = {}

    def fake_factory():
        called["called"] = True
        return "bench"

    monkeypatch.setattr("leaklens.cli.BENCHMARK_FACTORIES", {"gsm8k": fake_factory})
    result = build_benchmark("gsm8k")
    assert result == "bench"
    assert called["called"] is True


def test_cli_audit_end_to_end_with_fakes(monkeypatch, capsys):
    from leaklens.base import Benchmark, DetectorResult, ModelInterface

    class FakeModel(ModelInterface):
        name = "fake-model"
        supports_logprobs = False

        def generate(self, prompt, max_tokens):
            return "x"

    class FakeBenchmark(Benchmark):
        name = "fake-bench"

        def items(self):
            return []

    class FakeDetector:
        name = "fake_detector"

        def applicable(self, model):
            return True, None

        def run(self, model, benchmark, **kwargs):
            return DetectorResult(detector_name=self.name, applicable=True, skip_reason=None, aggregate_score=0.42)

    monkeypatch.setattr("leaklens.cli.build_model", lambda arg: FakeModel())
    monkeypatch.setattr("leaklens.cli.build_benchmark", lambda arg: FakeBenchmark())
    monkeypatch.setattr("leaklens.cli.ALL_DETECTORS", {"fake_detector": FakeDetector})

    main(["audit", "--model", "mlx:whatever", "--benchmark", "gsm8k", "--detectors", "fake_detector"])
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed["model_name"] == "fake-model"
    assert parsed["detector_results"][0]["aggregate_score"] == 0.42


def test_cli_audit_writes_to_output_file(monkeypatch, tmp_path):
    from leaklens.base import Benchmark, DetectorResult, ModelInterface

    class FakeModel(ModelInterface):
        name = "fake-model"
        supports_logprobs = False

        def generate(self, prompt, max_tokens):
            return "x"

    class FakeBenchmark(Benchmark):
        name = "fake-bench"

        def items(self):
            return []

    class FakeDetector:
        name = "fake_detector"

        def applicable(self, model):
            return True, None

        def run(self, model, benchmark, **kwargs):
            return DetectorResult(detector_name=self.name, applicable=True, skip_reason=None, aggregate_score=0.1)

    monkeypatch.setattr("leaklens.cli.build_model", lambda arg: FakeModel())
    monkeypatch.setattr("leaklens.cli.build_benchmark", lambda arg: FakeBenchmark())
    monkeypatch.setattr("leaklens.cli.ALL_DETECTORS", {"fake_detector": FakeDetector})

    out_path = tmp_path / "report.json"
    main(["audit", "--model", "mlx:whatever", "--benchmark", "gsm8k", "--output", str(out_path)])
    assert out_path.exists()
    parsed = json.loads(out_path.read_text())
    assert parsed["model_name"] == "fake-model"


def test_cli_audit_unknown_detector_exits(monkeypatch, capsys):
    from leaklens.base import Benchmark, ModelInterface

    class FakeModel(ModelInterface):
        name = "m"
        supports_logprobs = False

        def generate(self, prompt, max_tokens):
            return "x"

    class FakeBenchmark(Benchmark):
        name = "b"

        def items(self):
            return []

    monkeypatch.setattr("leaklens.cli.build_model", lambda arg: FakeModel())
    monkeypatch.setattr("leaklens.cli.build_benchmark", lambda arg: FakeBenchmark())

    with pytest.raises(SystemExit):
        main(["audit", "--model", "mlx:whatever", "--benchmark", "gsm8k", "--detectors", "not_a_real_detector"])
