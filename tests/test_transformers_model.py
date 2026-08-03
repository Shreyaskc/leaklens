"""Unit tests for TransformersModelInterface, mocking torch/transformers so
they don't need real model weights. The real backend was validated live
against EleutherAI/pythia-410m during the WikiMIA external-validation
experiment (see paper/main.tex Section "External Validation on an
Established Benchmark") -- these tests cover the plumbing/edge cases."""
from unittest.mock import MagicMock, patch


def test_transformers_model_interface_loads_and_selects_device():
    fake_torch = MagicMock()
    fake_torch.backends.mps.is_available.return_value = True
    fake_tokenizer = MagicMock()
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model

    with patch.dict(
        "sys.modules",
        {"torch": fake_torch, "transformers": MagicMock(AutoTokenizer=MagicMock(from_pretrained=MagicMock(return_value=fake_tokenizer)), AutoModelForCausalLM=MagicMock(from_pretrained=MagicMock(return_value=fake_model)))},
    ):
        from leaklens.models import TransformersModelInterface

        m = TransformersModelInterface("some/repo")

    assert m.name == "some/repo"
    assert m._device == "mps"
    fake_model.eval.assert_called_once()


def test_transformers_model_interface_falls_back_to_cpu_when_no_mps():
    fake_torch = MagicMock()
    fake_torch.backends.mps.is_available.return_value = False
    fake_tokenizer = MagicMock()
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model

    with patch.dict(
        "sys.modules",
        {"torch": fake_torch, "transformers": MagicMock(AutoTokenizer=MagicMock(from_pretrained=MagicMock(return_value=fake_tokenizer)), AutoModelForCausalLM=MagicMock(from_pretrained=MagicMock(return_value=fake_model)))},
    ):
        from leaklens.models import TransformersModelInterface

        m = TransformersModelInterface("some/repo")

    assert m._device == "cpu"


def test_transformers_model_interface_explicit_device_overrides_detection():
    fake_torch = MagicMock()
    fake_torch.backends.mps.is_available.return_value = True
    fake_tokenizer = MagicMock()
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model

    with patch.dict(
        "sys.modules",
        {"torch": fake_torch, "transformers": MagicMock(AutoTokenizer=MagicMock(from_pretrained=MagicMock(return_value=fake_tokenizer)), AutoModelForCausalLM=MagicMock(from_pretrained=MagicMock(return_value=fake_model)))},
    ):
        from leaklens.models import TransformersModelInterface

        m = TransformersModelInterface("some/repo", device="cpu")

    assert m._device == "cpu"
