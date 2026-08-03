"""Unit test for MLXModelInterface's adapter_path plumbing, mocking mlx_lm.load
so it doesn't need a real adapter file on disk."""
from unittest.mock import MagicMock, patch


def test_mlx_model_interface_without_adapter_calls_load_with_repo_only():
    fake_load = MagicMock(return_value=(MagicMock(), MagicMock()))
    with patch("mlx_lm.load", fake_load):
        from leaklens.models import MLXModelInterface

        m = MLXModelInterface("some/repo")
    fake_load.assert_called_once_with("some/repo")
    assert m.name == "some/repo"


def test_mlx_model_interface_with_adapter_passes_adapter_path():
    fake_load = MagicMock(return_value=(MagicMock(), MagicMock()))
    with patch("mlx_lm.load", fake_load):
        from leaklens.models import MLXModelInterface

        m = MLXModelInterface("some/repo", adapter_path="/tmp/my_adapter")
    fake_load.assert_called_once_with("some/repo", adapter_path="/tmp/my_adapter")
    assert m.name == "some/repo+adapter:/tmp/my_adapter"
