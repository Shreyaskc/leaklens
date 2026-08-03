"""Real-hardware integration tests for TransformersModelInterface against
an already-downloaded model (EleutherAI/pythia-410m, used in the WikiMIA
external-validation experiment -- see paper/main.tex). Mirrors
test_models.py's real-MLX-hardware pattern: skipped automatically if
torch/transformers aren't installed, rather than mocked, since the value
here is confirming actual numerical behavior, not just that functions get
called."""
import pytest

MODEL_REPO = "EleutherAI/pythia-410m"


def _transformers_available():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark_transformers = pytest.mark.skipif(not _transformers_available(), reason="torch/transformers not installed")


@pytest.fixture(scope="module")
def transformers_model():
    if not _transformers_available():
        pytest.skip("torch/transformers not installed")
    from leaklens.models import TransformersModelInterface

    try:
        return TransformersModelInterface(MODEL_REPO)
    except Exception as e:
        pytest.skip(f"could not load {MODEL_REPO}: {e}")


@pytestmark_transformers
def test_transformers_generate_produces_nonempty_text(transformers_model):
    text = transformers_model.generate("The capital of France is", max_tokens=10)
    assert isinstance(text, str)
    assert len(text) > 0


@pytestmark_transformers
def test_transformers_token_logprobs_are_all_nonpositive(transformers_model):
    lps = transformers_model.token_logprobs("The quick brown fox jumps over the lazy dog.")
    assert len(lps) > 0
    assert all(lp.logprob <= 1e-6 for lp in lps)


@pytestmark_transformers
def test_transformers_token_logprobs_short_text_returns_empty(transformers_model):
    assert transformers_model.token_logprobs("") == []


@pytestmark_transformers
def test_transformers_gibberish_scores_lower_than_natural_text(transformers_model):
    natural = transformers_model.token_logprobs("The sun rises in the east and sets in the west.")
    gibberish = transformers_model.token_logprobs("Zxjq plorf 7! banana wrench quietly=99 xkcd")
    natural_mean = sum(lp.logprob for lp in natural) / len(natural)
    gibberish_mean = sum(lp.logprob for lp in gibberish) / len(gibberish)
    assert natural_mean > gibberish_mean
