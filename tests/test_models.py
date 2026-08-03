import pytest

from leaklens.models import CallableModelInterface, MLXModelInterface


def test_callable_model_interface_generate():
    calls = []

    def fake_generate(prompt, max_tokens):
        calls.append((prompt, max_tokens))
        return "the response"

    m = CallableModelInterface("gpt-4o", fake_generate)
    assert m.generate("hi", max_tokens=10) == "the response"
    assert calls == [("hi", 10)]
    assert m.supports_logprobs is False


def test_callable_model_interface_no_logprobs():
    m = CallableModelInterface("gpt-4o", lambda p, t: "x")
    with pytest.raises(NotImplementedError):
        m.token_logprobs("text")


# --- Real-hardware integration tests against an already-downloaded MLX
# model. These are the load-bearing correctness checks for the logprob math
# that min_k_prob/perplexity_gap/paraphrase_gap will depend on -- verified
# against actual model behavior, not just code review. Skipped automatically
# if mlx_lm or the model isn't available (e.g. in a CI environment without
# Apple Silicon / the cached weights).

MODEL_REPO = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


def _mlx_available():
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark_mlx = pytest.mark.skipif(not _mlx_available(), reason="mlx_lm not installed")


@pytest.fixture(scope="module")
def mlx_model():
    if not _mlx_available():
        pytest.skip("mlx_lm not installed")
    try:
        return MLXModelInterface(MODEL_REPO)
    except Exception as e:
        pytest.skip(f"could not load {MODEL_REPO}: {e}")


@pytestmark_mlx
def test_mlx_generate_produces_nonempty_text(mlx_model):
    text = mlx_model.generate("Say hello in one word.", max_tokens=10)
    assert isinstance(text, str)
    assert len(text) > 0


@pytestmark_mlx
def test_mlx_token_logprobs_are_all_nonpositive(mlx_model):
    lps = mlx_model.token_logprobs("The quick brown fox jumps over the lazy dog.")
    assert len(lps) > 0
    assert all(lp.logprob <= 1e-6 for lp in lps)


@pytestmark_mlx
def test_mlx_token_logprobs_short_text_returns_empty(mlx_model):
    # A single token has no "next token" to score against.
    assert mlx_model.token_logprobs("") == []


@pytestmark_mlx
def test_mlx_repeated_sentence_scores_higher_than_first_occurrence(mlx_model):
    # In-context repetition is a well-known effect: a model should assign
    # much higher probability to a sentence it has *already seen* earlier
    # in the same context than to its first, unprimed occurrence.
    text = "The cat sat on the mat. The cat sat on the mat."
    lps = mlx_model.token_logprobs(text)
    first_half_mean = sum(lp.logprob for lp in lps[:6]) / 6
    second_half_mean = sum(lp.logprob for lp in lps[-6:]) / 6
    assert second_half_mean > first_half_mean


@pytestmark_mlx
def test_mlx_gibberish_scores_lower_than_natural_text(mlx_model):
    natural = mlx_model.token_logprobs("The sun rises in the east and sets in the west.")
    gibberish = mlx_model.token_logprobs("Zxjq plorf 7! banana wrench quietly=99 xkcd")
    natural_mean = sum(lp.logprob for lp in natural) / len(natural)
    gibberish_mean = sum(lp.logprob for lp in gibberish) / len(gibberish)
    assert natural_mean > gibberish_mean
