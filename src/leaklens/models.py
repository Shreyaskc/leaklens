"""Concrete ModelInterface implementations.

`CallableModelInterface` wraps any provider's SDK behind a plain Python
callable, the same duck-typing philosophy as wattbench's API wrappers — it
keeps leaklens from hard-depending on the OpenAI/Anthropic/Google SDKs.
`MLXModelInterface` wraps a local MLX-LM model and exposes real per-token
logprobs, so it's the only concrete backend here that satisfies
`supports_logprobs=True` without needing a rented GPU.
"""
from __future__ import annotations

from typing import Callable

from .base import ModelInterface, TokenLogprob


class CallableModelInterface(ModelInterface):
    """For API models: behavioral probes only (guided_completion, order_canary).

        model = CallableModelInterface("gpt-4o", lambda prompt, max_tokens: my_client.complete(prompt, max_tokens))
    """

    supports_logprobs = False

    def __init__(self, name: str, generate_fn: Callable[[str, int], str]):
        self.name = name
        self._generate_fn = generate_fn

    def generate(self, prompt: str, max_tokens: int) -> str:
        return self._generate_fn(prompt, max_tokens)


class MLXModelInterface(ModelInterface):
    """Local MLX-LM model: supports both behavioral probes and logprob-based
    membership-inference probes (min_k_prob, perplexity_gap, paraphrase_gap).
    """

    supports_logprobs = True

    def __init__(self, model_repo: str):
        from mlx_lm import load

        self.name = model_repo
        self._model, self._tokenizer = load(model_repo)

    def generate(self, prompt: str, max_tokens: int) -> str:
        from mlx_lm import generate

        return generate(self._model, self._tokenizer, prompt=prompt, max_tokens=max_tokens)

    def token_logprobs(self, text: str) -> list[TokenLogprob]:
        import mlx.core as mx
        import mlx.nn as nn

        token_ids = self._tokenizer.encode(text)
        if len(token_ids) < 2:
            return []

        input_ids = mx.array(token_ids)[None, :]
        logits = self._model(input_ids)
        log_probs = nn.log_softmax(logits.astype(mx.float32), axis=-1)

        results = []
        # log_probs[0, i] predicts token i+1 given tokens[0..i].
        for i in range(len(token_ids) - 1):
            next_token_id = token_ids[i + 1]
            lp = float(log_probs[0, i, next_token_id])
            token_str = self._tokenizer.decode([next_token_id])
            results.append(TokenLogprob(token=token_str, logprob=lp))
        return results
