"""Concrete ModelInterface implementations.

`CallableModelInterface` wraps any provider's SDK behind a plain Python
callable, the same duck-typing philosophy as wattbench's API wrappers — it
keeps leaklens from hard-depending on the OpenAI/Anthropic/Google SDKs.
`MLXModelInterface` wraps a local MLX-LM model and exposes real per-token
logprobs on Apple Silicon without a rented GPU. `TransformersModelInterface`
wraps any HF `transformers` causal LM instead -- needed for models with no
MLX conversion available (e.g. the older GPT-NeoX/Pythia/OPT-era models
that contamination benchmarks like WikiMIA were actually calibrated
against), at the cost of slower CPU/MPS inference vs. MLX's Metal backend.
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

    def __init__(self, model_repo: str, adapter_path: str | None = None):
        from mlx_lm import load

        self.name = model_repo if adapter_path is None else f"{model_repo}+adapter:{adapter_path}"
        self._model, self._tokenizer = (
            load(model_repo, adapter_path=adapter_path) if adapter_path else load(model_repo)
        )

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


class TransformersModelInterface(ModelInterface):
    """Local HF `transformers` causal LM: same capability as MLXModelInterface
    (behavioral probes + real per-token logprobs), for models without an
    MLX conversion. Uses MPS (Apple Silicon GPU via Metal) when available,
    falling back to CPU."""

    supports_logprobs = True

    def __init__(self, model_repo: str, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_repo
        self._device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(model_repo)
        self._model = AutoModelForCausalLM.from_pretrained(model_repo).to(self._device)
        self._model.eval()

    def generate(self, prompt: str, max_tokens: int) -> str:
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False, pad_token_id=self._tokenizer.eos_token_id
            )
        new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def token_logprobs(self, text: str) -> list[TokenLogprob]:
        import torch

        token_ids = self._tokenizer.encode(text)
        if len(token_ids) < 2:
            return []

        input_ids = torch.tensor([token_ids]).to(self._device)
        with torch.no_grad():
            logits = self._model(input_ids).logits
        log_probs = torch.log_softmax(logits.float(), dim=-1)

        results = []
        for i in range(len(token_ids) - 1):
            next_token_id = token_ids[i + 1]
            lp = float(log_probs[0, i, next_token_id])
            token_str = self._tokenizer.decode([next_token_id])
            results.append(TokenLogprob(token=token_str, logprob=lp))
        return results
