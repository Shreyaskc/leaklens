"""Order canary: tests whether a model has memorized the benchmark's exact
published item ordering, independent of item content -- in the tradition of
the "canary string" and order-sensitivity contamination probes (e.g. the
GPT-3 paper's contamination appendix checked models for benchmark-order
artifacts). If a model can predict what comes *next* in the benchmark's
sequence better than chance, that's a content-independent memorization
signal: genuine task competence has no reason to encode "and this dataset's
next question is...".

Method: for each adjacent pair (item[i], item[i+1]) in the benchmark's
canonical order, prompt the model with the tail of item[i]'s text and score
how closely the completion matches the head of item[i+1]'s text (the real
"next" item). Compare against a shuffled-control baseline: the same prompt
scored against a *randomly reassigned* "next" item. A real-pair score
reliably above the shuffled-control score is the order-memorization signal;
if both are similar, any completion similarity is just generic
content-level chance, not order memorization.
"""
from __future__ import annotations

import random

from ..base import Benchmark, DetectorResult, ModelInterface, now_iso
from .base import Detector
from .guided_completion import similarity_ratio

DEFAULT_PROMPT_TAIL_CHARS = 200
DEFAULT_CONTINUATION_HEAD_CHARS = 100


class OrderCanaryDetector(Detector):
    name = "order_canary"
    requires_logprobs = False
    source_citation = "Order-sensitivity / canary-style contamination probing, in the tradition of Brown et al. 2020 (GPT-3)'s benchmark-contamination appendix"

    def __init__(
        self,
        prompt_tail_chars: int = DEFAULT_PROMPT_TAIL_CHARS,
        continuation_head_chars: int = DEFAULT_CONTINUATION_HEAD_CHARS,
        max_completion_tokens: int = 64,
        shuffle_seed: int = 0,
    ):
        self.prompt_tail_chars = prompt_tail_chars
        self.continuation_head_chars = continuation_head_chars
        self.max_completion_tokens = max_completion_tokens
        self.shuffle_seed = shuffle_seed

    def run(self, model: ModelInterface, benchmark: Benchmark, **kwargs) -> DetectorResult:
        items = sorted(benchmark.items(), key=lambda it: it.order_index)
        if len(items) < 3:
            return DetectorResult(
                detector_name=self.name,
                applicable=False,
                skip_reason="benchmark has fewer than 3 items; order_canary needs adjacent pairs plus a shuffled control",
                aggregate_score=None,
            )

        rng = random.Random(self.shuffle_seed)
        # Shuffled control mapping: for each index i, a "fake next" index
        # that is NOT i+1 and not i itself.
        indices = list(range(len(items)))
        shuffled = indices[:]
        rng.shuffle(shuffled)
        for i in range(len(shuffled)):
            if shuffled[i] in (i, i + 1):
                swap_with = (i + 1) % len(shuffled)
                shuffled[i], shuffled[swap_with] = shuffled[swap_with], shuffled[i]

        real_scores: dict[str, float] = {}
        control_scores: dict[str, float] = {}
        for i in range(len(items) - 1):
            current, true_next, fake_next = items[i], items[i + 1], items[shuffled[i]]
            if len(current.text) < self.prompt_tail_chars or len(true_next.text) < self.continuation_head_chars:
                continue
            prompt = current.text[-self.prompt_tail_chars :]
            completion = model.generate(prompt, max_tokens=self.max_completion_tokens)

            true_head = true_next.text[: self.continuation_head_chars]
            real_scores[current.item_id] = similarity_ratio(completion[: len(true_head)], true_head)

            if len(fake_next.text) >= self.continuation_head_chars:
                fake_head = fake_next.text[: self.continuation_head_chars]
                control_scores[current.item_id] = similarity_ratio(completion[: len(fake_head)], fake_head)

        if not real_scores:
            return DetectorResult(
                detector_name=self.name,
                applicable=False,
                skip_reason="no adjacent item pairs met the minimum length thresholds",
                aggregate_score=None,
            )

        real_mean = sum(real_scores.values()) / len(real_scores)
        control_mean = sum(control_scores.values()) / len(control_scores) if control_scores else 0.0
        gap = real_mean - control_mean

        return DetectorResult(
            detector_name=self.name,
            applicable=True,
            skip_reason=None,
            aggregate_score=gap,
            per_item_scores=real_scores,
            metadata={
                "real_pair_mean_similarity": real_mean,
                "shuffled_control_mean_similarity": control_mean,
                "n_pairs_scored": len(real_scores),
                "generated_at": now_iso(),
                "scope_caveat": (
                    "aggregate_score is REAL-MINUS-CONTROL similarity, not raw "
                    "similarity -- a value near 0 means no order-memorization "
                    "signal beyond content-level chance, not 'no contamination "
                    "at all' (other detectors probe content memorization directly)."
                ),
            },
        )
