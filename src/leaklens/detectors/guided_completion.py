"""Guided completion: prompt the model with a prefix of a benchmark item and
measure how closely it reproduces the true continuation. High-fidelity
verbatim completion of a prefix the model was never shown at inference time
is a classic memorization signal (used e.g. in Carlini et al.'s training
data extraction work and widely in the "can the model complete this exact
benchmark item" contamination-probing literature).

Works on ANY model with generate() -- local or API -- unlike the
logprob-based detectors.
"""
from __future__ import annotations

import difflib

from ..base import Benchmark, BenchmarkItem, DetectorResult, ModelInterface, now_iso
from .base import Detector


def split_prefix_continuation(text: str, prefix_fraction: float) -> tuple[str, str]:
    """Split `text` at `prefix_fraction` of its length (by character count,
    not tokens -- keeps this detector tokenizer-agnostic across models)."""
    if not 0.0 < prefix_fraction < 1.0:
        raise ValueError("prefix_fraction must be in (0, 1)")
    split_at = int(len(text) * prefix_fraction)
    return text[:split_at], text[split_at:]


def similarity_ratio(a: str, b: str) -> float:
    """difflib's SequenceMatcher ratio: 2*M / T where M is matching chars
    and T is total length of both strings. 1.0 = identical, 0.0 = disjoint.
    A simple, well-understood, dependency-free similarity metric -- not
    claimed to be state-of-the-art, just transparent and reproducible."""
    return difflib.SequenceMatcher(None, a, b).ratio()


class GuidedCompletionDetector(Detector):
    name = "guided_completion"
    requires_logprobs = False
    source_citation = "Guided-completion / verbatim-extraction probing, in the tradition of Carlini et al. 2021 training-data-extraction attacks"

    def __init__(self, prefix_fraction: float = 0.7, max_completion_tokens: int = 128):
        self.prefix_fraction = prefix_fraction
        self.max_completion_tokens = max_completion_tokens

    def _score_item(self, model: ModelInterface, item: BenchmarkItem) -> float | None:
        if len(item.text) < 20:
            return None  # too short to meaningfully split into prefix/continuation
        prefix, true_continuation = split_prefix_continuation(item.text, self.prefix_fraction)
        completion = model.generate(prefix, max_tokens=self.max_completion_tokens)
        # Compare only up to the length of the true continuation -- the
        # model's free-running completion will keep going past that point,
        # and comparing the full (longer) completion would understate
        # similarity for reasons unrelated to memorization.
        completion_matched_len = completion[: len(true_continuation)]
        return similarity_ratio(completion_matched_len, true_continuation)

    def run(self, model: ModelInterface, benchmark: Benchmark, **kwargs) -> DetectorResult:
        per_item: dict[str, float] = {}
        n_skipped_too_short = 0
        for item in benchmark.items():
            score = self._score_item(model, item)
            if score is None:
                n_skipped_too_short += 1
                continue
            per_item[item.item_id] = score

        aggregate = sum(per_item.values()) / len(per_item) if per_item else None
        return DetectorResult(
            detector_name=self.name,
            applicable=True,
            skip_reason=None,
            aggregate_score=aggregate,
            per_item_scores=per_item,
            metadata={
                "prefix_fraction": self.prefix_fraction,
                "n_items_scored": len(per_item),
                "n_items_skipped_too_short": n_skipped_too_short,
                "generated_at": now_iso(),
                "scope_caveat": (
                    "High similarity indicates the model reproduces this item's "
                    "continuation closely; some similarity is expected by chance "
                    "for short or low-entropy continuations (e.g. multiple-choice "
                    "letter answers) -- interpret alongside a calibration baseline, "
                    "not as a standalone threshold."
                ),
            },
        )
