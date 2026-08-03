"""Min-K% Probability (Shi et al. 2023, "Detecting Pretraining Data from
Large Language Models"). Intuition: for text a model has memorized
(training member), even its LEAST likely tokens tend to have relatively
high probability, because the model has seen the exact sequence before. For
non-member text, the model's worst tokens are much worse -- genuine
uncertainty shows up in the tail, not just the average.

Score: the mean log-probability of the k% lowest-probability tokens in the
text (a less negative, i.e. higher, score is the contamination-suggestive
direction -- the opposite convention from raw perplexity). This is the
detector Shi et al. found to outperform plain perplexity/loss thresholding
for membership inference on pretraining corpora.

Requires per-token logprobs -- local models only (see
ModelInterface.supports_logprobs).
"""
from __future__ import annotations

from ..base import Benchmark, DetectorResult, ModelInterface, now_iso
from .base import Detector


def min_k_percent_logprob(logprobs: list[float], k_percent: float) -> float:
    """Mean of the lowest k_percent% of the given logprobs (already-negative
    log-probabilities). Raises on an empty list or an out-of-range k_percent
    -- silently returning e.g. 0.0 for an empty list would look like a valid
    score rather than "no data"."""
    if not logprobs:
        raise ValueError("logprobs must be non-empty")
    if not 0.0 < k_percent <= 100.0:
        raise ValueError("k_percent must be in (0, 100]")
    n_bottom = max(1, round(len(logprobs) * k_percent / 100.0))
    bottom_k = sorted(logprobs)[:n_bottom]  # most negative (least likely) first
    return sum(bottom_k) / len(bottom_k)


class MinKProbDetector(Detector):
    name = "min_k_prob"
    requires_logprobs = True
    source_citation = "Shi et al. 2023, 'Detecting Pretraining Data from Large Language Models' (Min-K% Prob)"

    def __init__(self, k_percent: float = 20.0):
        self.k_percent = k_percent

    def run(self, model: ModelInterface, benchmark: Benchmark, **kwargs) -> DetectorResult:
        per_item: dict[str, float] = {}
        n_skipped = 0
        for item in benchmark.items():
            token_logprobs = model.token_logprobs(item.text)
            if len(token_logprobs) < 5:
                n_skipped += 1
                continue
            lps = [tlp.logprob for tlp in token_logprobs]
            per_item[item.item_id] = min_k_percent_logprob(lps, self.k_percent)

        aggregate = sum(per_item.values()) / len(per_item) if per_item else None
        return DetectorResult(
            detector_name=self.name,
            applicable=True,
            skip_reason=None,
            aggregate_score=aggregate,
            per_item_scores=per_item,
            metadata={
                "k_percent": self.k_percent,
                "n_items_scored": len(per_item),
                "n_items_skipped_too_short": n_skipped,
                "generated_at": now_iso(),
                "scope_caveat": (
                    "Higher (less negative) scores suggest membership -- this is "
                    "a RELATIVE, corpus/model-specific signal, not an absolute "
                    "threshold; interpret against a calibration baseline of "
                    "known-clean text scored with the same model, not in isolation."
                ),
            },
        )
