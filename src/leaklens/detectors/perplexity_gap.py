"""Perplexity-vs-compressibility gap (Carlini et al. 2020, "Extracting
Training Data from Large Language Models" -- the "zlib entropy" baseline).

Intuition: raw perplexity alone confounds two things -- text the model
memorized, and text that's just inherently easy/predictable (repetitive,
boilerplate, low-entropy). zlib compression length is a model-independent
proxy for a text's *inherent* complexity. Comparing the model's
cross-entropy to the text's zlib-compressed size controls for that
confound: text the model finds surprisingly easy RELATIVE TO how complex it
looks structurally is the contamination-suggestive case, not text that's
just naturally simple.

Score: model cross-entropy (nats/token) divided by zlib-compressed byte
length. LOWER values are the contamination-suggestive direction (Carlini et
al. ranked samples by this ratio and manually inspected the lowest-ratio
ones as extraction candidates) -- the opposite direction from min_k_prob's
convention, and documented as such in this detector's metadata so report
cards don't get the sign backwards.

Requires per-token logprobs -- local models only.
"""
from __future__ import annotations

import zlib

from ..base import Benchmark, DetectorResult, ModelInterface, now_iso
from .base import Detector


def cross_entropy_nats(logprobs: list[float]) -> float:
    """Mean negative log-probability per token, in nats (natural log)."""
    if not logprobs:
        raise ValueError("logprobs must be non-empty")
    return -sum(logprobs) / len(logprobs)


def zlib_compressed_length(text: str) -> int:
    return len(zlib.compress(text.encode("utf-8")))


def perplexity_gap_score(logprobs: list[float], text: str) -> float:
    if not text:
        raise ValueError("text must be non-empty")
    # zlib.compress always emits a non-zero-length header+footer (8 bytes
    # for empty input; never 0), so compressed_len can't be 0 here -- the
    # real guard needed is against an empty `text` argument in the first place.
    compressed_len = zlib_compressed_length(text)
    return cross_entropy_nats(logprobs) / compressed_len


class PerplexityGapDetector(Detector):
    name = "perplexity_gap"
    requires_logprobs = True
    source_citation = "Carlini et al. 2020, 'Extracting Training Data from Large Language Models' (zlib-entropy baseline)"

    def run(self, model: ModelInterface, benchmark: Benchmark, **kwargs) -> DetectorResult:
        per_item: dict[str, float] = {}
        n_skipped = 0
        for item in benchmark.items():
            token_logprobs = model.token_logprobs(item.text)
            if len(token_logprobs) < 5:
                n_skipped += 1
                continue
            lps = [tlp.logprob for tlp in token_logprobs]
            per_item[item.item_id] = perplexity_gap_score(lps, item.text)

        aggregate = sum(per_item.values()) / len(per_item) if per_item else None
        return DetectorResult(
            detector_name=self.name,
            applicable=True,
            skip_reason=None,
            aggregate_score=aggregate,
            per_item_scores=per_item,
            metadata={
                "n_items_scored": len(per_item),
                "n_items_skipped_too_short": n_skipped,
                "generated_at": now_iso(),
                "scope_caveat": (
                    "LOWER scores are the contamination-suggestive direction here "
                    "(opposite convention from min_k_prob) -- text the model finds "
                    "surprisingly easy relative to its zlib-measured structural "
                    "complexity. A relative/ranking signal, not an absolute threshold."
                ),
            },
        )
