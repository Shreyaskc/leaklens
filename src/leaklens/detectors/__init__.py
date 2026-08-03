from .base import Detector
from .guided_completion import GuidedCompletionDetector, similarity_ratio, split_prefix_continuation
from .min_k_prob import MinKProbDetector, min_k_percent_logprob
from .ngram_overlap import KNOWN_INDEXES, NgramOverlapDetector
from .order_canary import OrderCanaryDetector
from .paraphrase_gap import ParaphraseGapDetector
from .perplexity_gap import PerplexityGapDetector, cross_entropy_nats, perplexity_gap_score, zlib_compressed_length

#: Phase-1 detectors (behavioral + corpus-overlap; work on any model) plus
#: Phase-2 logprob detectors (local models only -- see
#: ModelInterface.supports_logprobs / Detector.requires_logprobs).
#: paraphrase_gap additionally needs per-item paraphrase data that isn't
#: shipped yet (see paraphrase_gap.py's module docstring) -- it's registered
#: because the detector logic is real and testable, not because the
#: benchmark data pipeline for it is complete.
ALL_DETECTORS: dict[str, type[Detector]] = {
    "ngram_overlap": NgramOverlapDetector,
    "guided_completion": GuidedCompletionDetector,
    "order_canary": OrderCanaryDetector,
    "min_k_prob": MinKProbDetector,
    "perplexity_gap": PerplexityGapDetector,
    "paraphrase_gap": ParaphraseGapDetector,
}

__all__ = [
    "Detector",
    "NgramOverlapDetector",
    "GuidedCompletionDetector",
    "OrderCanaryDetector",
    "MinKProbDetector",
    "PerplexityGapDetector",
    "ParaphraseGapDetector",
    "KNOWN_INDEXES",
    "similarity_ratio",
    "split_prefix_continuation",
    "min_k_percent_logprob",
    "cross_entropy_nats",
    "perplexity_gap_score",
    "zlib_compressed_length",
    "ALL_DETECTORS",
]
