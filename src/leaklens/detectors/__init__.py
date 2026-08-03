from .base import Detector
from .guided_completion import GuidedCompletionDetector, similarity_ratio, split_prefix_continuation
from .ngram_overlap import KNOWN_INDEXES, NgramOverlapDetector
from .order_canary import OrderCanaryDetector

#: Phase-1 detectors (behavioral + corpus-overlap; no local-model requirement
#: except ngram_overlap, which needs no model at all). Phase-2 logprob
#: detectors (min_k_prob, perplexity_gap, paraphrase_gap) register here once
#: implemented.
ALL_DETECTORS: dict[str, type[Detector]] = {
    "ngram_overlap": NgramOverlapDetector,
    "guided_completion": GuidedCompletionDetector,
    "order_canary": OrderCanaryDetector,
}

__all__ = [
    "Detector",
    "NgramOverlapDetector",
    "GuidedCompletionDetector",
    "OrderCanaryDetector",
    "KNOWN_INDEXES",
    "similarity_ratio",
    "split_prefix_continuation",
    "ALL_DETECTORS",
]
