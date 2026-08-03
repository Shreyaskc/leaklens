"""Plugin interface every detector implements. See README.md's Deliverables
section for the six detectors this defines slots for."""
from __future__ import annotations

import abc

from ..base import Benchmark, DetectorResult, ModelInterface


class Detector(abc.ABC):
    name: str
    #: True if this detector needs real per-token logprobs (local models
    #: only); False if it only needs generate() (works on API models too).
    requires_logprobs: bool = False
    #: Citation for the method this detector implements, surfaced in report
    #: cards so leaklens is explicitly an aggregator, not a silent reimplementation.
    source_citation: str = ""

    def applicable(self, model: ModelInterface) -> tuple[bool, str | None]:
        if self.requires_logprobs and not model.supports_logprobs:
            return False, f"{model.name} does not expose token logprobs; {self.name} requires a local model"
        return True, None

    @abc.abstractmethod
    def run(self, model: ModelInterface, benchmark: Benchmark, **kwargs) -> DetectorResult:
        ...
