"""The top-level `leaklens.audit()` entry point: runs every applicable
detector against a (model, benchmark) pair and assembles a ReportCard."""
from __future__ import annotations

from . import __version__
from .base import Benchmark, DetectorResult, ModelInterface, ReportCard, now_iso
from .detectors import ALL_DETECTORS
from .detectors.base import Detector


def audit(
    model: ModelInterface,
    benchmark: Benchmark,
    detectors: list[Detector] | None = None,
) -> ReportCard:
    """Run every detector in `detectors` (default: all registered detectors)
    against `model` x `benchmark`. Detectors that aren't applicable to this
    model (e.g. a logprob detector against an API-only model) are still
    listed in the report card with `applicable=False` and a skip_reason,
    rather than silently omitted -- a report card should show what WASN'T
    checked, not just what was.
    """
    if detectors is None:
        detectors = [cls() for cls in ALL_DETECTORS.values()]

    results: list[DetectorResult] = []
    for detector in detectors:
        applicable, skip_reason = detector.applicable(model)
        if not applicable:
            results.append(
                DetectorResult(
                    detector_name=detector.name,
                    applicable=False,
                    skip_reason=skip_reason,
                    aggregate_score=None,
                )
            )
            continue
        results.append(detector.run(model, benchmark))

    return ReportCard(
        model_name=model.name,
        benchmark_name=benchmark.name,
        generated_at=now_iso(),
        detector_results=results,
        leaklens_version=__version__,
    )
