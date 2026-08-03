"""Core abstractions shared by every detector: a model interface that works
for both local (logprob-capable) and API (completion-only) models, a
benchmark item/adapter interface, and the report-card data model that
`leaklens.audit()` produces.
"""
from __future__ import annotations

import abc
import datetime as _dt
import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class TokenLogprob:
    token: str
    logprob: float


class ModelInterface(abc.ABC):
    """What a detector is allowed to assume about a model.

    Two capability tiers, matching the README's stated split:
      - Every model supports `generate()` (behavioral probes: guided
        completion, order canary).
      - Only local, logprob-exposing models support `token_logprobs()`
        (membership-inference probes: min-k%, perplexity gap, paraphrase gap).
        API models must raise NotImplementedError so detectors can check
        `supports_logprobs` and skip themselves cleanly rather than fail.
    """

    name: str
    supports_logprobs: bool = False

    @abc.abstractmethod
    def generate(self, prompt: str, max_tokens: int) -> str:
        ...

    def token_logprobs(self, text: str) -> list[TokenLogprob]:
        raise NotImplementedError(
            f"{self.name} does not expose token logprobs (supports_logprobs=False); "
            "logprob-based detectors (min_k_prob, perplexity_gap, paraphrase_gap) "
            "should check ModelInterface.supports_logprobs before calling this."
        )


@dataclass(frozen=True)
class BenchmarkItem:
    item_id: str
    text: str  # canonical full text representation used for overlap/completion probes
    order_index: int  # position in the benchmark's published/canonical ordering
    fields: dict = field(default_factory=dict)  # raw fields (question, answer, choices, ...)


class Benchmark(abc.ABC):
    """A benchmark adapter: turns some raw source (an HF dataset, a local
    file) into a list of BenchmarkItem in a *fixed, documented* canonical
    order — order_canary depends on that order being real and stable, not
    re-shuffled by the loader.
    """

    name: str

    @abc.abstractmethod
    def items(self) -> list[BenchmarkItem]:
        ...


@dataclass
class DetectorResult:
    detector_name: str
    applicable: bool
    skip_reason: str | None
    aggregate_score: float | None  # detector-specific scale; see each detector's docstring
    per_item_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class ReportCard:
    model_name: str
    benchmark_name: str
    generated_at: str
    detector_results: list[DetectorResult]
    leaklens_version: str

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        lines = [
            f"# Contamination report card: `{self.model_name}` x `{self.benchmark_name}`",
            f"_Generated {self.generated_at} with leaklens {self.leaklens_version}_",
            "",
            "| Detector | Applicable | Aggregate score | Notes |",
            "|---|---|---|---|",
        ]
        for r in self.detector_results:
            score = f"{r.aggregate_score:.3f}" if r.aggregate_score is not None else "—"
            notes = r.skip_reason or ""
            lines.append(f"| {r.detector_name} | {'yes' if r.applicable else 'no'} | {score} | {notes} |")
        return "\n".join(lines)


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()
