"""Paraphrase gap: compares model cross-entropy on a benchmark item's
original text vs. a meaning-preserving paraphrase of the same content. A
model that is genuinely competent at the underlying task should find the
original and a well-formed paraphrase roughly comparably easy; a model that
finds the EXACT original phrasing much easier than an equivalent paraphrase
is showing surface-form memorization rather than task competence -- the
classic distinguishing signature of contamination vs. genuine capability.

Score: cross_entropy(paraphrase) - cross_entropy(original), in nats/token.
Positive = original is easier than its paraphrase (contamination-suggestive
direction); near zero = no meaningful gap.

SCOPE LIMITATION (stated plainly, not silently glossed over): this detector
needs a paraphrase for each item, supplied via
`BenchmarkItem.fields["paraphrase"]`. The README's Phase 2 scope calls for
shipping human-spot-checked paraphrases as data for leaklens's built-in
benchmarks (MMLU/GSM8K/HumanEval/ARC/HellaSwag/TruthfulQA) -- that
paraphrase-generation-and-validation pipeline is a separate, not-yet-built
deliverable (real content creation + human review, not just code). This
module implements the DETECTOR logic against whatever paraphrase field is
present; items lacking one are skipped and counted, not silently dropped.

Requires per-token logprobs -- local models only.
"""
from __future__ import annotations

from ..base import Benchmark, DetectorResult, ModelInterface, now_iso
from .base import Detector
from .perplexity_gap import cross_entropy_nats

PARAPHRASE_FIELD = "paraphrase"


class ParaphraseGapDetector(Detector):
    name = "paraphrase_gap"
    requires_logprobs = True
    source_citation = (
        "Paraphrase/perturbation-robustness membership-inference probing, in the tradition of "
        "Carlini et al. 2020's training-data-extraction work and subsequent perturbation-based MIA studies"
    )

    def run(self, model: ModelInterface, benchmark: Benchmark, **kwargs) -> DetectorResult:
        per_item: dict[str, float] = {}
        n_no_paraphrase = 0
        n_skipped_too_short = 0

        for item in benchmark.items():
            paraphrase_text = item.fields.get(PARAPHRASE_FIELD)
            if not paraphrase_text:
                n_no_paraphrase += 1
                continue

            original_lps = model.token_logprobs(item.text)
            paraphrase_lps = model.token_logprobs(paraphrase_text)
            if len(original_lps) < 5 or len(paraphrase_lps) < 5:
                n_skipped_too_short += 1
                continue

            original_ce = cross_entropy_nats([tlp.logprob for tlp in original_lps])
            paraphrase_ce = cross_entropy_nats([tlp.logprob for tlp in paraphrase_lps])
            per_item[item.item_id] = paraphrase_ce - original_ce

        aggregate = sum(per_item.values()) / len(per_item) if per_item else None
        return DetectorResult(
            detector_name=self.name,
            applicable=True,
            skip_reason=None,
            aggregate_score=aggregate,
            per_item_scores=per_item,
            metadata={
                "n_items_scored": len(per_item),
                "n_items_no_paraphrase_available": n_no_paraphrase,
                "n_items_skipped_too_short": n_skipped_too_short,
                "generated_at": now_iso(),
                "scope_caveat": (
                    f"Requires item.fields[{PARAPHRASE_FIELD!r}]; leaklens does not "
                    "yet ship pre-generated paraphrases for its built-in benchmarks "
                    "(a separate content-creation deliverable, not code) -- most "
                    "audits will see a high n_items_no_paraphrase_available until "
                    "that data exists or the caller supplies their own paraphrases."
                ),
            },
        )
