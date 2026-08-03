"""N-gram overlap against public pretraining-corpus indexes, via the
infini-gram public API (Liu et al., 2024, "Infini-gram: Scaling Unbounded
n-gram Language Models to a Trillion Tokens").

Endpoint and request/response schema verified live against
https://api.infini-gram.io/ during development (not assumed from memory):
POST {"index": ..., "query_type": "count", "query": <text>} ->
{"count": int, "approx": bool, "token_ids": [...], "tokens": [...]} on
success, {"error": str} on failure (e.g. unknown index name).

IMPORTANT SCOPE NOTE: this detector answers "does this benchmark item's
text appear in this specific public training corpus?" -- it is a
corpus-level check, not a model-conditioned one. A count > 0 means the text
exists in the queried corpus (e.g. Dolma); it does NOT prove the specific
model being audited was trained on that exact corpus or saw that exact
document. Report cards must not overstate this into "model X saw item Y" --
see leaklens.base.DetectorResult.metadata for the caveat text this detector
attaches.
"""
from __future__ import annotations

import requests

from ..base import Benchmark, DetectorResult, ModelInterface, now_iso
from .base import Detector

API_URL = "https://api.infini-gram.io/"
DEFAULT_INDEX = "v4_dolma-v1_7_llama"
KNOWN_INDEXES = {
    "v4_dolma-v1_7_llama": "Dolma v1.7 (OLMo/Llama tokenizer)",
    "v4_piletrain_llama": "The Pile, train split (Llama tokenizer)",
    "v4_c4train_llama": "C4, train split (Llama tokenizer)",
    "v4_rpj_llama_s4": "RedPajama (Llama tokenizer, shard 4)",
}


class NgramOverlapDetector(Detector):
    name = "ngram_overlap"
    requires_logprobs = False
    source_citation = "Liu et al. 2024, 'Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens'"

    def __init__(self, index: str = DEFAULT_INDEX, timeout_s: float = 15.0, session: requests.Session | None = None):
        if index not in KNOWN_INDEXES:
            raise ValueError(f"Unknown infini-gram index {index!r}. Known: {', '.join(KNOWN_INDEXES)}")
        self.index = index
        self.timeout_s = timeout_s
        self._session = session or requests.Session()

    def query_count(self, text: str) -> int:
        """Raw n-gram count of `text` in the configured corpus index.
        Raises requests.RequestException on network failure or a
        RuntimeError if the API returns an error payload (e.g. text too
        long for a single n-gram query — infini-gram treats the whole
        string as one query, tokenized to its constituent tokens)."""
        resp = self._session.post(
            API_URL,
            json={"index": self.index, "query_type": "count", "query": text},
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        payload = resp.json()
        if "error" in payload:
            raise RuntimeError(f"infini-gram API error for index {self.index!r}: {payload['error']}")
        return payload["count"]

    def run(self, model: ModelInterface, benchmark: Benchmark, **kwargs) -> DetectorResult:
        # model is unused for this detector -- see module docstring's scope note.
        del model
        per_item: dict[str, float] = {}
        errors: dict[str, str] = {}
        for item in benchmark.items():
            try:
                count = self.query_count(item.text)
                per_item[item.item_id] = float(count)
            except (requests.RequestException, RuntimeError) as e:
                errors[item.item_id] = str(e)

        n_contaminated = sum(1 for c in per_item.values() if c > 0)
        n_queried = len(per_item)
        aggregate = (n_contaminated / n_queried) if n_queried else None

        return DetectorResult(
            detector_name=self.name,
            applicable=True,
            skip_reason=None,
            aggregate_score=aggregate,
            per_item_scores=per_item,
            metadata={
                "index": self.index,
                "index_description": KNOWN_INDEXES[self.index],
                "n_items_total": len(benchmark.items()),
                "n_items_queried": n_queried,
                "n_items_query_failed": len(errors),
                "query_errors": errors,
                "generated_at": now_iso(),
                "scope_caveat": (
                    "This detector reports corpus-level presence in the queried "
                    "public index, not proof that the audited model itself was "
                    "trained on this exact corpus or document."
                ),
            },
        )
