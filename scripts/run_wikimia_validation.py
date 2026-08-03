"""External validation: run leaklens's logprob-based detectors against
WikiMIA (Shi et al. 2023's own published membership-inference benchmark:
https://huggingface.co/datasets/swj0419/WikiMIA), which has real ground-truth
member/non-member labels from actual model training runs -- not a
synthetic LoRA fine-tune like scripts/run_calibration.py. This directly
answers the reviewer-facing gap of "no comparison to an established
benchmark": the calibration pilot validates the DETECTORS' mechanics on a
model we controlled end to end; this validates them against real,
independently-constructed ground truth.

label=1 means the text WAS seen during the reference model's pretraining
(member); label=0 means it was not (non-member) -- verified against the
dataset's own documentation, not assumed.

We do NOT reproduce Shi et al.'s original reported AUCs exactly: WikiMIA's
splits were calibrated against older models (LLaMA-1/2, GPT-Neo, OPT,
Pythia, text-davinci) with training cutoffs specific to those releases.
We run a different, newer, locally-available model (Llama-3.2-3B-Instruct)
instead, so this is a genuine, independent re-check of whether leaklens's
detector implementations separate real member/non-member text on this
established benchmark -- not a reproduction of the paper's own numbers,
and the paper states that distinction explicitly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from leaklens.base import Benchmark, BenchmarkItem
from leaklens.detectors import MinKProbDetector, PerplexityGapDetector
from leaklens.models import MLXModelInterface

MODEL_REPO = "mlx-community/Llama-3.2-3B-Instruct-4bit"
SPLIT = "WikiMIA_length64"
OUTPUT_PATH = Path(__file__).parent.parent / "calibration_runs" / "wikimia_validation_results.json"


class ListBenchmark(Benchmark):
    def __init__(self, name, items):
        self.name = name
        self._items = items

    def items(self):
        return self._items


def auc_roc(scores_positive, scores_negative):
    n_pos, n_neg = len(scores_positive), len(scores_negative)
    count = 0.0
    for sp in scores_positive:
        for sn in scores_negative:
            if sp > sn:
                count += 1.0
            elif sp == sn:
                count += 0.5
    return count / (n_pos * n_neg)


def main():
    from datasets import load_dataset

    ds = load_dataset("swj0419/WikiMIA", split=SPLIT)
    items = []
    labels = []
    for i, row in enumerate(ds):
        items.append(BenchmarkItem(item_id=f"wikimia-{i}", text=row["input"], order_index=i))
        labels.append(row["label"])

    print(f"Loaded {len(items)} WikiMIA items ({SPLIT}): {sum(labels)} member, {len(labels)-sum(labels)} non-member", file=sys.stderr)

    print("Loading model...", file=sys.stderr)
    model = MLXModelInterface(MODEL_REPO)
    benchmark = ListBenchmark("wikimia", items)

    results = {}
    for name, detector in [("min_k_prob", MinKProbDetector()), ("perplexity_gap", PerplexityGapDetector())]:
        print(f"Running {name}...", file=sys.stderr)
        result = detector.run(model, benchmark)

        member_scores = [result.per_item_scores[it.item_id] for it, lbl in zip(items, labels) if lbl == 1 and it.item_id in result.per_item_scores]
        nonmember_scores = [result.per_item_scores[it.item_id] for it, lbl in zip(items, labels) if lbl == 0 and it.item_id in result.per_item_scores]

        # perplexity_gap's convention is LOWER = more member-like (opposite of min_k_prob) -- see its module docstring.
        if name == "perplexity_gap":
            member_scores = [-s for s in member_scores]
            nonmember_scores = [-s for s in nonmember_scores]

        auc = auc_roc(member_scores, nonmember_scores)
        results[name] = {
            "auc": auc,
            "n_member": len(member_scores),
            "n_nonmember": len(nonmember_scores),
        }
        print(f"  {name}: AUC={auc:.4f} (n_member={len(member_scores)}, n_nonmember={len(nonmember_scores)})", file=sys.stderr)

    output = {
        "benchmark": "WikiMIA (Shi et al. 2023)",
        "split": SPLIT,
        "model": MODEL_REPO,
        "n_items_total": len(items),
        "scope_note": (
            "Independent re-check on an established benchmark, NOT a reproduction "
            "of Shi et al.'s original reported AUCs -- WikiMIA's splits were "
            "calibrated against older models (LLaMA-1/2, GPT-Neo, OPT, Pythia, "
            "text-davinci) with different training cutoffs than the model used here."
        ),
        "results": results,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {OUTPUT_PATH}", file=sys.stderr)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
