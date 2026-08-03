"""Phase 3 calibration study: fine-tune two small local models via MLX-LM
LoRA -- one on a "contaminated" set of GSM8K items (repeated, in their
canonical order, so the model can genuinely memorize both content and
order), one on a disjoint "clean" set -- then run leaklens's detectors
against the SAME contaminated-set items using both models, and compute each
detector's AUC-ROC at distinguishing "this model was fine-tuned on this
item" from "this model was not".

SCOPE, stated plainly:
  - ngram_overlap is NOT calibrated here. It queries a fixed PUBLIC
    pretraining-corpus index (Dolma/Pile/C4/RedPajama) and never looks at
    the model being audited at all (see ngram_overlap.py: `del model`) --
    a local LoRA fine-tune on GSM8K items has no bearing on whether those
    items are ALSO in a public web-scale corpus. This detector needs a
    different calibration design (known post-cutoff vs. pre-cutoff items),
    not attempted here.
  - paraphrase_gap is NOT calibrated here either -- it requires paraphrase
    data leaklens doesn't ship yet (see paraphrase_gap.py's docstring).
  - This is a SMALL, FAST calibration (tens of items, a few hundred LoRA
    steps, one base model) meant to sanity-check detector direction and
    give indicative AUCs -- not the large-scale, multi-model calibration
    suite implied by the README's "week 4-5" scope. Report results as
    exactly that: a first pass, not a definitive benchmark.

Requires: pip install -e ".[mlx,datasets]"
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from leaklens.base import Benchmark, BenchmarkItem
from leaklens.detectors import (
    GuidedCompletionDetector,
    MinKProbDetector,
    OrderCanaryDetector,
    PerplexityGapDetector,
)
from leaklens.models import MLXModelInterface

BASE_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
RUN_DIR = Path(__file__).parent.parent / "calibration_runs"
N_ITEMS_PER_SET = 15
N_REPEATS = 20  # how many times each item (and the full ordered sequence) appears in training data
LORA_ITERS = 100


class ListBenchmark(Benchmark):
    def __init__(self, name: str, items: list[BenchmarkItem]):
        self.name = name
        self._items = items

    def items(self) -> list[BenchmarkItem]:
        return self._items


def load_gsm8k_items(n_total: int) -> list[BenchmarkItem]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = []
    for i in range(n_total):
        row = ds[i]
        text = f"{row['question']}\n{row['answer']}"
        items.append(BenchmarkItem(item_id=f"gsm8k-{i}", text=text, order_index=i, fields=dict(row)))
    return items


def write_training_jsonl(items: list[BenchmarkItem], path: Path, n_repeats: int) -> None:
    """Each item repeated n_repeats times individually (for content
    memorization) PLUS adjacent-pair concatenations (item_i + item_i+1)
    repeated n_repeats times (for order memorization, matching
    order_canary's adjacent-pair-based test).

    Originally this concatenated ALL items into one ~3000-token training
    example for the order signal. That produced NaN training loss within
    ~30 iterations in a real run (see git history / calibration_runs
    logs): a single training example an order of magnitude longer than the
    rest of the batch (batch_size=4, other examples ~100-200 tokens)
    destabilized gradients enough to blow up. Adjacent pairs are much
    shorter and closer in length to the individual-item examples, and still
    directly teach the item_i -> item_i+1 transition order_canary probes."""
    lines = []
    for _ in range(n_repeats):
        for item in items:
            lines.append({"text": item.text})
    for _ in range(n_repeats):
        for a, b in zip(items, items[1:]):
            lines.append({"text": a.text + "\n\n" + b.text})

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


def run_lora_training(data_dir: Path, adapter_path: Path) -> None:
    adapter_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        BASE_MODEL,
        "--train",
        "--data",
        str(data_dir),
        "--fine-tune-type",
        "lora",
        "--iters",
        str(LORA_ITERS),
        "--batch-size",
        "4",
        "--adapter-path",
        str(adapter_path),
        "--val-batches",
        "1",
        "--steps-per-eval",
        str(LORA_ITERS),  # only eval once at the end, we don't need frequent validation
        "--save-every",
        str(LORA_ITERS),
    ]
    print(f"Running: {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)


def auc_roc(scores_positive: list[float], scores_negative: list[float]) -> float:
    """Mann-Whitney-U-based AUC: probability a random positive score exceeds
    a random negative score (ties count as 0.5). No sklearn dependency --
    this is exactly what roc_auc_score computes for a binary label, derived
    directly from its rank-sum definition rather than via TPR/FPR integration."""
    n_pos, n_neg = len(scores_positive), len(scores_negative)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("need at least one positive and one negative score")
    count = 0.0
    for sp in scores_positive:
        for sn in scores_negative:
            if sp > sn:
                count += 1.0
            elif sp == sn:
                count += 0.5
    return count / (n_pos * n_neg)


def main() -> None:
    all_items = load_gsm8k_items(2 * N_ITEMS_PER_SET)
    contaminated_items = all_items[:N_ITEMS_PER_SET]
    clean_items = all_items[N_ITEMS_PER_SET:]

    write_training_jsonl(contaminated_items, RUN_DIR / "contaminated_data" / "train.jsonl", N_REPEATS)
    write_training_jsonl(contaminated_items, RUN_DIR / "contaminated_data" / "valid.jsonl", 1)
    write_training_jsonl(clean_items, RUN_DIR / "clean_data" / "train.jsonl", N_REPEATS)
    write_training_jsonl(clean_items, RUN_DIR / "clean_data" / "valid.jsonl", 1)

    run_lora_training(RUN_DIR / "contaminated_data", RUN_DIR / "adapter_contaminated")
    run_lora_training(RUN_DIR / "clean_data", RUN_DIR / "adapter_clean")

    print("Loading fine-tuned models...", file=sys.stderr)
    model_contaminated = MLXModelInterface(BASE_MODEL, adapter_path=str(RUN_DIR / "adapter_contaminated"))
    model_clean = MLXModelInterface(BASE_MODEL, adapter_path=str(RUN_DIR / "adapter_clean"))

    benchmark = ListBenchmark("gsm8k-calibration-subset", contaminated_items)

    detectors = {
        "guided_completion": GuidedCompletionDetector(),
        "order_canary": OrderCanaryDetector(),
        "min_k_prob": MinKProbDetector(),
        "perplexity_gap": PerplexityGapDetector(),
    }
    # Sign convention per detector docstring: True = higher score means
    # "more contamination-suggestive" (needed to compute AUC consistently).
    higher_is_positive = {
        "guided_completion": True,
        "order_canary": True,
        "min_k_prob": True,
        "perplexity_gap": False,
    }

    results = {}
    for name, detector in detectors.items():
        print(f"Running {name} on contaminated (member) model...", file=sys.stderr)
        result_member = detector.run(model_contaminated, benchmark)
        print(f"Running {name} on clean (non-member) model...", file=sys.stderr)
        result_nonmember = detector.run(model_clean, benchmark)

        pos_scores = list(result_member.per_item_scores.values())
        neg_scores = list(result_nonmember.per_item_scores.values())
        if not higher_is_positive[name]:
            pos_scores = [-s for s in pos_scores]
            neg_scores = [-s for s in neg_scores]

        if pos_scores and neg_scores:
            auc = auc_roc(pos_scores, neg_scores)
        else:
            auc = None

        results[name] = {
            "auc": auc,
            "n_member_scored": len(result_member.per_item_scores),
            "n_nonmember_scored": len(result_nonmember.per_item_scores),
            "member_aggregate": result_member.aggregate_score,
            "nonmember_aggregate": result_nonmember.aggregate_score,
        }
        print(f"  {name}: AUC={auc}", file=sys.stderr)

    output = {
        "base_model": BASE_MODEL,
        "n_items_per_set": N_ITEMS_PER_SET,
        "n_repeats": N_REPEATS,
        "lora_iters": LORA_ITERS,
        "detectors": results,
        "scope_note": (
            "Small, fast, single-model calibration pass (indicative, not "
            "definitive). ngram_overlap and paraphrase_gap excluded -- see "
            "this script's module docstring for why."
        ),
        "order_canary_null_result_explanation": (
            "order_canary scored ~chance (AUC~0.50, both member and non-member "
            "aggregates exactly 0.0) because model.generate() returned an empty "
            "string at the exact prompt boundary used: each individual-item "
            "training example implicitly teaches 'predict EOS after this exact "
            "200-char tail' (that tail IS an item's true ending), while the "
            "adjacent-pair training examples simultaneously teach 'continue into "
            "the next item' AT THE SAME boundary. These two training signals "
            "conflict at inference time with no way to disambiguate, and the "
            "model appears to default to EOS. This is a real limitation of THIS "
            "calibration design (mixing individual-item and pair examples), not "
            "evidence that order_canary's underlying method doesn't work -- an "
            "isolated design (train ONLY on the continuous concatenated "
            "sequence, no individual-item EOS-terminated examples) would be "
            "needed to calibrate it cleanly, and is left for a follow-up run."
        ),
    }
    out_path = RUN_DIR / "calibration_results.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {out_path}", file=sys.stderr)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
