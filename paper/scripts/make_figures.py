"""Generates the paper's figures: the real AUC-comparison chart (from
calibration/WikiMIA results files) and the toolkit architecture diagram.
Run from the repo root: python paper/scripts/make_figures.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#7a7972"
GRIDLINE = "#e3e2db"
AXIS = "#c3c2b7"
BLUE = "#2a6fae"
AMBER = "#c8853a"
CHANCE_LINE = "#b0453f"
LOGPROB_FILL = "#e8f0f9"
LOGPROB_EDGE = "#2a6fae"
BEHAVIORAL_FILL = "#faf1e6"
BEHAVIORAL_EDGE = "#c8853a"

ROOT = Path(__file__).parent.parent.parent
CALIB_PATH = ROOT / "calibration_runs" / "calibration_results.json"
WIKIMIA_LLAMA_PATH = ROOT / "calibration_runs" / "wikimia_validation_results.json"
WIKIMIA_PYTHIA_PATH = ROOT / "calibration_runs" / "wikimia_validation_pythia410m.json"
OUT_PATH = Path(__file__).parent.parent / "figures" / "auc_comparison.pdf"
ARCH_OUT_PATH = Path(__file__).parent.parent / "figures" / "architecture.pdf"


def _style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def main():
    calib = json.loads(CALIB_PATH.read_text())["detectors"]
    wikimia_llama = json.loads(WIKIMIA_LLAMA_PATH.read_text())["results"]
    wikimia_pythia = json.loads(WIKIMIA_PYTHIA_PATH.read_text())["results"]

    calib_labels = ["guided_completion", "min_k_prob", "perplexity_gap", "order_canary"]
    calib_aucs = [calib[k]["auc"] for k in calib_labels]

    ext_labels = ["min_k_prob\n(Llama-3.2-3B)", "perplexity_gap\n(Llama-3.2-3B)", "min_k_prob\n(Pythia-410M)", "perplexity_gap\n(Pythia-410M)"]
    ext_aucs = [
        wikimia_llama["min_k_prob"]["auc"],
        wikimia_llama["perplexity_gap"]["auc"],
        wikimia_pythia["min_k_prob"]["auc"],
        wikimia_pythia["perplexity_gap"]["auc"],
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.1), gridspec_kw={"width_ratios": [1, 1.15]})
    fig.patch.set_facecolor("white")

    ax1.bar(range(len(calib_labels)), calib_aucs, color=BLUE, width=0.55, zorder=3)
    ax1.axhline(0.5, color=CHANCE_LINE, linestyle="--", linewidth=1.1, zorder=2)
    ax1.set_xticks(range(len(calib_labels)))
    ax1.set_xticklabels([l.replace("_", "\n") for l in calib_labels], fontsize=8)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("AUC-ROC", color=INK_PRIMARY, fontsize=10)
    ax1.set_title("Calibration pilot (n=15, LoRA fine-tune)", fontsize=9.5, color=INK_PRIMARY)
    _style_axis(ax1)

    ax2.bar(range(len(ext_labels)), ext_aucs, color=AMBER, width=0.55, zorder=3)
    ax2.axhline(0.5, color=CHANCE_LINE, linestyle="--", linewidth=1.1, zorder=2, label="chance (0.5)")
    ax2.set_xticks(range(len(ext_labels)))
    ax2.set_xticklabels(ext_labels, fontsize=7.5)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("WikiMIA external validation", fontsize=9.5, color=INK_PRIMARY)
    ax2.legend(loc="upper right", fontsize=8, frameon=False)
    _style_axis(ax2)

    fig.tight_layout()
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")


def _box(ax, xy, w, h, text, fill, edge, fontsize=8.2):
    box = FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=fill, edgecolor=edge, linewidth=1.2, zorder=3,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, color=INK_PRIMARY, zorder=4)
    return (xy[0] + w / 2, xy[1]), (xy[0] + w / 2, xy[1] + h)  # (bottom-center, top-center)


def _arrow(ax, start, end, color=INK_MUTED):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, color=color, linewidth=1.1, zorder=2))


def make_architecture_diagram():
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8.6)
    ax.axis("off")

    # Inputs: Model + Benchmark
    model_bottom, model_top = _box(ax, (0.4, 7.2), 2.6, 1.0, "Model\n(local: MLX/transformers\nor API: CallableModelInterface)", "#f2f2ef", AXIS)
    bench_bottom, bench_top = _box(ax, (7.6, 7.2), 3.0, 1.0, "Benchmark\n(MMLU, GSM8K, HumanEval,\nARC, HellaSwag, TruthfulQA)", "#f2f2ef", AXIS)

    # audit() entry point
    audit_bottom, audit_top = _box(ax, (3.8, 5.7), 3.4, 0.9, "leaklens.audit(model, benchmark)", "#eeeeee", INK_PRIMARY, fontsize=8.8)
    _arrow(ax, (model_bottom[0], model_bottom[1]), (audit_top[0] - 0.6, audit_top[1]))
    _arrow(ax, (bench_bottom[0], bench_bottom[1]), (audit_top[0] + 0.6, audit_top[1]))

    # Six detectors, two rows of three -- colored by requires_logprobs.
    detectors = [
        ("ngram_overlap", False),
        ("guided_completion", False),
        ("order_canary", False),
        ("min_k_prob", True),
        ("perplexity_gap", True),
        ("paraphrase_gap", True),
    ]
    det_w, det_h, gap = 1.55, 0.85, 0.25
    total_w = 6 * det_w + 5 * gap
    x0 = (11 - total_w) / 2
    y = 4.3
    det_positions = []
    for i, (name, needs_logprobs) in enumerate(detectors):
        x = x0 + i * (det_w + gap)
        fill, edge = (LOGPROB_FILL, LOGPROB_EDGE) if needs_logprobs else (BEHAVIORAL_FILL, BEHAVIORAL_EDGE)
        bottom, top = _box(ax, (x, y), det_w, det_h, name, fill, edge, fontsize=7.3)
        det_positions.append((bottom, top, x + det_w / 2))
        _arrow(ax, (audit_bottom[0], audit_bottom[1]), (x + det_w / 2, top[1]))

    # Legend for the two detector colors.
    _box(ax, (0.4, 2.55), 0.3, 0.25, "", LOGPROB_FILL, LOGPROB_EDGE)
    ax.text(0.85, 2.68, "requires token logprobs (local model only)", ha="left", va="center", fontsize=7.2, color=INK_MUTED)
    _box(ax, (0.4, 2.2), 0.3, 0.25, "", BEHAVIORAL_FILL, BEHAVIORAL_EDGE)
    ax.text(0.85, 2.33, "behavioral (any model with generate())", ha="left", va="center", fontsize=7.2, color=INK_MUTED)

    # ReportCard output -- single converging arrow from the detector row's midpoint.
    rc_bottom, rc_top = _box(ax, (3.8, 0.5), 3.4, 0.9, "ReportCard\n(per detector: ran? score? why not?)", "#eeeeee", INK_PRIMARY, fontsize=8.4)
    mid_x = x0 + total_w / 2
    _arrow(ax, (mid_x, y), (rc_top[0], rc_top[1]))

    fig.savefig(ARCH_OUT_PATH, bbox_inches="tight")
    print(f"Wrote {ARCH_OUT_PATH}")


if __name__ == "__main__":
    main()
    make_architecture_diagram()
