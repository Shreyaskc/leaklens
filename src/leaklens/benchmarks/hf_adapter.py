"""Generic HuggingFace-dataset benchmark adapter, plus built-in adapters for
MMLU, GSM8K, HumanEval, ARC, HellaSwag, and TruthfulQA.

Dataset repo IDs, config names, split names, and column names below were
verified live against the HF datasets-server API during development (not
assumed from training-data memory) -- see leaklens's test suite for the
same checks re-run as a CI regression guard against upstream schema drift.
"""
from __future__ import annotations

from ..base import Benchmark, BenchmarkItem


class HFBenchmark(Benchmark):
    """Wraps any HF `datasets.Dataset` with a caller-supplied `text_fn` that
    turns one row into the canonical text string used by the detectors.
    Row order in the loaded split is preserved as `order_index` -- do not
    shuffle before passing to this adapter, or order_canary becomes
    meaningless (it depends on a real, stable published ordering).
    """

    def __init__(self, name: str, repo_id: str, config: str, split: str, text_fn, id_fn=None):
        self.name = name
        self.repo_id = repo_id
        self.config = config
        self.split = split
        self._text_fn = text_fn
        self._id_fn = id_fn
        self._items: list[BenchmarkItem] | None = None

    def items(self) -> list[BenchmarkItem]:
        if self._items is None:
            from datasets import load_dataset

            ds = load_dataset(self.repo_id, self.config, split=self.split)
            built = []
            for i, row in enumerate(ds):
                item_id = self._id_fn(row, i) if self._id_fn else f"{self.name}-{i}"
                built.append(
                    BenchmarkItem(item_id=item_id, text=self._text_fn(row), order_index=i, fields=dict(row))
                )
            self._items = built
        return self._items


def _mmlu_text(row: dict) -> str:
    choices = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(row["choices"]))
    return f"{row['question']}\n{choices}"


def mmlu(subject: str = "all", split: str = "test") -> HFBenchmark:
    return HFBenchmark(
        name=f"mmlu-{subject}",
        repo_id="cais/mmlu",
        config=subject,
        split=split,
        text_fn=_mmlu_text,
        id_fn=lambda row, i: f"mmlu-{subject}-{i}",
    )


def gsm8k(split: str = "test") -> HFBenchmark:
    return HFBenchmark(
        name="gsm8k",
        repo_id="openai/gsm8k",
        config="main",
        split=split,
        text_fn=lambda row: f"{row['question']}\n{row['answer']}",
        id_fn=lambda row, i: f"gsm8k-{i}",
    )


def humaneval() -> HFBenchmark:
    return HFBenchmark(
        name="humaneval",
        repo_id="openai/openai_humaneval",
        config="openai_humaneval",
        split="test",
        text_fn=lambda row: f"{row['prompt']}{row['canonical_solution']}",
        id_fn=lambda row, i: row["task_id"],
    )


def _arc_text(row: dict) -> str:
    labels = row["choices"]["label"]
    texts = row["choices"]["text"]
    choices = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
    return f"{row['question']}\n{choices}"


def arc(challenge: bool = True, split: str = "test") -> HFBenchmark:
    config = "ARC-Challenge" if challenge else "ARC-Easy"
    return HFBenchmark(
        name=f"arc-{'challenge' if challenge else 'easy'}",
        repo_id="allenai/ai2_arc",
        config=config,
        split=split,
        text_fn=_arc_text,
        id_fn=lambda row, i: row["id"],
    )


def hellaswag(split: str = "validation") -> HFBenchmark:
    return HFBenchmark(
        name="hellaswag",
        repo_id="Rowan/hellaswag",
        config="default",
        split=split,
        text_fn=lambda row: row["ctx"] + " " + row["endings"][int(row["label"])] if row["label"] != "" else row["ctx"],
        id_fn=lambda row, i: f"hellaswag-{row['ind']}",
    )


def truthfulqa(split: str = "validation") -> HFBenchmark:
    return HFBenchmark(
        name="truthfulqa",
        repo_id="truthfulqa/truthful_qa",
        config="multiple_choice",
        split=split,
        text_fn=lambda row: row["question"],
        id_fn=lambda row, i: f"truthfulqa-{i}",
    )
