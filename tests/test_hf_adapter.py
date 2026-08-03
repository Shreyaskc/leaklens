"""Unit tests mock datasets.load_dataset for determinism/speed. The actual
dataset repo IDs, configs, splits, and column names used by the built-in
adapters (mmlu/gsm8k/humaneval/arc/hellaswag/truthfulqa) were verified live
against the HF datasets-server API during development -- see the module
docstring in leaklens/benchmarks/hf_adapter.py."""
from unittest.mock import patch

from leaklens.benchmarks import arc, gsm8k, hellaswag, humaneval, mmlu, truthfulqa
from leaklens.benchmarks.hf_adapter import HFBenchmark


def test_hf_benchmark_preserves_row_order_as_order_index():
    fake_rows = [{"text": "row0"}, {"text": "row1"}, {"text": "row2"}]
    with patch("datasets.load_dataset", return_value=fake_rows):
        bench = HFBenchmark(name="t", repo_id="r", config="c", split="s", text_fn=lambda row: row["text"])
        items = bench.items()
    assert [it.order_index for it in items] == [0, 1, 2]
    assert [it.text for it in items] == ["row0", "row1", "row2"]


def test_hf_benchmark_caches_items():
    fake_rows = [{"text": "row0"}]
    with patch("datasets.load_dataset", return_value=fake_rows) as mock_load:
        bench = HFBenchmark(name="t", repo_id="r", config="c", split="s", text_fn=lambda row: row["text"])
        bench.items()
        bench.items()
    mock_load.assert_called_once()


def test_hf_benchmark_default_id_fn():
    fake_rows = [{"text": "a"}, {"text": "b"}]
    with patch("datasets.load_dataset", return_value=fake_rows):
        bench = HFBenchmark(name="mybench", repo_id="r", config="c", split="s", text_fn=lambda row: row["text"])
        items = bench.items()
    assert items[0].item_id == "mybench-0"
    assert items[1].item_id == "mybench-1"


def test_mmlu_text_fn_formats_question_and_choices():
    fake_rows = [{"question": "What is 2+2?", "choices": ["3", "4", "5", "6"], "answer": 1, "subject": "math"}]
    with patch("datasets.load_dataset", return_value=fake_rows):
        bench = mmlu(subject="all")
        items = bench.items()
    assert "What is 2+2?" in items[0].text
    assert "A. 3" in items[0].text
    assert "B. 4" in items[0].text
    assert bench.repo_id == "cais/mmlu"


def test_gsm8k_text_fn():
    fake_rows = [{"question": "Q?", "answer": "A."}]
    with patch("datasets.load_dataset", return_value=fake_rows):
        bench = gsm8k()
        items = bench.items()
    assert items[0].text == "Q?\nA."
    assert bench.repo_id == "openai/gsm8k"
    assert bench.config == "main"


def test_humaneval_text_fn_and_id():
    fake_rows = [{"task_id": "HumanEval/0", "prompt": "def f():\n", "canonical_solution": "    return 1\n"}]
    with patch("datasets.load_dataset", return_value=fake_rows):
        items = humaneval().items()
    assert items[0].text == "def f():\n    return 1\n"
    assert items[0].item_id == "HumanEval/0"


def test_arc_text_fn():
    fake_rows = [{"id": "arc1", "question": "Q?", "choices": {"label": ["A", "B"], "text": ["opt1", "opt2"]}}]
    with patch("datasets.load_dataset", return_value=fake_rows):
        bench = arc(challenge=True)
        items = bench.items()
    assert "Q?" in items[0].text
    assert "A. opt1" in items[0].text
    assert bench.config == "ARC-Challenge"


def test_arc_easy_uses_easy_config():
    with patch("datasets.load_dataset", return_value=[]):
        bench = arc(challenge=False)
    assert bench.config == "ARC-Easy"


def test_hellaswag_text_fn_uses_labeled_ending():
    fake_rows = [{"ind": 7, "ctx": "A man walks", "endings": ["into a bar.", "away."], "label": "1"}]
    with patch("datasets.load_dataset", return_value=fake_rows):
        items = hellaswag().items()
    assert items[0].text == "A man walks away."
    assert items[0].item_id == "hellaswag-7"


def test_truthfulqa_text_fn():
    fake_rows = [{"question": "Is the earth flat?"}]
    with patch("datasets.load_dataset", return_value=fake_rows):
        bench = truthfulqa()
        items = bench.items()
    assert items[0].text == "Is the earth flat?"
    assert bench.config == "multiple_choice"
