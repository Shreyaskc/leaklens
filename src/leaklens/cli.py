"""`leaklens audit --model X --benchmark gsm8k`"""
from __future__ import annotations

import argparse
import sys

from . import benchmarks as bm
from .audit import audit
from .detectors import ALL_DETECTORS
from .models import MLXModelInterface

BENCHMARK_FACTORIES = {
    "mmlu": bm.mmlu,
    "gsm8k": bm.gsm8k,
    "humaneval": bm.humaneval,
    "arc": bm.arc,
    "hellaswag": bm.hellaswag,
    "truthfulqa": bm.truthfulqa,
}


def build_model(model_arg: str):
    if model_arg.startswith("mlx:"):
        return MLXModelInterface(model_arg[len("mlx:") :])
    raise ValueError(
        f"Unrecognized --model {model_arg!r}. CLI currently supports local MLX "
        "models via the 'mlx:<repo_id>' prefix (e.g. mlx:mlx-community/Qwen2.5-0.5B-Instruct-4bit). "
        "For API models, use the Python API with leaklens.CallableModelInterface directly."
    )


def build_benchmark(benchmark_arg: str):
    if benchmark_arg not in BENCHMARK_FACTORIES:
        raise ValueError(f"Unknown --benchmark {benchmark_arg!r}. Known: {', '.join(BENCHMARK_FACTORIES)}")
    return BENCHMARK_FACTORIES[benchmark_arg]()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="leaklens")
    sub = parser.add_subparsers(dest="command", required=True)

    audit_parser = sub.add_parser("audit", help="Run a contamination audit and print/save a report card")
    audit_parser.add_argument("--model", required=True, help="e.g. mlx:mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    audit_parser.add_argument("--benchmark", required=True, choices=list(BENCHMARK_FACTORIES))
    audit_parser.add_argument(
        "--detectors", default=None, help=f"Comma-separated subset of {list(ALL_DETECTORS)}; default: all"
    )
    audit_parser.add_argument("--output", default=None, help="Write the report card JSON to this path")
    audit_parser.add_argument("--markdown", action="store_true", help="Print the Markdown table instead of JSON")

    args = parser.parse_args(argv)
    if args.command == "audit":
        model = build_model(args.model)
        benchmark = build_benchmark(args.benchmark)

        selected = None
        if args.detectors:
            names = [d.strip() for d in args.detectors.split(",")]
            unknown = set(names) - set(ALL_DETECTORS)
            if unknown:
                print(f"Unknown detector(s): {unknown}. Known: {list(ALL_DETECTORS)}", file=sys.stderr)
                sys.exit(1)
            selected = [ALL_DETECTORS[n]() for n in names]

        report = audit(model, benchmark, detectors=selected)
        text = report.to_markdown() if args.markdown else report.to_json()
        if args.output:
            with open(args.output, "w") as f:
                f.write(text)
            print(f"Wrote report card to {args.output}", file=sys.stderr)
        else:
            print(text)


if __name__ == "__main__":
    main()
