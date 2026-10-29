from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment import APISettings, ExperimentSettings, run_experiment
from .models import SeedQuestion
from .web import serve


def _read_seeds(path: Path) -> list[SeedQuestion]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("questions", data.get("seeds"))
    if not isinstance(data, list):
        raise ValueError("Seed file must be a JSON list or contain a questions/seeds list.")
    return [SeedQuestion.from_dict(row, index) for index, row in enumerate(data)]


def run(args: argparse.Namespace) -> Path:
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    seeds = _read_seeds(Path(args.seeds))
    run_experiment(
        seeds,
        ExperimentSettings(
            provider=args.provider,
            count=args.count,
            components=args.components,
            clusters=args.clusters,
            activation_quantile=args.activation_quantile,
            neighbor_fraction=args.neighbor_fraction,
            temperature=args.temperature,
            api=APISettings() if args.provider == "openai" else None,
        ),
        output_dir=output,
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="structsynth",
        description="Induce spectral semantic templates and generate isomorphic questions.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run the full experiment.")
    run_parser.add_argument("--seeds", required=True, help="Seed-question JSON file.")
    run_parser.add_argument("--out", required=True, help="Output directory.")
    run_parser.add_argument(
        "--provider", choices=("openai", "mock"), default="openai"
    )
    run_parser.add_argument("--count", type=int, default=8)
    run_parser.add_argument("--components", type=int, default=8)
    run_parser.add_argument("--clusters", type=int, default=8)
    run_parser.add_argument("--activation-quantile", type=float, default=0.80)
    run_parser.add_argument("--neighbor-fraction", type=float, default=0.35)
    run_parser.add_argument("--temperature", type=float, default=0.8)
    run_parser.set_defaults(func=run)

    web_parser = subparsers.add_parser("web", help="Start the local web interface.")
    web_parser.add_argument("--host", default="127.0.0.1")
    web_parser.add_argument("--port", type=int, default=8765)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run":
        output = args.func(args)
        print(json.dumps({"status": "ok", "output": str(output)}, ensure_ascii=False))
    elif args.command == "web":
        serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
