"""Command-line entry point: run the matrix, build the report, make data.

memorydrift run --model anthropic/claude-sonnet-5-5 --epochs 5
memorydrift report logs/ --out results/
memorydrift scenarios --count 40 --out scenarios.jsonl
"""

from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from memorydrift import __version__
from memorydrift.episode import Condition
from memorydrift.prompts import MemoryPolicy

_ALL_CONDITIONS = ",".join(c.value for c in Condition)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memorydrift",
        description="Agent memory under staleness and poisoning.",
    )
    parser.add_argument(
        "--version", action="version", version=f"memorydrift {__version__}"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser(
        "run", help="Run conditions against one or more models."
    )
    run.add_argument(
        "--model",
        action="append",
        required=True,
        help="Inspect model name. Repeat to run several models.",
    )
    run.add_argument(
        "--conditions",
        default=_ALL_CONDITIONS,
        help=f"Comma-separated conditions. Default: {_ALL_CONDITIONS}.",
    )
    run.add_argument(
        "--memory-policy",
        default=MemoryPolicy.FREE_FORM.value,
        choices=[p.value for p in MemoryPolicy],
    )
    run.add_argument("--epochs", type=int, default=5)
    run.add_argument("--limit", type=int, help="Only the first N scenarios.")
    run.add_argument("--log-dir", default="logs")
    run.add_argument("--scenarios", help="Scenario JSON Lines file.")
    run.add_argument("--max-connections", type=int)
    run.add_argument(
        "--no-feedback",
        action="store_true",
        help="Withhold the post-session review from session logs.",
    )
    run.add_argument(
        "--no-freshness-signal",
        action="store_true",
        help="Withhold the list_amendments tool.",
    )

    report = commands.add_parser(
        "report", help="Build result tables from eval logs."
    )
    report.add_argument("log_dir", type=Path)
    report.add_argument("--out", type=Path, help="Directory for the output.")
    report.add_argument(
        "--prices",
        type=Path,
        help="JSON file of USD per million tokens, by model name.",
    )
    report.add_argument("--k", type=int, help="The k of pass^k.")
    report.add_argument("--resamples", type=int, default=10_000)
    report.add_argument("--seed", type=int, default=0)

    scenarios = commands.add_parser(
        "scenarios", help="Generate a fresh scenario file."
    )
    scenarios.add_argument("--count", type=int, default=40)
    scenarios.add_argument("--first-seed", type=int, default=0)
    scenarios.add_argument("--out", type=Path, required=True)
    return parser


def _run(args: argparse.Namespace) -> int:
    # Imported here so that `memorydrift report` does not pay for it.
    from inspect_ai import eval as inspect_eval

    from memorydrift.task import memorydrift

    conditions = [Condition(c.strip()) for c in args.conditions.split(",")]
    tasks = [
        memorydrift(
            condition=condition.value,
            memory_policy=args.memory_policy,
            feedback=not args.no_feedback,
            freshness_signal=not args.no_freshness_signal,
            scenarios=args.scenarios,
        )
        for condition in conditions
    ]
    logs = inspect_eval(
        tasks,
        model=args.model,
        epochs=args.epochs,
        limit=args.limit,
        log_dir=args.log_dir,
        max_connections=args.max_connections,
        # A sample that errors is reported, never silently scored as wrong.
        fail_on_error=False,
    )
    failed = [log for log in logs if log.status != "success"]
    for log in failed:
        print(f"{log.eval.task} on {log.eval.model}: {log.status}")
    return 1 if failed else 0


def _report(args: argparse.Namespace) -> int:
    from memorydrift.analysis import Price, build_report, load_episodes
    from memorydrift.report import render_markdown, report_to_json

    episodes, exclusions = load_episodes(args.log_dir)
    if not episodes:
        print(f"No scored MemoryDrift episodes under {args.log_dir}.")
        return 1
    prices = None
    if args.prices:
        raw = json.loads(args.prices.read_text(encoding="utf-8"))
        prices = {
            model: Price(**price)
            for model, price in raw.items()
            if not model.startswith("_")
        }
    report = build_report(
        episodes,
        exclusions,
        k=args.k,
        prices=prices,
        resamples=args.resamples,
        seed=args.seed,
    )
    markdown = render_markdown(report)
    if args.out is None:
        print(markdown)
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.md").write_text(markdown, encoding="utf-8")
    payload = {
        "generated_on": datetime.date.today().isoformat(),
        "memorydrift_version": __version__,
        "inspect_ai_version": importlib.metadata.version("inspect-ai"),
        "resamples": args.resamples,
        "confidence": 0.95,
        "seed": args.seed,
        **report_to_json(report),
    }
    (args.out / "results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.out / 'results.md'} and {args.out / 'results.json'}.")
    return 0


def _scenarios(args: argparse.Namespace) -> int:
    from memorydrift.scenario import dump_scenarios, generate_scenarios

    seeds = range(args.first_seed, args.first_seed + args.count)
    dump_scenarios(generate_scenarios(seeds), args.out)
    print(f"Wrote {args.count} scenarios to {args.out}.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the command-line interface.

    Args:
      argv: Arguments, without the program name. Defaults to `sys.argv`.

    Returns:
      Process exit code.
    """
    args = _build_parser().parse_args(argv)
    handlers = {"run": _run, "report": _report, "scenarios": _scenarios}
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
