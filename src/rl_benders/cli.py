"""Command-line interface for exact and RL-guided Benders experiments."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from rl_benders.benchmark import run_benchmark, save_report_csv, save_report_json
from rl_benders.benders import BendersConfig, BendersResult, solve_with_policy
from rl_benders.control import (
    AdaptiveHeuristicPolicy,
    CutBatchAction,
    CutBatchPolicy,
    FixedActionPolicy,
    RandomPolicy,
)
from rl_benders.domain import StochasticFacilityLocationInstance, load_instance, save_instance
from rl_benders.experiment import ResearchConfig, run_research_experiment, save_research_report
from rl_benders.generator import GeneratorConfig, generate_instance
from rl_benders.rl import QLearningConfig, TabularQPolicy, train_q_policy


def _add_generator_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--facilities", type=int, default=5)
    parser.add_argument("--customers", type=int, default=8)
    parser.add_argument("--scenarios", type=int, default=8)
    parser.add_argument("--spatial-regime", choices=("uniform", "clustered"), default="uniform")
    parser.add_argument("--demand-regime", choices=("stable", "volatile"), default="stable")
    parser.add_argument("--seed", type=int, default=0)


def _generated_instance(
    args: argparse.Namespace,
    *,
    seed: int | None = None,
) -> StochasticFacilityLocationInstance:
    return generate_instance(
        GeneratorConfig(
            facility_count=args.facilities,
            customer_count=args.customers,
            scenario_count=args.scenarios,
            spatial_regime=args.spatial_regime,
            demand_regime=args.demand_regime,
            seed=args.seed if seed is None else seed,
        )
    )


def _load_or_generate(args: argparse.Namespace) -> StochasticFacilityLocationInstance:
    return load_instance(args.input) if args.input else _generated_instance(args)


def _policy(mode: str, checkpoint: str | None, seed: int) -> CutBatchPolicy:
    if mode == "all":
        return FixedActionPolicy(CutBatchAction.ALL)
    if mode == "one":
        return FixedActionPolicy(CutBatchAction.ONE)
    if mode == "quarter":
        return FixedActionPolicy(CutBatchAction.QUARTER)
    if mode == "half":
        return FixedActionPolicy(CutBatchAction.HALF)
    if mode == "heuristic":
        return AdaptiveHeuristicPolicy()
    if mode == "random":
        return RandomPolicy(seed=seed)
    if checkpoint is None:
        raise ValueError("--checkpoint is required for mode=q")
    policy = TabularQPolicy.load(checkpoint, seed=seed)
    policy.epsilon = 0.0
    return policy


def _result_payload(result: BendersResult, *, include_records: bool) -> dict[str, object]:
    payload = asdict(result)
    if not include_records:
        payload.pop("records", None)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rl-benders",
        description="Certified Benders decomposition with reinforcement-learning cut control.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="write a deterministic synthetic instance")
    _add_generator_arguments(generate)
    generate.add_argument("--output", type=Path, required=True)

    solve = subparsers.add_parser("solve", help="solve one instance with a selected cut policy")
    solve.add_argument("--input", type=Path)
    _add_generator_arguments(solve)
    solve.add_argument(
        "--mode",
        choices=("all", "one", "quarter", "half", "heuristic", "random", "q"),
        default="all",
    )
    solve.add_argument("--checkpoint")
    solve.add_argument("--max-policy-decisions", type=int, default=80)
    solve.add_argument("--include-records", action="store_true")
    solve.add_argument("--output", type=Path)

    train = subparsers.add_parser("train", help="train a tabular Q-policy on generated instances")
    _add_generator_arguments(train)
    train.add_argument("--instances", type=int, default=24)
    train.add_argument("--episodes", type=int, default=240)
    train.add_argument("--max-policy-decisions", type=int, default=80)
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--output-report", type=Path)

    benchmark = subparsers.add_parser(
        "benchmark", help="compare fixed, heuristic, random, and optional Q policies"
    )
    _add_generator_arguments(benchmark)
    benchmark.add_argument("--instances", type=int, default=8)
    benchmark.add_argument("--checkpoint")
    benchmark.add_argument("--max-policy-decisions", type=int, default=80)
    benchmark.add_argument("--output-json", type=Path)
    benchmark.add_argument("--output-csv", type=Path)

    research = subparsers.add_parser(
        "research", help="run the frozen train/distribution-shift protocol"
    )
    research.add_argument("--train-instances", type=int, default=24)
    research.add_argument("--evaluation-instances", type=int, default=8)
    research.add_argument("--facilities", type=int, default=5)
    research.add_argument("--customers", type=int, default=8)
    research.add_argument("--scenarios", type=int, default=8)
    research.add_argument("--episodes", type=int, default=240)
    research.add_argument("--seed", type=int, default=2026)
    research.add_argument("--max-policy-decisions", type=int, default=80)
    research.add_argument("--checkpoint", type=Path, required=True)
    research.add_argument("--output-report", type=Path, required=True)
    return parser


def _write_or_print(payload: dict[str, object], output: Path | None) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if output is None:
        print(text, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            instance = _generated_instance(args)
            save_instance(instance, args.output)
            _write_or_print({"output": str(args.output), **instance.to_dict()}, None)
            return 0

        if args.command == "solve":
            instance = _load_or_generate(args)
            policy = _policy(args.mode, args.checkpoint, args.seed)
            result = solve_with_policy(
                instance,
                policy,
                config=BendersConfig(max_policy_decisions=args.max_policy_decisions),
            )
            _write_or_print(
                _result_payload(result, include_records=args.include_records),
                args.output,
            )
            return 0

        if args.command == "train":
            if args.instances <= 0:
                raise ValueError("--instances must be positive")
            instances = [
                _generated_instance(args, seed=args.seed + index) for index in range(args.instances)
            ]
            policy, summary = train_q_policy(
                instances,
                q_config=QLearningConfig(episodes=args.episodes, seed=args.seed),
                benders_config=BendersConfig(max_policy_decisions=args.max_policy_decisions),
            )
            policy.save(args.checkpoint)
            payload = {"checkpoint": str(args.checkpoint), **summary.to_dict()}
            _write_or_print(payload, args.output_report)
            return 0

        if args.command == "benchmark":
            if args.instances <= 0:
                raise ValueError("--instances must be positive")
            instances = [
                _generated_instance(args, seed=args.seed + index) for index in range(args.instances)
            ]
            q_policy = (
                TabularQPolicy.load(args.checkpoint, seed=args.seed) if args.checkpoint else None
            )
            report = run_benchmark(
                instances,
                q_policy=q_policy,
                benders_config=BendersConfig(max_policy_decisions=args.max_policy_decisions),
                random_seed=args.seed,
            )
            if args.output_json:
                save_report_json(report, args.output_json)
            if args.output_csv:
                save_report_csv(report, args.output_csv)
            _write_or_print(report.to_dict(), None)
            return 0

        config = ResearchConfig(
            train_instances=args.train_instances,
            evaluation_instances=args.evaluation_instances,
            facilities=args.facilities,
            customers=args.customers,
            scenarios=args.scenarios,
            episodes=args.episodes,
            seed=args.seed,
        )
        policy, report = run_research_experiment(
            config,
            benders_config=BendersConfig(max_policy_decisions=args.max_policy_decisions),
        )
        policy.save(args.checkpoint)
        save_research_report(report, args.output_report)
        _write_or_print(
            {
                "checkpoint": str(args.checkpoint),
                "report": str(args.output_report),
                "training_certification_rate": report.training["certification_rate"],
            },
            None,
        )
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}))
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
