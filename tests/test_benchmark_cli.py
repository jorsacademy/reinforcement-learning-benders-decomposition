from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl_benders.benchmark import run_benchmark, save_report_csv, save_report_json
from rl_benders.cli import main
from rl_benders.domain import StochasticFacilityLocationInstance
from rl_benders.rl import TabularQPolicy


def test_benchmark_policies_are_certified_and_match_reference(
    tiny_instance: StochasticFacilityLocationInstance,
) -> None:
    report = run_benchmark([tiny_instance], q_policy=TabularQPolicy(seed=0))
    assert len(report.rows) == 6
    assert all(row.certified for row in report.rows)
    assert all(abs(row.objective_gap_percent) <= 1e-7 for row in report.rows)
    assert "q_learning" in report.summary


def test_benchmark_writers(
    tmp_path: Path,
    tiny_instance: StochasticFacilityLocationInstance,
) -> None:
    report = run_benchmark([tiny_instance], include_random=False)
    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "report.csv"
    save_report_json(report, json_path)
    save_report_csv(report, csv_path)
    assert json.loads(json_path.read_text())["rows"]
    assert "objective_gap_percent" in csv_path.read_text()


def test_cli_generate_solve_train_and_benchmark(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    instance_path = tmp_path / "instance.json"
    policy_path = tmp_path / "policy.npz"
    report_path = tmp_path / "training.json"
    benchmark_path = tmp_path / "benchmark.json"

    assert (
        main(
            [
                "generate",
                "--facilities",
                "3",
                "--customers",
                "3",
                "--scenarios",
                "3",
                "--seed",
                "10",
                "--output",
                str(instance_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main(["solve", "--input", str(instance_path), "--mode", "all"]) == 0
    solve_payload = json.loads(capsys.readouterr().out)
    assert solve_payload["globally_certified"] is True

    assert (
        main(
            [
                "train",
                "--facilities",
                "3",
                "--customers",
                "3",
                "--scenarios",
                "3",
                "--instances",
                "2",
                "--episodes",
                "2",
                "--seed",
                "20",
                "--checkpoint",
                str(policy_path),
                "--output-report",
                str(report_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert policy_path.exists() and report_path.exists()

    assert (
        main(
            [
                "benchmark",
                "--facilities",
                "3",
                "--customers",
                "3",
                "--scenarios",
                "3",
                "--instances",
                "1",
                "--seed",
                "30",
                "--checkpoint",
                str(policy_path),
                "--output-json",
                str(benchmark_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert json.loads(benchmark_path.read_text())["summary"]["q_learning"]["certified_rate"] == 1.0


def test_tiny_research_protocol_runs_all_distribution_shifts(tmp_path: Path) -> None:
    from rl_benders.benders import BendersConfig
    from rl_benders.experiment import (
        ResearchConfig,
        run_research_experiment,
        save_research_report,
    )

    policy, report = run_research_experiment(
        ResearchConfig(
            train_instances=2,
            evaluation_instances=1,
            facilities=3,
            customers=3,
            scenarios=3,
            episodes=2,
            seed=4,
        ),
        benders_config=BendersConfig(
            max_policy_decisions=20,
            max_completion_iterations=40,
        ),
    )
    research_config = policy.metadata["research_config"]
    assert isinstance(research_config, dict)
    assert research_config["episodes"] == 2
    assert set(report.scenarios) == {
        "in_distribution",
        "demand_volatility_shift",
        "spatial_cluster_shift",
        "scenario_count_shift",
        "problem_size_shift",
    }
    assert all(
        scenario["summary"]["q_learning"]["certified_rate"] == 1.0
        for scenario in report.scenarios.values()
    )
    output = tmp_path / "research.json"
    save_research_report(report, output)
    assert json.loads(output.read_text())["training"]["episodes"]
