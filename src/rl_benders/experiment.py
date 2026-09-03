"""Frozen train/test protocol for RL-controlled Benders decomposition."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from rl_benders.benchmark import BenchmarkReport, run_benchmark
from rl_benders.benders import BendersConfig, RewardConfig
from rl_benders.domain import (
    DemandRegime,
    SpatialRegime,
    StochasticFacilityLocationInstance,
)
from rl_benders.generator import GeneratorConfig, generate_instance
from rl_benders.rl import QLearningConfig, TabularQPolicy, TrainingSummary, train_q_policy


@dataclass(frozen=True, slots=True)
class ResearchConfig:
    train_instances: int = 24
    evaluation_instances: int = 8
    facilities: int = 5
    customers: int = 8
    scenarios: int = 8
    episodes: int = 240
    seed: int = 2026

    def __post_init__(self) -> None:
        values = (
            self.train_instances,
            self.evaluation_instances,
            self.facilities,
            self.customers,
            self.scenarios,
            self.episodes,
        )
        if any(value <= 0 for value in values):
            raise ValueError("all research configuration counts must be positive")


@dataclass(frozen=True, slots=True)
class ResearchReport:
    configuration: dict[str, object]
    training: dict[str, object]
    scenarios: dict[str, dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "configuration": self.configuration,
            "training": self.training,
            "scenarios": self.scenarios,
        }


def _instances(
    count: int,
    *,
    seed_start: int,
    facilities: int,
    customers: int,
    scenarios: int,
    spatial_regime: SpatialRegime = "uniform",
    demand_regime: DemandRegime = "stable",
) -> list[StochasticFacilityLocationInstance]:
    return [
        generate_instance(
            GeneratorConfig(
                facility_count=facilities,
                customer_count=customers,
                scenario_count=scenarios,
                spatial_regime=spatial_regime,
                demand_regime=demand_regime,
                seed=seed_start + offset,
            )
        )
        for offset in range(count)
    ]


def _compact_benchmark(report: BenchmarkReport) -> dict[str, object]:
    return {
        "summary": report.summary,
        "rows": [row.to_dict() for row in report.rows],
        "references": report.references,
    }


def run_research_experiment(
    config: ResearchConfig | None = None,
    *,
    benders_config: BendersConfig | None = None,
    reward_config: RewardConfig | None = None,
) -> tuple[TabularQPolicy, ResearchReport]:
    """Train on one distribution and evaluate on disjoint structural shifts."""

    config = config or ResearchConfig()
    benders_config = benders_config or BendersConfig()
    reward_config = reward_config or RewardConfig()
    train_seed_start = config.seed * 10_000
    evaluation_seed_start = train_seed_start + 1_000_000

    training_instances = _instances(
        config.train_instances,
        seed_start=train_seed_start,
        facilities=config.facilities,
        customers=config.customers,
        scenarios=config.scenarios,
    )
    q_config = QLearningConfig(episodes=config.episodes, seed=config.seed)
    policy, training_summary = train_q_policy(
        training_instances,
        q_config=q_config,
        benders_config=benders_config,
        reward_config=reward_config,
    )

    scenario_instances = {
        "in_distribution": _instances(
            config.evaluation_instances,
            seed_start=evaluation_seed_start,
            facilities=config.facilities,
            customers=config.customers,
            scenarios=config.scenarios,
        ),
        "demand_volatility_shift": _instances(
            config.evaluation_instances,
            seed_start=evaluation_seed_start + 100_000,
            facilities=config.facilities,
            customers=config.customers,
            scenarios=config.scenarios,
            demand_regime="volatile",
        ),
        "spatial_cluster_shift": _instances(
            config.evaluation_instances,
            seed_start=evaluation_seed_start + 200_000,
            facilities=config.facilities,
            customers=config.customers,
            scenarios=config.scenarios,
            spatial_regime="clustered",
        ),
        "scenario_count_shift": _instances(
            config.evaluation_instances,
            seed_start=evaluation_seed_start + 300_000,
            facilities=config.facilities,
            customers=config.customers,
            scenarios=max(config.scenarios + 2, 2 * config.scenarios),
        ),
        "problem_size_shift": _instances(
            config.evaluation_instances,
            seed_start=evaluation_seed_start + 400_000,
            facilities=config.facilities + 1,
            customers=config.customers + 3,
            scenarios=config.scenarios,
        ),
    }

    scenario_reports: dict[str, dict[str, object]] = {}
    for name, instances in scenario_instances.items():
        scenario_reports[name] = _compact_benchmark(
            run_benchmark(
                instances,
                q_policy=policy,
                benders_config=benders_config,
                reward_config=reward_config,
                random_seed=config.seed,
            )
        )

    report = ResearchReport(
        configuration={
            **asdict(config),
            "train_seed_start": train_seed_start,
            "evaluation_seed_start": evaluation_seed_start,
            "benders_config": asdict(benders_config),
            "reward_config": asdict(reward_config),
            "q_learning_config": asdict(q_config),
        },
        training=training_summary.to_dict(),
        scenarios=scenario_reports,
    )
    policy.metadata["research_config"] = asdict(config)
    return policy, report


def save_research_report(report: ResearchReport, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def save_training_summary(summary: TrainingSummary, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
