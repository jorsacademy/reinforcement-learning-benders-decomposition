"""Repeated-instance benchmarks for classical and RL-controlled Benders policies."""

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from rl_benders.benders import BendersConfig, BendersResult, RewardConfig, solve_with_policy
from rl_benders.control import (
    AdaptiveHeuristicPolicy,
    CutBatchAction,
    CutBatchPolicy,
    FixedActionPolicy,
    RandomPolicy,
)
from rl_benders.domain import StochasticFacilityLocationInstance
from rl_benders.oracle import OracleResult, solve_extensive_form
from rl_benders.rl import TabularQPolicy


@dataclass(frozen=True, slots=True)
class BenchmarkRow:
    instance: str
    method: str
    facilities: int
    customers: int
    scenarios: int
    objective: float
    reference_objective: float
    objective_gap_percent: float
    lower_bound: float
    upper_bound: float
    certified: bool
    y: tuple[int, ...]
    policy_decisions: int
    completion_iterations: int
    completion_phase_used: bool
    cut_count: int
    master_solves: int
    subproblem_solves: int
    master_nodes: int
    master_seconds: float
    subproblem_seconds: float
    total_seconds: float
    cumulative_reward: float
    action_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    rows: tuple[BenchmarkRow, ...]
    summary: dict[str, dict[str, float]]
    references: dict[str, dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "rows": [row.to_dict() for row in self.rows],
            "summary": self.summary,
            "references": self.references,
        }


def _gap_percent(value: float, reference: float) -> float:
    return 100.0 * (value - reference) / max(1.0, abs(reference))


def _row(
    instance: StochasticFacilityLocationInstance,
    result: BendersResult,
    reference: OracleResult,
) -> BenchmarkRow:
    return BenchmarkRow(
        instance=instance.name,
        method=result.policy_name,
        facilities=instance.facility_count,
        customers=instance.customer_count,
        scenarios=instance.scenario_count,
        objective=result.objective,
        reference_objective=reference.objective,
        objective_gap_percent=_gap_percent(result.objective, reference.objective),
        lower_bound=result.lower_bound,
        upper_bound=result.upper_bound,
        certified=result.globally_certified,
        y=result.y,
        policy_decisions=result.policy_decisions,
        completion_iterations=result.completion_iterations,
        completion_phase_used=result.completion_phase_used,
        cut_count=result.cut_count,
        master_solves=result.master_solves,
        subproblem_solves=result.subproblem_solves,
        master_nodes=result.total_master_nodes,
        master_seconds=result.total_master_seconds,
        subproblem_seconds=result.total_subproblem_seconds,
        total_seconds=result.total_runtime_seconds,
        cumulative_reward=result.cumulative_reward,
        action_counts=result.action_counts,
    )


def _confidence_half_width(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def _attribute_values(rows: list[BenchmarkRow], attribute: str) -> list[float]:
    return [float(getattr(row, attribute)) for row in rows]


def _summarize(rows: list[BenchmarkRow]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for method in sorted({row.method for row in rows}):
        selected = [row for row in rows if row.method == method]

        gaps = _attribute_values(selected, "objective_gap_percent")
        times = _attribute_values(selected, "total_seconds")
        decisions = _attribute_values(selected, "policy_decisions")
        cuts = _attribute_values(selected, "cut_count")
        rewards = _attribute_values(selected, "cumulative_reward")
        nodes = _attribute_values(selected, "master_nodes")
        summary[method] = {
            "instances": float(len(selected)),
            "certified_rate": statistics.fmean(float(row.certified) for row in selected),
            "completion_phase_rate": statistics.fmean(
                float(row.completion_phase_used) for row in selected
            ),
            "mean_objective_gap_percent": statistics.fmean(gaps),
            "max_objective_gap_percent": max(gaps),
            "mean_policy_decisions": statistics.fmean(decisions),
            "mean_cut_count": statistics.fmean(cuts),
            "mean_master_nodes": statistics.fmean(nodes),
            "mean_total_seconds": statistics.fmean(times),
            "total_seconds_ci95_half_width": _confidence_half_width(times),
            "mean_cumulative_reward": statistics.fmean(rewards),
        }
    return summary


def run_benchmark(
    instances: list[StochasticFacilityLocationInstance]
    | tuple[StochasticFacilityLocationInstance, ...],
    *,
    q_policy: TabularQPolicy | None = None,
    benders_config: BendersConfig | None = None,
    reward_config: RewardConfig | None = None,
    include_random: bool = True,
    random_seed: int = 0,
) -> BenchmarkReport:
    """Compare policies on identical instances against monolithic exact solutions."""

    if not instances:
        raise ValueError("instances must be nonempty")
    benders_config = benders_config or BendersConfig()
    reward_config = reward_config or RewardConfig()
    rows: list[BenchmarkRow] = []
    references: dict[str, dict[str, object]] = {}

    for instance_index, instance in enumerate(instances):
        reference = solve_extensive_form(instance)
        references[instance.name] = {
            "method": reference.method,
            "objective": reference.objective,
            "y": list(reference.y),
            "runtime_seconds": reference.runtime_seconds,
            "mip_node_count": reference.mip_node_count,
        }
        policies: list[CutBatchPolicy] = [
            FixedActionPolicy(CutBatchAction.ALL),
            FixedActionPolicy(CutBatchAction.ONE),
            FixedActionPolicy(CutBatchAction.HALF),
            AdaptiveHeuristicPolicy(),
        ]
        if include_random:
            policies.append(RandomPolicy(seed=random_seed + instance_index))
        if q_policy is not None:
            q_policy.epsilon = 0.0
            policies.append(q_policy)

        for policy in policies:
            result = solve_with_policy(
                instance,
                policy,
                config=benders_config,
                reward_config=reward_config,
                safe_complete=True,
            )
            if not result.globally_certified:
                raise RuntimeError(f"policy {policy.name} returned an uncertified result")
            if abs(result.objective - reference.objective) > 1e-5 * max(
                1.0, abs(reference.objective)
            ):
                raise RuntimeError(
                    "certified Benders objective disagrees with extensive form for "
                    f"{instance.name}: {result.objective} vs {reference.objective}"
                )
            rows.append(_row(instance, result, reference))

    return BenchmarkReport(
        rows=tuple(rows),
        summary=_summarize(rows),
        references=references,
    )


def save_report_json(report: BenchmarkReport, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def save_report_csv(report: BenchmarkReport, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [row.to_dict() for row in report.rows]
    if not rows:
        raise ValueError("report has no rows")
    flattened: list[dict[str, object]] = []
    for row in rows:
        copy = dict(row)
        copy["y"] = json.dumps(copy["y"])
        copy["action_counts"] = json.dumps(copy["action_counts"], sort_keys=True)
        flattened.append(copy)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)
