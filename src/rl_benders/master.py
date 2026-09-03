"""Restricted Benders master problem."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from rl_benders.cuts import BendersCut
from rl_benders.domain import StochasticFacilityLocationInstance


@dataclass(frozen=True, slots=True)
class MasterResult:
    success: bool
    status: str
    objective: float
    y: tuple[int, ...]
    theta: tuple[float, ...]
    mip_node_count: int
    mip_gap: float | None
    runtime_seconds: float
    message: str


def solve_master(
    instance: StochasticFacilityLocationInstance,
    cuts: tuple[BendersCut, ...] | list[BendersCut],
    *,
    mip_rel_gap: float = 0.0,
    time_limit: float | None = None,
) -> MasterResult:
    """Solve the binary first-stage master with scenario-specific theta variables."""

    if mip_rel_gap < 0:
        raise ValueError("mip_rel_gap must be nonnegative")
    if time_limit is not None and time_limit <= 0:
        raise ValueError("time_limit must be positive")
    facility_count = instance.facility_count
    scenario_count = instance.scenario_count
    variable_count = facility_count + scenario_count

    objective = np.concatenate(
        [
            np.asarray(instance.opening_costs, dtype=float),
            np.asarray([scenario.probability for scenario in instance.scenarios], dtype=float),
        ]
    )
    lower = np.concatenate([np.zeros(facility_count), np.zeros(scenario_count)])
    upper = np.concatenate([np.ones(facility_count), np.full(scenario_count, np.inf)])
    integrality = np.concatenate(
        [np.ones(facility_count, dtype=int), np.zeros(scenario_count, dtype=int)]
    )

    constraints: LinearConstraint | None = None
    if cuts:
        rows = np.zeros((len(cuts), variable_count), dtype=float)
        lb = np.zeros(len(cuts), dtype=float)
        for row_index, cut in enumerate(cuts):
            if cut.scenario_index < 0 or cut.scenario_index >= scenario_count:
                raise ValueError("cut scenario index out of range")
            if len(cut.beta) != facility_count:
                raise ValueError("cut coefficient dimension does not match the instance")
            rows[row_index, :facility_count] = -np.asarray(cut.beta, dtype=float)
            rows[row_index, facility_count + cut.scenario_index] = 1.0
            lb[row_index] = cut.alpha
        constraints = LinearConstraint(rows, lb=lb, ub=np.full(len(cuts), np.inf))

    options: dict[str, float | bool] = {"presolve": True, "mip_rel_gap": mip_rel_gap}
    if time_limit is not None:
        options["time_limit"] = time_limit
    started = time.perf_counter()
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options=options,
    )
    runtime = time.perf_counter() - started
    if not result.success or result.x is None or result.fun is None:
        raise RuntimeError(f"Benders master failed: {result.message}")
    values = np.asarray(result.x, dtype=float)
    y = tuple(int(round(value)) for value in values[:facility_count])
    theta = tuple(float(max(0.0, value)) for value in values[facility_count:])
    node_count_raw = getattr(result, "mip_node_count", 0)
    gap_raw = getattr(result, "mip_gap", None)
    node_count = int(node_count_raw) if node_count_raw is not None else 0
    gap = float(gap_raw) if gap_raw is not None else None
    return MasterResult(
        success=True,
        status="optimal" if result.status == 0 else "feasible",
        objective=float(result.fun),
        y=y,
        theta=theta,
        mip_node_count=node_count,
        mip_gap=gap,
        runtime_seconds=runtime,
        message=str(result.message),
    )
