"""Monolithic and finite-enumeration reference solvers."""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from rl_benders.domain import StochasticFacilityLocationInstance
from rl_benders.subproblem import evaluate_recourse


@dataclass(frozen=True, slots=True)
class OracleResult:
    method: str
    objective: float
    y: tuple[int, ...]
    runtime_seconds: float
    mip_node_count: int
    message: str


def first_stage_cost(instance: StochasticFacilityLocationInstance, y: tuple[int, ...]) -> float:
    if len(y) != instance.facility_count:
        raise ValueError("first-stage vector has the wrong length")
    return float(sum(cost * value for cost, value in zip(instance.opening_costs, y, strict=True)))


def solve_extensive_form(
    instance: StochasticFacilityLocationInstance,
    *,
    mip_rel_gap: float = 0.0,
    time_limit: float | None = None,
) -> OracleResult:
    """Solve the complete deterministic equivalent with SciPy/HiGHS."""

    if mip_rel_gap < 0:
        raise ValueError("mip_rel_gap must be nonnegative")
    if time_limit is not None and time_limit <= 0:
        raise ValueError("time_limit must be positive")

    facility_count = instance.facility_count
    customer_count = instance.customer_count
    scenario_count = instance.scenario_count
    scenario_block = customer_count * facility_count + customer_count
    variable_count = facility_count + scenario_count * scenario_block

    objective = np.zeros(variable_count, dtype=float)
    objective[:facility_count] = np.asarray(instance.opening_costs, dtype=float)
    shipping = np.asarray(instance.shipping_costs, dtype=float).reshape(-1)
    shortage = np.asarray(instance.shortage_penalties, dtype=float)
    for scenario_index, scenario in enumerate(instance.scenarios):
        start = facility_count + scenario_index * scenario_block
        objective[start : start + customer_count * facility_count] = (
            scenario.probability * shipping
        )
        objective[
            start + customer_count * facility_count : start + scenario_block
        ] = scenario.probability * shortage

    rows: list[np.ndarray] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    for scenario_index, scenario in enumerate(instance.scenarios):
        block_start = facility_count + scenario_index * scenario_block
        shipment_start = block_start
        unmet_start = block_start + customer_count * facility_count
        for customer in range(customer_count):
            row = np.zeros(variable_count, dtype=float)
            start = shipment_start + customer * facility_count
            row[start : start + facility_count] = 1.0
            row[unmet_start + customer] = 1.0
            rows.append(row)
            lower_bounds.append(scenario.demand[customer])
            upper_bounds.append(np.inf)
        for facility in range(facility_count):
            row = np.zeros(variable_count, dtype=float)
            row[facility] = -instance.facilities[facility].capacity
            for customer in range(customer_count):
                row[shipment_start + customer * facility_count + facility] = 1.0
            rows.append(row)
            lower_bounds.append(-np.inf)
            upper_bounds.append(0.0)

    lower = np.zeros(variable_count, dtype=float)
    upper = np.full(variable_count, np.inf, dtype=float)
    upper[:facility_count] = 1.0
    integrality = np.zeros(variable_count, dtype=int)
    integrality[:facility_count] = 1
    options: dict[str, float | bool] = {"presolve": True, "mip_rel_gap": mip_rel_gap}
    if time_limit is not None:
        options["time_limit"] = time_limit

    started = time.perf_counter()
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(
            np.asarray(rows, dtype=float),
            lb=np.asarray(lower_bounds, dtype=float),
            ub=np.asarray(upper_bounds, dtype=float),
        ),
        options=options,
    )
    runtime = time.perf_counter() - started
    if not result.success or result.x is None or result.fun is None:
        raise RuntimeError(f"extensive-form solve failed: {result.message}")
    node_count_raw = getattr(result, "mip_node_count", 0)
    node_count = int(node_count_raw) if node_count_raw is not None else 0
    y = tuple(round(value) for value in np.asarray(result.x[:facility_count], dtype=float))
    return OracleResult(
        method="extensive_form",
        objective=float(result.fun),
        y=y,
        runtime_seconds=runtime,
        mip_node_count=node_count,
        message=str(result.message),
    )


def solve_by_enumeration(
    instance: StochasticFacilityLocationInstance,
    *,
    max_facilities: int = 16,
) -> OracleResult:
    """Enumerate all binary first-stage vectors and solve exact recourse LPs."""

    if instance.facility_count > max_facilities:
        raise ValueError(
            f"enumeration is capped at {max_facilities} facilities; "
            f"received {instance.facility_count}"
        )
    started = time.perf_counter()
    best_objective = float("inf")
    best_y: tuple[int, ...] | None = None
    for values in itertools.product((0, 1), repeat=instance.facility_count):
        y = tuple(int(value) for value in values)
        expected_recourse, _ = evaluate_recourse(instance, y)
        objective = first_stage_cost(instance, y) + expected_recourse
        if objective < best_objective - 1e-9:
            best_objective = objective
            best_y = y
    if best_y is None:  # pragma: no cover - guarded by positive facility count
        raise RuntimeError("enumeration produced no first-stage vector")
    return OracleResult(
        method="enumeration",
        objective=float(best_objective),
        y=best_y,
        runtime_seconds=time.perf_counter() - started,
        mip_node_count=0,
        message="complete first-stage enumeration",
    )
