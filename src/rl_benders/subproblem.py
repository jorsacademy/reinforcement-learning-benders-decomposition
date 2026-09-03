"""Exact recourse LPs and dual-derived Benders optimality cuts."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from rl_benders.cuts import BendersCut
from rl_benders.domain import DemandScenario, StochasticFacilityLocationInstance


@dataclass(frozen=True, slots=True)
class RecourseResult:
    scenario_index: int
    success: bool
    objective: float
    shipments: tuple[tuple[float, ...], ...]
    unmet_demand: tuple[float, ...]
    demand_duals: tuple[float, ...]
    capacity_duals: tuple[float, ...]
    cut: BendersCut
    primal_dual_residual: float
    runtime_seconds: float
    message: str


def _validate_open_vector(
    instance: StochasticFacilityLocationInstance,
    y: tuple[int, ...] | tuple[float, ...],
) -> np.ndarray:
    values = np.asarray(y, dtype=float)
    if values.shape != (instance.facility_count,):
        raise ValueError(
            f"expected {instance.facility_count} first-stage values, received {values.shape}"
        )
    if not np.all(np.isfinite(values)) or np.any(values < -1e-9) or np.any(values > 1.0 + 1e-9):
        raise ValueError("first-stage values must be finite and lie in [0, 1]")
    return values


def _solve_primal(
    instance: StochasticFacilityLocationInstance,
    scenario: DemandScenario,
    y: np.ndarray,
) -> tuple[float, np.ndarray, str]:
    customer_count = instance.customer_count
    facility_count = instance.facility_count
    shipment_count = customer_count * facility_count
    objective = np.concatenate(
        [
            np.asarray(instance.shipping_costs, dtype=float).reshape(-1),
            np.asarray(instance.shortage_penalties, dtype=float),
        ]
    )

    rows: list[np.ndarray] = []
    upper_bounds: list[float] = []
    for customer in range(customer_count):
        row = np.zeros(shipment_count + customer_count, dtype=float)
        start = customer * facility_count
        row[start : start + facility_count] = -1.0
        row[shipment_count + customer] = -1.0
        rows.append(row)
        upper_bounds.append(-scenario.demand[customer])

    for facility in range(facility_count):
        row = np.zeros(shipment_count + customer_count, dtype=float)
        for customer in range(customer_count):
            row[customer * facility_count + facility] = 1.0
        rows.append(row)
        upper_bounds.append(instance.facilities[facility].capacity * y[facility])

    result = linprog(
        objective,
        A_ub=np.asarray(rows, dtype=float),
        b_ub=np.asarray(upper_bounds, dtype=float),
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success or result.x is None or result.fun is None:
        raise RuntimeError(f"recourse primal failed: {result.message}")
    return float(result.fun), np.asarray(result.x, dtype=float), str(result.message)


def _solve_dual(
    instance: StochasticFacilityLocationInstance,
    scenario: DemandScenario,
    y: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, str]:
    customer_count = instance.customer_count
    facility_count = instance.facility_count
    demand = np.asarray(scenario.demand, dtype=float)
    capacities = np.asarray(instance.capacities, dtype=float)
    penalties = np.asarray(instance.shortage_penalties, dtype=float)
    shipping = np.asarray(instance.shipping_costs, dtype=float)

    objective = np.concatenate([-demand, capacities * y])
    rows: list[np.ndarray] = []
    upper_bounds: list[float] = []

    for customer in range(customer_count):
        for facility in range(facility_count):
            row = np.zeros(customer_count + facility_count, dtype=float)
            row[customer] = 1.0
            row[customer_count + facility] = -1.0
            rows.append(row)
            upper_bounds.append(shipping[customer, facility])

    for customer in range(customer_count):
        row = np.zeros(customer_count + facility_count, dtype=float)
        row[customer] = 1.0
        rows.append(row)
        upper_bounds.append(penalties[customer])

    lambda_upper = np.maximum(
        0.0,
        np.max(penalties[:, None] - shipping, axis=0),
    )
    bounds = [(0.0, None)] * customer_count + [(0.0, float(limit)) for limit in lambda_upper]
    result = linprog(
        objective,
        A_ub=np.asarray(rows, dtype=float),
        b_ub=np.asarray(upper_bounds, dtype=float),
        bounds=bounds,
        method="highs",
    )
    if not result.success or result.x is None or result.fun is None:
        raise RuntimeError(f"recourse dual failed: {result.message}")
    values = np.asarray(result.x, dtype=float)
    pi = values[:customer_count]
    lambdas = values[customer_count:]
    dual_value = float(demand @ pi - (capacities * y) @ lambdas)
    return dual_value, pi, lambdas, str(result.message)


def solve_recourse(
    instance: StochasticFacilityLocationInstance,
    scenario_index: int,
    y: tuple[int, ...] | tuple[float, ...],
    *,
    duality_tolerance: float = 1e-7,
) -> RecourseResult:
    """Solve one scenario exactly and construct a globally valid Benders cut."""

    if scenario_index < 0 or scenario_index >= instance.scenario_count:
        raise IndexError("scenario_index out of range")
    y_values = _validate_open_vector(instance, y)
    scenario = instance.scenarios[scenario_index]
    started = time.perf_counter()
    primal_value, primal_solution, primal_message = _solve_primal(instance, scenario, y_values)
    dual_value, pi, lambdas, dual_message = _solve_dual(instance, scenario, y_values)
    residual = abs(primal_value - dual_value)
    scale = max(1.0, abs(primal_value), abs(dual_value))
    if residual > duality_tolerance * scale:
        raise RuntimeError(
            "recourse primal/dual mismatch: "
            f"primal={primal_value:.12g}, dual={dual_value:.12g}, residual={residual:.3e}"
        )

    customer_count = instance.customer_count
    facility_count = instance.facility_count
    shipment_count = customer_count * facility_count
    shipments = primal_solution[:shipment_count].reshape(customer_count, facility_count)
    unmet = primal_solution[shipment_count:]
    capacities = np.asarray(instance.capacities, dtype=float)
    beta = tuple(float(-capacities[index] * lambdas[index]) for index in range(facility_count))
    alpha = float(np.asarray(scenario.demand, dtype=float) @ pi)
    source_y = tuple(round(value) for value in y_values)
    cut = BendersCut(
        scenario_index=scenario_index,
        alpha=alpha,
        beta=beta,
        source_y=source_y,
        source_recourse=primal_value,
    )
    return RecourseResult(
        scenario_index=scenario_index,
        success=True,
        objective=primal_value,
        shipments=tuple(tuple(float(value) for value in row) for row in shipments),
        unmet_demand=tuple(float(value) for value in unmet),
        demand_duals=tuple(float(value) for value in pi),
        capacity_duals=tuple(float(value) for value in lambdas),
        cut=cut,
        primal_dual_residual=residual,
        runtime_seconds=time.perf_counter() - started,
        message=f"primal: {primal_message}; dual: {dual_message}",
    )


def evaluate_recourse(
    instance: StochasticFacilityLocationInstance,
    y: tuple[int, ...] | tuple[float, ...],
) -> tuple[float, tuple[RecourseResult, ...]]:
    results = tuple(
        solve_recourse(instance, scenario_index, y)
        for scenario_index in range(instance.scenario_count)
    )
    expected = sum(
        scenario.probability * result.objective
        for scenario, result in zip(instance.scenarios, results, strict=True)
    )
    return float(expected), results
