"""Reproducible synthetic stochastic facility-location instances."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl_benders.domain import (
    Customer,
    DemandRegime,
    DemandScenario,
    Facility,
    SpatialRegime,
    StochasticFacilityLocationInstance,
)


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    facility_count: int = 5
    customer_count: int = 8
    scenario_count: int = 8
    spatial_regime: SpatialRegime = "uniform"
    demand_regime: DemandRegime = "stable"
    capacity_factor: float = 1.22
    shipping_scale: float = 14.0
    shipping_base: float = 1.5
    shortage_multiplier: float = 2.8
    opening_cost_per_capacity: float = 15.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.facility_count <= 0 or self.customer_count <= 0 or self.scenario_count <= 0:
            raise ValueError("facility_count, customer_count, and scenario_count must be positive")
        if self.capacity_factor <= 0:
            raise ValueError("capacity_factor must be positive")
        if self.shipping_scale <= 0 or self.shipping_base < 0:
            raise ValueError("shipping cost parameters are invalid")
        if self.shortage_multiplier <= 1:
            raise ValueError("shortage_multiplier must exceed one")
        if self.opening_cost_per_capacity <= 0:
            raise ValueError("opening_cost_per_capacity must be positive")


def _coordinates(
    rng: np.random.Generator,
    count: int,
    regime: SpatialRegime,
) -> np.ndarray:
    if regime == "uniform":
        return rng.uniform(0.0, 1.0, size=(count, 2))
    center_count = max(2, min(4, count // 3 + 1))
    centers = rng.uniform(0.15, 0.85, size=(center_count, 2))
    assignments = rng.integers(0, center_count, size=count)
    coordinates = centers[assignments] + rng.normal(0.0, 0.075, size=(count, 2))
    return np.clip(coordinates, 0.0, 1.0)


def generate_instance(
    config: GeneratorConfig | None = None,
    **overrides: object,
) -> StochasticFacilityLocationInstance:
    """Generate a finite instance with complete recourse.

    Keyword overrides are accepted for convenient experiment construction and are
    validated by ``GeneratorConfig`` before any random numbers are drawn.
    """

    if config is not None and overrides:
        raise ValueError("provide either config or keyword overrides, not both")
    if config is None:
        config = GeneratorConfig(**overrides)  # type: ignore[arg-type]
    rng = np.random.default_rng(config.seed)

    facility_xy = _coordinates(rng, config.facility_count, config.spatial_regime)
    customer_xy = _coordinates(rng, config.customer_count, config.spatial_regime)

    delta = customer_xy[:, None, :] - facility_xy[None, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=2))
    shipping_noise = rng.uniform(0.92, 1.08, size=distances.shape)
    shipping = config.shipping_base + config.shipping_scale * distances * shipping_noise

    base_demand = rng.uniform(4.0, 10.0, size=config.customer_count)
    sigma = 0.14 if config.demand_regime == "stable" else 0.38
    raw_probabilities = rng.uniform(0.7, 1.3, size=config.scenario_count)
    probabilities = raw_probabilities / np.sum(raw_probabilities)
    probabilities[-1] = 1.0 - float(np.sum(probabilities[:-1]))

    scenario_demands: list[np.ndarray] = []
    for _ in range(config.scenario_count):
        common = rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma)
        idiosyncratic = rng.lognormal(
            mean=-0.5 * (0.55 * sigma) ** 2,
            sigma=0.55 * sigma,
            size=config.customer_count,
        )
        demand = np.maximum(0.25, base_demand * common * idiosyncratic)
        scenario_demands.append(demand)

    expected_demand = np.sum(
        np.asarray(scenario_demands) * probabilities[:, None],
        axis=0,
    )
    total_target_capacity = config.capacity_factor * float(np.sum(expected_demand))
    raw_capacity = rng.uniform(0.75, 1.25, size=config.facility_count)
    capacities = total_target_capacity * raw_capacity / float(np.sum(raw_capacity))

    facilities = tuple(
        Facility(
            id=index + 1,
            x=float(facility_xy[index, 0]),
            y=float(facility_xy[index, 1]),
            capacity=float(capacities[index]),
            opening_cost=float(
                capacities[index]
                * config.opening_cost_per_capacity
                * rng.uniform(0.85, 1.15)
            ),
        )
        for index in range(config.facility_count)
    )

    customers: list[Customer] = []
    for index in range(config.customer_count):
        maximum_shipping = float(np.max(shipping[index]))
        penalty = config.shortage_multiplier * maximum_shipping + rng.uniform(4.0, 10.0)
        customers.append(
            Customer(
                id=index + 1,
                x=float(customer_xy[index, 0]),
                y=float(customer_xy[index, 1]),
                shortage_penalty=float(penalty),
            )
        )

    scenarios = tuple(
        DemandScenario(
            id=index + 1,
            probability=float(probabilities[index]),
            demand=tuple(float(value) for value in scenario_demands[index]),
        )
        for index in range(config.scenario_count)
    )

    name = (
        f"sfl-{config.spatial_regime}-{config.demand_regime}"
        f"-f{config.facility_count}-c{config.customer_count}"
        f"-s{config.scenario_count}-seed{config.seed}"
    )
    return StochasticFacilityLocationInstance(
        name=name,
        facilities=facilities,
        customers=tuple(customers),
        scenarios=scenarios,
        shipping_costs=tuple(tuple(float(value) for value in row) for row in shipping),
    )
