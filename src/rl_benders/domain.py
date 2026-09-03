"""Typed data model for two-stage stochastic capacitated facility location."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

SpatialRegime: TypeAlias = Literal["uniform", "clustered"]
DemandRegime: TypeAlias = Literal["stable", "volatile"]
Number: TypeAlias = int | float


def _as_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _as_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


@dataclass(frozen=True, slots=True)
class Facility:
    """A candidate first-stage facility."""

    id: int
    x: float
    y: float
    capacity: float
    opening_cost: float

    def __post_init__(self) -> None:
        if self.id <= 0:
            raise ValueError("facility id must be positive")
        values = (self.x, self.y, self.capacity, self.opening_cost)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("facility values must be finite")
        if self.capacity <= 0:
            raise ValueError("facility capacity must be positive")
        if self.opening_cost < 0:
            raise ValueError("facility opening_cost must be nonnegative")


@dataclass(frozen=True, slots=True)
class Customer:
    """A second-stage demand location."""

    id: int
    x: float
    y: float
    shortage_penalty: float

    def __post_init__(self) -> None:
        if self.id <= 0:
            raise ValueError("customer id must be positive")
        values = (self.x, self.y, self.shortage_penalty)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("customer values must be finite")
        if self.shortage_penalty <= 0:
            raise ValueError("shortage_penalty must be positive")


@dataclass(frozen=True, slots=True)
class DemandScenario:
    """A finite demand scenario with a probability weight."""

    id: int
    probability: float
    demand: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.id <= 0:
            raise ValueError("scenario id must be positive")
        if not math.isfinite(self.probability) or self.probability <= 0:
            raise ValueError("scenario probability must be positive and finite")
        if not self.demand:
            raise ValueError("scenario demand must be nonempty")
        if not all(math.isfinite(value) and value >= 0 for value in self.demand):
            raise ValueError("scenario demands must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class StochasticFacilityLocationInstance:
    """Finite two-stage facility-location instance with complete recourse.

    First-stage variables decide which facilities to open. In each scenario,
    continuous transportation and unmet-demand variables form the recourse LP.
    """

    name: str
    facilities: tuple[Facility, ...]
    customers: tuple[Customer, ...]
    scenarios: tuple[DemandScenario, ...]
    shipping_costs: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("instance name must be nonempty")
        if not self.facilities or not self.customers or not self.scenarios:
            raise ValueError("facilities, customers, and scenarios must be nonempty")
        facility_ids = tuple(facility.id for facility in self.facilities)
        if facility_ids != tuple(range(1, len(self.facilities) + 1)):
            raise ValueError("facility ids must be contiguous and ordered from 1")
        customer_ids = tuple(customer.id for customer in self.customers)
        if customer_ids != tuple(range(1, len(self.customers) + 1)):
            raise ValueError("customer ids must be contiguous and ordered from 1")
        scenario_ids = tuple(scenario.id for scenario in self.scenarios)
        if scenario_ids != tuple(range(1, len(self.scenarios) + 1)):
            raise ValueError("scenario ids must be contiguous and ordered from 1")
        if any(len(scenario.demand) != self.customer_count for scenario in self.scenarios):
            raise ValueError("every scenario demand vector must match the customer count")
        probability_sum = sum(scenario.probability for scenario in self.scenarios)
        if not math.isclose(probability_sum, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("scenario probabilities must sum to one")
        if len(self.shipping_costs) != self.customer_count:
            raise ValueError("shipping_costs must have one row per customer")
        for row in self.shipping_costs:
            if len(row) != self.facility_count:
                raise ValueError("shipping_costs must have one column per facility")
            if not all(math.isfinite(value) and value >= 0 for value in row):
                raise ValueError("shipping costs must be finite and nonnegative")
        for customer_index, customer in enumerate(self.customers):
            cheapest = min(self.shipping_costs[customer_index])
            if customer.shortage_penalty <= cheapest:
                raise ValueError(
                    "each shortage penalty must exceed the customer's cheapest shipping cost"
                )

    @property
    def facility_count(self) -> int:
        return len(self.facilities)

    @property
    def customer_count(self) -> int:
        return len(self.customers)

    @property
    def scenario_count(self) -> int:
        return len(self.scenarios)

    @property
    def opening_costs(self) -> tuple[float, ...]:
        return tuple(facility.opening_cost for facility in self.facilities)

    @property
    def capacities(self) -> tuple[float, ...]:
        return tuple(facility.capacity for facility in self.facilities)

    @property
    def shortage_penalties(self) -> tuple[float, ...]:
        return tuple(customer.shortage_penalty for customer in self.customers)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "facilities": [asdict(facility) for facility in self.facilities],
            "customers": [asdict(customer) for customer in self.customers],
            "scenarios": [asdict(scenario) for scenario in self.scenarios],
            "shipping_costs": [list(row) for row in self.shipping_costs],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> StochasticFacilityLocationInstance:
        raw_facilities = payload.get("facilities")
        raw_customers = payload.get("customers")
        raw_scenarios = payload.get("scenarios")
        raw_costs = payload.get("shipping_costs")
        if not isinstance(raw_facilities, list):
            raise ValueError("facilities must be a list")
        if not isinstance(raw_customers, list):
            raise ValueError("customers must be a list")
        if not isinstance(raw_scenarios, list):
            raise ValueError("scenarios must be a list")
        if not isinstance(raw_costs, list):
            raise ValueError("shipping_costs must be a list")

        facilities: list[Facility] = []
        for raw in raw_facilities:
            if not isinstance(raw, dict):
                raise ValueError("every facility must be an object")
            facilities.append(
                Facility(
                    id=_as_int(raw.get("id"), "facility.id"),
                    x=_as_float(raw.get("x"), "facility.x"),
                    y=_as_float(raw.get("y"), "facility.y"),
                    capacity=_as_float(raw.get("capacity"), "facility.capacity"),
                    opening_cost=_as_float(raw.get("opening_cost"), "facility.opening_cost"),
                )
            )

        customers: list[Customer] = []
        for raw in raw_customers:
            if not isinstance(raw, dict):
                raise ValueError("every customer must be an object")
            customers.append(
                Customer(
                    id=_as_int(raw.get("id"), "customer.id"),
                    x=_as_float(raw.get("x"), "customer.x"),
                    y=_as_float(raw.get("y"), "customer.y"),
                    shortage_penalty=_as_float(
                        raw.get("shortage_penalty"), "customer.shortage_penalty"
                    ),
                )
            )

        scenarios: list[DemandScenario] = []
        for raw in raw_scenarios:
            if not isinstance(raw, dict):
                raise ValueError("every scenario must be an object")
            raw_demand = raw.get("demand")
            if not isinstance(raw_demand, list):
                raise ValueError("scenario.demand must be a list")
            scenarios.append(
                DemandScenario(
                    id=_as_int(raw.get("id"), "scenario.id"),
                    probability=_as_float(raw.get("probability"), "scenario.probability"),
                    demand=tuple(_as_float(value, "scenario.demand") for value in raw_demand),
                )
            )

        costs: list[tuple[float, ...]] = []
        for row in raw_costs:
            if not isinstance(row, list):
                raise ValueError("each shipping_costs row must be a list")
            costs.append(tuple(_as_float(value, "shipping_cost") for value in row))

        name_value = payload.get("name")
        if not isinstance(name_value, str):
            raise ValueError("name must be a string")
        return cls(
            name=name_value,
            facilities=tuple(facilities),
            customers=tuple(customers),
            scenarios=tuple(scenarios),
            shipping_costs=tuple(costs),
        )


def save_instance(instance: StochasticFacilityLocationInstance, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(instance.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_instance(path: str | Path) -> StochasticFacilityLocationInstance:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("instance JSON must be an object")
    return StochasticFacilityLocationInstance.from_dict(cast(dict[str, object], raw))
