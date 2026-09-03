from __future__ import annotations

from pathlib import Path

import pytest

from rl_benders.domain import (
    Customer,
    DemandScenario,
    Facility,
    StochasticFacilityLocationInstance,
    load_instance,
    save_instance,
)
from rl_benders.generator import GeneratorConfig, generate_instance


def test_generator_is_deterministic_and_well_formed() -> None:
    config = GeneratorConfig(
        facility_count=4,
        customer_count=6,
        scenario_count=5,
        spatial_regime="clustered",
        demand_regime="volatile",
        seed=17,
    )
    first = generate_instance(config)
    second = generate_instance(config)
    assert first == second
    assert first.facility_count == 4
    assert first.customer_count == 6
    assert first.scenario_count == 5
    assert sum(scenario.probability for scenario in first.scenarios) == pytest.approx(1.0)
    assert all(
        customer.shortage_penalty > min(first.shipping_costs[index])
        for index, customer in enumerate(first.customers)
    )


def test_instance_json_round_trip(tmp_path: Path) -> None:
    instance = generate_instance(GeneratorConfig(seed=3))
    path = tmp_path / "instance.json"
    save_instance(instance, path)
    assert load_instance(path) == instance


def test_instance_rejects_invalid_probability_sum() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        StochasticFacilityLocationInstance(
            name="bad",
            facilities=(Facility(1, 0.0, 0.0, 5.0, 1.0),),
            customers=(Customer(1, 0.0, 0.0, 10.0),),
            scenarios=(DemandScenario(1, 0.8, (1.0,)),),
            shipping_costs=((1.0,),),
        )
