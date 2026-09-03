from __future__ import annotations

import pytest

from rl_benders.domain import (
    Customer,
    DemandScenario,
    Facility,
    StochasticFacilityLocationInstance,
)


@pytest.fixture
def tiny_instance() -> StochasticFacilityLocationInstance:
    return StochasticFacilityLocationInstance(
        name="tiny-stochastic-facility-location",
        facilities=(
            Facility(1, 0.0, 0.0, 8.0, 70.0),
            Facility(2, 1.0, 0.0, 7.0, 65.0),
            Facility(3, 0.5, 1.0, 6.0, 55.0),
        ),
        customers=(
            Customer(1, 0.0, 0.0, 25.0),
            Customer(2, 1.0, 0.0, 24.0),
            Customer(3, 0.5, 1.0, 26.0),
        ),
        scenarios=(
            DemandScenario(1, 0.40, (4.0, 5.0, 3.0)),
            DemandScenario(2, 0.35, (6.0, 2.0, 4.0)),
            DemandScenario(3, 0.25, (2.0, 6.0, 5.0)),
        ),
        shipping_costs=(
            (4.0, 7.0, 9.0),
            (6.0, 3.0, 8.0),
            (8.0, 6.0, 2.0),
        ),
    )
