from __future__ import annotations

import itertools

import pytest

from rl_benders.domain import StochasticFacilityLocationInstance
from rl_benders.subproblem import evaluate_recourse, solve_recourse


def test_recourse_primal_dual_strong_duality(
    tiny_instance: StochasticFacilityLocationInstance,
) -> None:
    for scenario_index in range(tiny_instance.scenario_count):
        for y in itertools.product((0, 1), repeat=tiny_instance.facility_count):
            result = solve_recourse(tiny_instance, scenario_index, y)
            assert result.success
            assert result.primal_dual_residual <= 1e-8
            assert result.cut.rhs(y) == pytest.approx(result.objective, abs=1e-8)
            assert all(value >= -1e-9 for value in result.unmet_demand)


def test_every_generated_cut_is_globally_valid_on_binary_domain(
    tiny_instance: StochasticFacilityLocationInstance,
) -> None:
    source_y = (1, 0, 1)
    for scenario_index in range(tiny_instance.scenario_count):
        cut = solve_recourse(tiny_instance, scenario_index, source_y).cut
        for candidate_y in itertools.product((0, 1), repeat=tiny_instance.facility_count):
            candidate = solve_recourse(tiny_instance, scenario_index, candidate_y)
            assert cut.rhs(candidate_y) <= candidate.objective + 1e-7


def test_expected_recourse_uses_scenario_probabilities(
    tiny_instance: StochasticFacilityLocationInstance,
) -> None:
    expected, results = evaluate_recourse(tiny_instance, (0, 1, 1))
    manual = sum(
        scenario.probability * result.objective
        for scenario, result in zip(tiny_instance.scenarios, results, strict=True)
    )
    assert expected == pytest.approx(manual)
