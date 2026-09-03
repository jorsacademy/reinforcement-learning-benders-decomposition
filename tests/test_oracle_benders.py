from __future__ import annotations

import pytest

from rl_benders.benders import BendersConfig, solve_classical_benders, solve_with_policy
from rl_benders.control import CutBatchAction, FixedActionPolicy
from rl_benders.domain import StochasticFacilityLocationInstance
from rl_benders.master import solve_master
from rl_benders.oracle import solve_by_enumeration, solve_extensive_form


def test_extensive_form_matches_complete_first_stage_enumeration(
    tiny_instance: StochasticFacilityLocationInstance,
) -> None:
    extensive = solve_extensive_form(tiny_instance)
    enumeration = solve_by_enumeration(tiny_instance)
    assert extensive.objective == pytest.approx(172.5)
    assert extensive.objective == pytest.approx(enumeration.objective, abs=1e-8)
    assert extensive.y == enumeration.y == (0, 1, 1)


def test_classical_multicut_benders_matches_oracle(
    tiny_instance: StochasticFacilityLocationInstance,
) -> None:
    result = solve_classical_benders(tiny_instance)
    oracle = solve_extensive_form(tiny_instance)
    assert result.globally_certified
    assert result.objective == pytest.approx(oracle.objective, abs=1e-7)
    assert result.lower_bound == pytest.approx(result.upper_bound, abs=1e-7)
    assert result.y == oracle.y
    assert result.cut_count > 0


def test_empty_master_is_a_valid_lower_bound(
    tiny_instance: StochasticFacilityLocationInstance,
) -> None:
    result = solve_master(tiny_instance, [])
    assert result.objective == pytest.approx(0.0)
    assert result.y == (0, 0, 0)
    assert result.theta == pytest.approx((0.0, 0.0, 0.0))


def test_short_policy_horizon_uses_exact_completion_without_losing_optimality(
    tiny_instance: StochasticFacilityLocationInstance,
) -> None:
    result = solve_with_policy(
        tiny_instance,
        FixedActionPolicy(CutBatchAction.ONE),
        config=BendersConfig(max_policy_decisions=1, max_completion_iterations=30),
        safe_complete=True,
    )
    assert result.globally_certified
    assert result.completion_phase_used
    assert result.policy_decisions == 1
    assert result.objective == pytest.approx(172.5, abs=1e-7)


def test_without_safe_completion_a_truncated_run_is_not_certified(
    tiny_instance: StochasticFacilityLocationInstance,
) -> None:
    result = solve_with_policy(
        tiny_instance,
        FixedActionPolicy(CutBatchAction.ONE),
        config=BendersConfig(max_policy_decisions=1),
        safe_complete=False,
    )
    assert not result.globally_certified
    assert result.lower_bound < result.upper_bound
