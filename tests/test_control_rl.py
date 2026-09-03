from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rl_benders.benders import BendersConfig, BendersEnvironment, solve_with_policy
from rl_benders.control import ACTION_ORDER, CutBatchAction, encode_state
from rl_benders.domain import StochasticFacilityLocationInstance
from rl_benders.rl import QLearningConfig, TabularQPolicy, train_q_policy


def test_state_encoding_stays_inside_declared_table(
    tiny_instance: StochasticFacilityLocationInstance,
) -> None:
    environment = BendersEnvironment(tiny_instance)
    observation = environment.reset()
    state = encode_state(observation, scenario_count=tiny_instance.scenario_count)
    assert len(state) == 5
    assert 0 <= state[0] < 4
    assert 0 <= state[1] < 4
    assert all(0 <= value < 3 for value in state[2:])


def test_untrained_q_policy_degrades_to_all_cuts(
    tiny_instance: StochasticFacilityLocationInstance,
) -> None:
    environment = BendersEnvironment(tiny_instance)
    observation = environment.reset()
    policy = TabularQPolicy(seed=0)
    policy.epsilon = 0.0
    assert policy.select(observation, environment.state()) is CutBatchAction.ALL


def test_q_update_moves_selected_action_toward_positive_target() -> None:
    policy = TabularQPolicy(seed=0)
    state = (3, 3, 0, 0, 0)
    action = CutBatchAction.HALF
    policy.update(
        state,
        action,
        reward=2.0,
        next_state=(2, 2, 0, 1, 0),
        terminal=True,
        learning_rate=0.5,
        discount_factor=0.9,
    )
    action_index = ACTION_ORDER.index(action)
    assert policy.q_values[(*state, action_index)] == pytest.approx(1.0)


def test_q_checkpoint_round_trip(tmp_path: Path) -> None:
    policy = TabularQPolicy(seed=0)
    policy.q_values[3, 3, 2, 2, 2, 1] = 4.25
    path = tmp_path / "policy.npz"
    policy.save(path, metadata={"test": True})
    loaded = TabularQPolicy.load(path)
    assert np.array_equal(loaded.q_values, policy.q_values)
    assert loaded.metadata["test"] is True


def test_small_q_training_and_certified_evaluation(
    tiny_instance: StochasticFacilityLocationInstance,
) -> None:
    policy, summary = train_q_policy(
        [tiny_instance],
        q_config=QLearningConfig(episodes=4, epsilon_start=0.5, epsilon_end=0.0, seed=7),
        benders_config=BendersConfig(max_policy_decisions=20),
    )
    assert len(summary.episodes) == 4
    assert np.all(np.isfinite(policy.q_values))
    result = solve_with_policy(
        tiny_instance,
        policy,
        config=BendersConfig(max_policy_decisions=20),
    )
    assert result.globally_certified
    assert result.objective == pytest.approx(172.5, abs=1e-7)
