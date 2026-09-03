"""Auditable tabular Q-learning for safe Benders cut-batch control."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from rl_benders.benders import BendersConfig, BendersEnvironment, RewardConfig
from rl_benders.control import ACTION_ORDER, BendersObservation, CutBatchAction, StateKey
from rl_benders.domain import StochasticFacilityLocationInstance

Q_SHAPE = (4, 4, 3, 3, 3, len(ACTION_ORDER))
CHECKPOINT_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class QLearningConfig:
    episodes: int = 200
    learning_rate: float = 0.18
    discount_factor: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    seed: int = 0

    def __post_init__(self) -> None:
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must lie in (0, 1]")
        if not 0 <= self.discount_factor <= 1:
            raise ValueError("discount_factor must lie in [0, 1]")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")


@dataclass(frozen=True, slots=True)
class EpisodeSummary:
    episode: int
    instance_name: str
    epsilon: float
    cumulative_reward: float
    decisions: int
    certified_within_policy_horizon: bool
    truncated: bool
    final_relative_gap: float


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    episodes: tuple[EpisodeSummary, ...]

    @property
    def mean_reward(self) -> float:
        return float(np.mean([episode.cumulative_reward for episode in self.episodes]))

    @property
    def certification_rate(self) -> float:
        return float(
            np.mean([episode.certified_within_policy_horizon for episode in self.episodes])
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "mean_reward": self.mean_reward,
            "certification_rate": self.certification_rate,
            "episodes": [
                {
                    "episode": item.episode,
                    "instance_name": item.instance_name,
                    "epsilon": item.epsilon,
                    "cumulative_reward": item.cumulative_reward,
                    "decisions": item.decisions,
                    "certified_within_policy_horizon": item.certified_within_policy_horizon,
                    "truncated": item.truncated,
                    "final_relative_gap": item.final_relative_gap,
                }
                for item in self.episodes
            ],
        }


class TabularQPolicy:
    """Epsilon-greedy Q-learning over a compact solver-progress state space."""

    name = "q_learning"

    def __init__(
        self,
        *,
        seed: int = 0,
        q_values: np.ndarray | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        if q_values is None:
            self.q_values = np.zeros(Q_SHAPE, dtype=float)
        else:
            values = np.asarray(q_values, dtype=float)
            if values.shape != Q_SHAPE or not np.all(np.isfinite(values)):
                raise ValueError(f"q_values must be finite with shape {Q_SHAPE}")
            self.q_values = values.copy()
        self.epsilon = 0.0
        self._rng = np.random.default_rng(seed)
        self.metadata = dict(metadata or {})

    @staticmethod
    def _state_index(state: StateKey) -> tuple[int, int, int, int, int]:
        limits = (4, 4, 3, 3, 3)
        if len(state) != len(limits):
            raise ValueError("state has the wrong dimension")
        if any(value < 0 or value >= limit for value, limit in zip(state, limits, strict=True)):
            raise ValueError(f"state {state} is outside the declared tabular space")
        return state

    def select(self, observation: BendersObservation, state: StateKey) -> CutBatchAction:
        del observation
        index = self._state_index(state)
        if self._rng.random() < self.epsilon:
            return ACTION_ORDER[int(self._rng.integers(0, len(ACTION_ORDER)))]
        action_values = self.q_values[index]
        maximum = float(np.max(action_values))
        ties = np.flatnonzero(np.isclose(action_values, maximum, atol=1e-12, rtol=0.0))
        # Prefer the larger cut batch under exact ties. This makes an untrained
        # checkpoint degrade to the conservative all-cut policy.
        action_index = int(ties[-1])
        return ACTION_ORDER[action_index]

    def update(
        self,
        state: StateKey,
        action: CutBatchAction,
        reward: float,
        next_state: StateKey,
        *,
        terminal: bool,
        learning_rate: float,
        discount_factor: float,
    ) -> None:
        state_index = self._state_index(state)
        next_index = self._state_index(next_state)
        action_index = ACTION_ORDER.index(action)
        current = float(self.q_values[(*state_index, action_index)])
        bootstrap = 0.0 if terminal else float(np.max(self.q_values[next_index]))
        target = reward + discount_factor * bootstrap
        self.q_values[(*state_index, action_index)] = current + learning_rate * (target - current)

    def save(self, path: str | Path, *, metadata: dict[str, object] | None = None) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        combined = dict(self.metadata)
        if metadata:
            combined.update(metadata)
        combined.update(
            {
                "checkpoint_version": CHECKPOINT_VERSION,
                "actions": [action.value for action in ACTION_ORDER],
                "q_shape": list(Q_SHAPE),
            }
        )
        np.savez_compressed(
            output,
            q_values=self.q_values,
            metadata_json=np.asarray([json.dumps(combined, sort_keys=True)]),
        )

    @classmethod
    def load(cls, path: str | Path, *, seed: int = 0) -> TabularQPolicy:
        with np.load(Path(path), allow_pickle=False) as payload:
            q_values = np.asarray(payload["q_values"], dtype=float)
            metadata = cast(
                dict[str, object],
                json.loads(str(payload["metadata_json"][0])),
            )
        if metadata.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise ValueError("unsupported Q-policy checkpoint version")
        if metadata.get("actions") != [action.value for action in ACTION_ORDER]:
            raise ValueError("checkpoint action order does not match this package")
        return cls(seed=seed, q_values=q_values, metadata=metadata)


def epsilon_for_episode(config: QLearningConfig, episode: int) -> float:
    if config.episodes == 1:
        return config.epsilon_end
    fraction = episode / (config.episodes - 1)
    return float(config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start))


def train_q_policy(
    instances: list[StochasticFacilityLocationInstance]
    | tuple[StochasticFacilityLocationInstance, ...],
    *,
    q_config: QLearningConfig | None = None,
    benders_config: BendersConfig | None = None,
    reward_config: RewardConfig | None = None,
) -> tuple[TabularQPolicy, TrainingSummary]:
    """Train on complete Benders episodes with disjoint instances supplied by the caller."""

    if not instances:
        raise ValueError("training instances must be nonempty")
    q_config = q_config or QLearningConfig()
    benders_config = benders_config or BendersConfig()
    reward_config = reward_config or RewardConfig()
    policy = TabularQPolicy(seed=q_config.seed)
    rng = np.random.default_rng(q_config.seed)
    summaries: list[EpisodeSummary] = []

    for episode in range(q_config.episodes):
        instance_index = int(rng.integers(0, len(instances)))
        instance = instances[instance_index]
        policy.epsilon = epsilon_for_episode(q_config, episode)
        environment = BendersEnvironment(
            instance,
            config=benders_config,
            reward_config=reward_config,
        )
        environment.reset()
        total_reward = 0.0
        truncated = False
        while not environment.terminated:
            state = environment.state()
            action = policy.select(environment.observation, state)
            transition = environment.step(action)
            total_reward += transition.reward
            terminal = transition.terminated or transition.truncated
            policy.update(
                transition.state,
                transition.action,
                transition.reward,
                transition.next_state,
                terminal=terminal,
                learning_rate=q_config.learning_rate,
                discount_factor=q_config.discount_factor,
            )
            if transition.truncated:
                truncated = True
                break
        summaries.append(
            EpisodeSummary(
                episode=episode,
                instance_name=instance.name,
                epsilon=policy.epsilon,
                cumulative_reward=total_reward,
                decisions=environment.result(policy.name).policy_decisions,
                certified_within_policy_horizon=environment.terminated,
                truncated=truncated,
                final_relative_gap=environment.observation.relative_gap,
            )
        )

    policy.epsilon = 0.0
    policy.metadata.update(
        {
            "training_episodes": q_config.episodes,
            "training_instance_names": [instance.name for instance in instances],
            "q_learning_config": {
                "learning_rate": q_config.learning_rate,
                "discount_factor": q_config.discount_factor,
                "epsilon_start": q_config.epsilon_start,
                "epsilon_end": q_config.epsilon_end,
                "seed": q_config.seed,
            },
        }
    )
    return policy, TrainingSummary(episodes=tuple(summaries))
