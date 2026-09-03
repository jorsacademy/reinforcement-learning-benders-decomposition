"""Cut-batch actions, observations, state encoding, and transparent baselines."""

from __future__ import annotations

import enum
import math
import random
from dataclasses import dataclass, field
from typing import Protocol, TypeAlias

from rl_benders.cuts import BendersCut
from rl_benders.master import MasterResult


class CutBatchAction(str, enum.Enum):
    """How many currently violated scenario cuts to add."""

    ONE = "one"
    QUARTER = "quarter"
    HALF = "half"
    ALL = "all"


ACTION_ORDER: tuple[CutBatchAction, ...] = (
    CutBatchAction.ONE,
    CutBatchAction.QUARTER,
    CutBatchAction.HALF,
    CutBatchAction.ALL,
)

StateKey: TypeAlias = tuple[int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class CutCandidate:
    scenario_index: int
    cut: BendersCut
    recourse_objective: float
    theta_value: float
    violation: float
    normalized_violation: float


@dataclass(frozen=True, slots=True)
class BendersObservation:
    iteration: int
    master: MasterResult
    lower_bound: float
    incumbent_upper_bound: float
    incumbent_y: tuple[int, ...]
    current_objective: float
    current_expected_recourse: float
    relative_gap: float
    cut_count: int
    candidates: tuple[CutCandidate, ...]
    stall_count: int
    subproblem_seconds: float

    @property
    def violated_count(self) -> int:
        return len(self.candidates)


class CutBatchPolicy(Protocol):
    name: str

    def select(self, observation: BendersObservation, state: StateKey) -> CutBatchAction:
        """Choose a cut-batch action for the current exact observation."""


@dataclass(slots=True)
class FixedActionPolicy:
    action: CutBatchAction
    name: str = "fixed"

    def __post_init__(self) -> None:
        self.name = f"fixed_{self.action.value}"

    def select(self, observation: BendersObservation, state: StateKey) -> CutBatchAction:
        del observation, state
        return self.action


@dataclass(slots=True)
class AdaptiveHeuristicPolicy:
    """Transparent gap/violation heuristic used as a non-learning baseline."""

    name: str = "adaptive_heuristic"

    def select(self, observation: BendersObservation, state: StateKey) -> CutBatchAction:
        del state
        scenario_count = max(1, len(observation.master.theta))
        fraction = observation.violated_count / scenario_count
        if observation.relative_gap <= 0.03 or observation.stall_count >= 1:
            return CutBatchAction.ALL
        if fraction >= 0.75:
            return CutBatchAction.HALF
        if fraction >= 0.35:
            return CutBatchAction.QUARTER
        return CutBatchAction.ONE


@dataclass(slots=True)
class RandomPolicy:
    seed: int = 0
    name: str = "random_action"
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def select(self, observation: BendersObservation, state: StateKey) -> CutBatchAction:
        del observation, state
        return self._rng.choice(ACTION_ORDER)


def action_batch_size(action: CutBatchAction, candidate_count: int) -> int:
    if candidate_count <= 0:
        return 0
    if action is CutBatchAction.ONE:
        return 1
    if action is CutBatchAction.QUARTER:
        return max(1, math.ceil(0.25 * candidate_count))
    if action is CutBatchAction.HALF:
        return max(1, math.ceil(0.50 * candidate_count))
    return candidate_count


def encode_state(
    observation: BendersObservation,
    *,
    scenario_count: int,
) -> StateKey:
    """Discretize solver progress into a compact, auditable tabular-RL state."""

    gap = observation.relative_gap
    if gap <= 0.01:
        gap_bucket = 0
    elif gap <= 0.05:
        gap_bucket = 1
    elif gap <= 0.20:
        gap_bucket = 2
    else:
        gap_bucket = 3

    violated_fraction = observation.violated_count / max(1, scenario_count)
    if observation.violated_count == 0:
        violation_bucket = 0
    elif violated_fraction <= 0.25:
        violation_bucket = 1
    elif violated_fraction <= 0.75:
        violation_bucket = 2
    else:
        violation_bucket = 3

    total_violation = sum(max(0.0, candidate.violation) for candidate in observation.candidates)
    maximum_violation = max(
        (max(0.0, candidate.violation) for candidate in observation.candidates),
        default=0.0,
    )
    concentration = maximum_violation / total_violation if total_violation > 0 else 0.0
    if concentration < 0.40:
        concentration_bucket = 0
    elif concentration < 0.75:
        concentration_bucket = 1
    else:
        concentration_bucket = 2

    cuts_per_scenario = observation.cut_count / max(1, scenario_count)
    if cuts_per_scenario < 1.0:
        cut_load_bucket = 0
    elif cuts_per_scenario < 3.0:
        cut_load_bucket = 1
    else:
        cut_load_bucket = 2

    stall_bucket = min(observation.stall_count, 2)
    return (
        gap_bucket,
        violation_bucket,
        concentration_bucket,
        cut_load_bucket,
        stall_bucket,
    )
