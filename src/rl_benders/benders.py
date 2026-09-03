"""Certified Benders decomposition with policy-controlled cut batching."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rl_benders.control import (
    BendersObservation,
    CutBatchAction,
    CutBatchPolicy,
    CutCandidate,
    FixedActionPolicy,
    StateKey,
    action_batch_size,
    encode_state,
)
from rl_benders.cuts import BendersCut
from rl_benders.domain import StochasticFacilityLocationInstance
from rl_benders.master import MasterResult, solve_master
from rl_benders.oracle import first_stage_cost
from rl_benders.subproblem import RecourseResult, evaluate_recourse


@dataclass(frozen=True, slots=True)
class BendersConfig:
    optimality_tolerance: float = 1e-7
    cut_violation_tolerance: float = 1e-7
    max_policy_decisions: int = 80
    max_completion_iterations: int = 120
    stall_limit: int = 3
    minimum_gap_progress: float = 1e-6
    certification_gap_threshold: float = 0.005

    def __post_init__(self) -> None:
        if self.optimality_tolerance <= 0 or self.cut_violation_tolerance <= 0:
            raise ValueError("tolerances must be positive")
        if self.max_policy_decisions <= 0 or self.max_completion_iterations <= 0:
            raise ValueError("iteration limits must be positive")
        if self.stall_limit < 0 or self.minimum_gap_progress < 0:
            raise ValueError("stall controls must be nonnegative")
        if self.certification_gap_threshold < 0:
            raise ValueError("certification_gap_threshold must be nonnegative")


@dataclass(frozen=True, slots=True)
class RewardConfig:
    progress_weight: float = 5.0
    iteration_cost: float = 0.05
    selected_cut_cost: float = 0.01
    master_cut_cost: float = 0.001
    terminal_bonus: float = 2.0
    truncation_penalty: float = 2.0

    def __post_init__(self) -> None:
        if self.progress_weight < 0 or self.iteration_cost < 0:
            raise ValueError("reward weights must be nonnegative")
        if self.selected_cut_cost < 0 or self.master_cut_cost < 0:
            raise ValueError("reward costs must be nonnegative")
        if self.terminal_bonus < 0 or self.truncation_penalty < 0:
            raise ValueError("terminal reward parameters must be nonnegative")


@dataclass(frozen=True, slots=True)
class BendersIterationRecord:
    iteration: int
    phase: str
    requested_action: str
    effective_action: str
    forced_all: bool
    selected_scenarios: tuple[int, ...]
    selected_cut_count: int
    total_cut_count: int
    y: tuple[int, ...]
    lower_bound: float
    incumbent_upper_bound: float
    relative_gap: float
    violated_cut_count: int
    master_seconds: float
    subproblem_seconds: float
    master_nodes: int
    reward: float | None


@dataclass(frozen=True, slots=True)
class BendersResult:
    policy_name: str
    converged: bool
    globally_certified: bool
    objective: float
    lower_bound: float
    upper_bound: float
    y: tuple[int, ...]
    policy_decisions: int
    completion_iterations: int
    completion_phase_used: bool
    cut_count: int
    master_solves: int
    subproblem_solves: int
    total_master_nodes: int
    total_master_seconds: float
    total_subproblem_seconds: float
    total_runtime_seconds: float
    cumulative_reward: float
    action_counts: dict[str, int]
    records: tuple[BendersIterationRecord, ...]


@dataclass(frozen=True, slots=True)
class EnvironmentStep:
    state: StateKey
    action: CutBatchAction
    reward: float
    next_state: StateKey
    terminated: bool
    truncated: bool
    forced_all: bool


@dataclass
class BendersEnvironment:
    """A Gym-like exact Benders control environment.

    The policy chooses only the number of currently violated *valid* cuts to add.
    Every scenario subproblem is solved exactly before the choice is exposed.
    """

    instance: StochasticFacilityLocationInstance
    config: BendersConfig = field(default_factory=BendersConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        self._cuts: list[BendersCut] = []
        self._cut_keys: set[tuple[object, ...]] = set()
        self._observation: BendersObservation | None = None
        self._records: list[BendersIterationRecord] = []
        self._incumbent_upper = float("inf")
        self._incumbent_y: tuple[int, ...] | None = None
        self._policy_decisions = 0
        self._completion_iterations = 0
        self._master_solves = 0
        self._subproblem_solves = 0
        self._total_master_nodes = 0
        self._total_master_seconds = 0.0
        self._total_subproblem_seconds = 0.0
        self._cumulative_reward = 0.0
        self._action_counts = {action.value: 0 for action in CutBatchAction}
        self._started = 0.0

    @property
    def observation(self) -> BendersObservation:
        if self._observation is None:
            raise RuntimeError("environment has not been reset")
        return self._observation

    @property
    def terminated(self) -> bool:
        if self._observation is None:
            return False
        return self._is_certified(self._observation)

    def reset(self) -> BendersObservation:
        self.__post_init__()
        self._started = time.perf_counter()
        self._observation = self._evaluate(stall_count=0)
        return self._observation

    def _evaluate(self, *, stall_count: int) -> BendersObservation:
        master = solve_master(self.instance, self._cuts)
        self._master_solves += 1
        self._total_master_nodes += master.mip_node_count
        self._total_master_seconds += master.runtime_seconds
        expected_recourse, recourse_results = evaluate_recourse(self.instance, master.y)
        self._subproblem_solves += self.instance.scenario_count
        subproblem_seconds = sum(result.runtime_seconds for result in recourse_results)
        self._total_subproblem_seconds += subproblem_seconds
        current_objective = first_stage_cost(self.instance, master.y) + expected_recourse
        if current_objective < self._incumbent_upper - 1e-9 or self._incumbent_y is None:
            self._incumbent_upper = current_objective
            self._incumbent_y = master.y
        lower_bound = master.objective
        relative_gap = max(0.0, self._incumbent_upper - lower_bound) / max(
            1.0, abs(self._incumbent_upper)
        )
        candidates = self._cut_candidates(master, recourse_results)
        return BendersObservation(
            iteration=self._master_solves - 1,
            master=master,
            lower_bound=lower_bound,
            incumbent_upper_bound=self._incumbent_upper,
            incumbent_y=self._incumbent_y,
            current_objective=current_objective,
            current_expected_recourse=expected_recourse,
            relative_gap=relative_gap,
            cut_count=len(self._cuts),
            candidates=candidates,
            stall_count=stall_count,
            subproblem_seconds=subproblem_seconds,
        )

    def _cut_candidates(
        self,
        master: MasterResult,
        recourse_results: tuple[RecourseResult, ...],
    ) -> tuple[CutCandidate, ...]:
        candidates: list[CutCandidate] = []
        for result in recourse_results:
            theta = master.theta[result.scenario_index]
            violation = result.objective - theta
            if violation <= self.config.cut_violation_tolerance:
                continue
            candidates.append(
                CutCandidate(
                    scenario_index=result.scenario_index,
                    cut=result.cut,
                    recourse_objective=result.objective,
                    theta_value=theta,
                    violation=violation,
                    normalized_violation=violation / max(1.0, abs(result.objective)),
                )
            )
        candidates.sort(
            key=lambda candidate: (
                -candidate.normalized_violation,
                -candidate.violation,
                candidate.scenario_index,
            )
        )
        return tuple(candidates)

    def _is_certified(self, observation: BendersObservation) -> bool:
        return (
            not observation.candidates
            and observation.relative_gap <= self.config.optimality_tolerance
        )

    def state(self) -> StateKey:
        return encode_state(self.observation, scenario_count=self.instance.scenario_count)

    def _add_selected_cuts(
        self,
        observation: BendersObservation,
        requested_action: CutBatchAction,
        *,
        force_all: bool,
    ) -> tuple[CutBatchAction, tuple[int, ...], bool]:
        candidates = observation.candidates
        effective_action = CutBatchAction.ALL if force_all else requested_action
        selected_count = action_batch_size(effective_action, len(candidates))
        selected = list(candidates[:selected_count])
        added_scenarios: list[int] = []
        for candidate in selected:
            key = candidate.cut.normalized_key()
            if key in self._cut_keys:
                continue
            self._cuts.append(candidate.cut)
            self._cut_keys.add(key)
            added_scenarios.append(candidate.scenario_index)

        forced = force_all
        if candidates and not added_scenarios:
            forced = True
            effective_action = CutBatchAction.ALL
            for candidate in candidates:
                key = candidate.cut.normalized_key()
                if key in self._cut_keys:
                    continue
                self._cuts.append(candidate.cut)
                self._cut_keys.add(key)
                added_scenarios.append(candidate.scenario_index)
        if candidates and not added_scenarios:
            raise RuntimeError(
                "violated recourse values produced no new Benders cut; check numerical tolerances"
            )
        return effective_action, tuple(added_scenarios), forced

    def _reward(
        self,
        previous: BendersObservation,
        current: BendersObservation,
        selected_cut_count: int,
        *,
        terminated: bool,
        truncated: bool,
    ) -> float:
        improvement = max(0.0, previous.relative_gap - current.relative_gap) / max(
            previous.relative_gap, self.config.optimality_tolerance
        )
        reward = self.reward_config.progress_weight * improvement
        reward -= self.reward_config.iteration_cost
        reward -= self.reward_config.selected_cut_cost * selected_cut_count
        reward -= self.reward_config.master_cut_cost * current.cut_count
        if terminated:
            reward += self.reward_config.terminal_bonus
        if truncated:
            reward -= self.reward_config.truncation_penalty
        return float(reward)

    def step(
        self,
        action: CutBatchAction,
        *,
        phase: str = "policy",
        enforce_policy_limit: bool = True,
        force_all: bool = False,
    ) -> EnvironmentStep:
        previous = self.observation
        if self._is_certified(previous):
            raise RuntimeError("cannot step a certified environment")
        state = self.state()
        automatic_force = (
            previous.stall_count >= self.config.stall_limit
            or previous.relative_gap <= self.config.certification_gap_threshold
        )
        effective_action, selected_scenarios, forced = self._add_selected_cuts(
            previous,
            action,
            force_all=force_all or automatic_force,
        )
        if phase == "policy":
            self._policy_decisions += 1
            self._action_counts[action.value] += 1
        else:
            self._completion_iterations += 1

        new_stall_count = previous.stall_count
        current = self._evaluate(stall_count=new_stall_count)
        gap_progress = previous.relative_gap - current.relative_gap
        if gap_progress > self.config.minimum_gap_progress:
            new_stall_count = 0
        else:
            new_stall_count = previous.stall_count + 1
        if current.stall_count != new_stall_count:
            current = BendersObservation(
                iteration=current.iteration,
                master=current.master,
                lower_bound=current.lower_bound,
                incumbent_upper_bound=current.incumbent_upper_bound,
                incumbent_y=current.incumbent_y,
                current_objective=current.current_objective,
                current_expected_recourse=current.current_expected_recourse,
                relative_gap=current.relative_gap,
                cut_count=current.cut_count,
                candidates=current.candidates,
                stall_count=new_stall_count,
                subproblem_seconds=current.subproblem_seconds,
            )
        self._observation = current
        terminated = self._is_certified(current)
        truncated = (
            enforce_policy_limit
            and phase == "policy"
            and self._policy_decisions >= self.config.max_policy_decisions
            and not terminated
        )
        reward = self._reward(
            previous,
            current,
            len(selected_scenarios),
            terminated=terminated,
            truncated=truncated,
        )
        if phase == "policy":
            self._cumulative_reward += reward
        self._records.append(
            BendersIterationRecord(
                iteration=previous.iteration,
                phase=phase,
                requested_action=action.value,
                effective_action=effective_action.value,
                forced_all=forced,
                selected_scenarios=selected_scenarios,
                selected_cut_count=len(selected_scenarios),
                total_cut_count=len(self._cuts),
                y=previous.master.y,
                lower_bound=previous.lower_bound,
                incumbent_upper_bound=previous.incumbent_upper_bound,
                relative_gap=previous.relative_gap,
                violated_cut_count=previous.violated_count,
                master_seconds=previous.master.runtime_seconds,
                subproblem_seconds=previous.subproblem_seconds,
                master_nodes=previous.master.mip_node_count,
                reward=reward if phase == "policy" else None,
            )
        )
        return EnvironmentStep(
            state=state,
            action=action,
            reward=reward,
            next_state=self.state(),
            terminated=terminated,
            truncated=truncated,
            forced_all=forced,
        )

    def complete_exactly(self) -> None:
        for _ in range(self.config.max_completion_iterations):
            if self.terminated:
                return
            self.step(
                CutBatchAction.ALL,
                phase="completion",
                enforce_policy_limit=False,
                force_all=True,
            )
        raise RuntimeError(
            "exact all-cut completion did not converge within max_completion_iterations"
        )

    def result(self, policy_name: str) -> BendersResult:
        observation = self.observation
        incumbent_y = self._incumbent_y
        if incumbent_y is None:  # pragma: no cover - reset always evaluates a feasible y
            raise RuntimeError("no incumbent first-stage solution")
        return BendersResult(
            policy_name=policy_name,
            converged=self.terminated,
            globally_certified=self.terminated,
            objective=self._incumbent_upper,
            lower_bound=observation.lower_bound,
            upper_bound=self._incumbent_upper,
            y=incumbent_y,
            policy_decisions=self._policy_decisions,
            completion_iterations=self._completion_iterations,
            completion_phase_used=self._completion_iterations > 0,
            cut_count=len(self._cuts),
            master_solves=self._master_solves,
            subproblem_solves=self._subproblem_solves,
            total_master_nodes=self._total_master_nodes,
            total_master_seconds=self._total_master_seconds,
            total_subproblem_seconds=self._total_subproblem_seconds,
            total_runtime_seconds=time.perf_counter() - self._started,
            cumulative_reward=self._cumulative_reward,
            action_counts=dict(self._action_counts),
            records=tuple(self._records),
        )


def solve_with_policy(
    instance: StochasticFacilityLocationInstance,
    policy: CutBatchPolicy,
    *,
    config: BendersConfig | None = None,
    reward_config: RewardConfig | None = None,
    safe_complete: bool = True,
) -> BendersResult:
    environment = BendersEnvironment(
        instance,
        config=config or BendersConfig(),
        reward_config=reward_config or RewardConfig(),
    )
    observation = environment.reset()
    while not environment.terminated:
        state = environment.state()
        action = policy.select(observation, state)
        step = environment.step(action)
        observation = environment.observation
        if step.truncated:
            break
    if not environment.terminated and safe_complete:
        environment.complete_exactly()
    return environment.result(policy.name)


def solve_classical_benders(
    instance: StochasticFacilityLocationInstance,
    *,
    config: BendersConfig | None = None,
) -> BendersResult:
    return solve_with_policy(
        instance,
        FixedActionPolicy(CutBatchAction.ALL),
        config=config,
    )
