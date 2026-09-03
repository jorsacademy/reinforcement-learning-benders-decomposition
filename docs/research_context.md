# Research context

Benders decomposition has a long history as an exact decomposition method for mixed-variable and two-stage stochastic programs. Modern learning-enhanced work asks which expensive solver decisions can be predicted or controlled without sacrificing correctness.

## Valuable-cut learning

Jia and Shen classify potentially valuable Benders cuts for two-stage stochastic programs. Their LearnBD framework demonstrates that cut selection can be treated as a supervised-learning problem.

This repository addresses a different decision: after all currently violated exact cuts are known, how large should the added batch be?

## Reinforcement learning for algorithm control

Li, Agyeman, Mitrai, and Daoutidis use reinforcement learning to choose master-problem optimality gaps in inexact generalized Benders decomposition. Their state and reward balance per-iteration effort against optimization progress.

The present project borrows the sequential-control perspective but keeps every master solve exact and assigns the agent a smaller action space: cut-batch size.

## Graph-based learned Benders components

Recent work uses graph imitation learning, reinforcement learning, and self-supervised models to propose master decisions or approximate subproblem primal-dual solutions, combined with verification mechanisms.

Those systems target a broader and more difficult acceleration problem. This repository intentionally establishes a lower-complexity reference implementation with an explicit exact fallback before adding graph policies or learned subproblem surrogates.

## Adaptive subproblem selection

Scenario-rich Benders methods can avoid solving every subproblem at every iteration by predicting which scenarios will generate useful cuts. That can reduce subproblem work, but it introduces an additional certification question because unevaluated scenarios may hide violations.

The current implementation always solves every scenario. This makes its cut set fully observed and its certificate simple. Adaptive subproblem selection is a future extension that would require scheduled full sweeps or another safe stopping rule.

## Position of this repository

The contribution is methodological rather than scale-oriented:

```text
exact multicut Benders
+ sequential RL control
+ deterministic work reward
+ safety overrides
+ exact completion
+ monolithic verification
+ distribution-shift protocol
```

It is intended as an auditable baseline for later DQN/PPO, GNN state encoders, inexact-master schedules, scenario selection, and branch-and-Benders-cut experiments.
