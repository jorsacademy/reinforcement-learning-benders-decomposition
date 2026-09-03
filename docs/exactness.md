# Exactness and safety contract

This document states what the implementation certifies and what it does not.

## 1. Complete recourse

For every scenario and every first-stage vector, unmet-demand variables are nonnegative and appear in each demand constraint. Consequently, every recourse LP is feasible. Positive shortage penalties make the recourse objective bounded below.

No feasibility cuts are needed for the declared model. Every generated Benders cut is an optimality cut.

## 2. Validity of a scenario cut

For a fixed scenario, any dual-feasible pair `(pi, lambda)` satisfies weak duality:

```text
Q_s(y) >= demand_s · pi - sum_j capacity_j * lambda_j * y_j
```

for every first-stage vector `y` in `[0,1]^J`. The code solves an explicit dual LP and creates exactly this affine function.

Before accepting the cut, the implementation also solves the primal recourse LP and checks primal–dual objective agreement within a scaled numerical tolerance. This is not needed for weak-duality validity, but it detects sign, indexing, and solver-integration errors.

Regression tests generate cuts at multiple source points and evaluate them against exact recourse at every binary first-stage vector on a hand-checkable fixture.

## 3. Lower and upper bounds

The Benders master contains only globally valid lower estimators of scenario recourse. Its optimal value is therefore a lower bound on the stochastic program.

Evaluating all exact recourse LPs at a binary master solution produces a feasible first-stage/recourse policy. Its expected cost is an upper bound. The best upper bound seen so far is retained as the incumbent.

## 4. What the RL policy can change

The policy sees the set of exact violated cuts and selects only a batch size. It cannot:

- change coefficients;
- create a cut;
- delete an existing cut;
- change the first-stage or recourse model;
- change solver tolerances;
- declare convergence.

Cuts are ranked by normalized violation, so the action determines a deterministic prefix of the candidate list.

## 5. Safety overrides

The environment overrides the requested action with `all` when:

- the gap is inside the configured certification threshold; or
- the gap has failed to make sufficient progress for the configured number of iterations.

If a requested batch contains only duplicate cuts, all violated cuts are attempted. If no new cut can be added despite a recourse violation, the algorithm raises an error rather than continuing silently.

## 6. Policy horizon and exact completion

The RL phase has a finite decision horizon. Reaching it is **truncation**, not optimization convergence.

With `safe_complete=True`, the current exact cut pool and incumbent are retained and the algorithm switches to classical all-cut Benders iterations. This phase continues until certification or until a separate hard completion limit is reached. Exceeding that limit raises an error.

A truncated result returned with `safe_complete=False` is explicitly marked uncertified.

## 7. Certification condition

The result is globally certified for the declared finite model only when both conditions hold:

1. every exact scenario recourse value is within cut-violation tolerance of its master `theta` value;
2. the relative difference between the incumbent upper bound and master lower bound is within optimality tolerance.

The benchmark harness then independently solves the extensive-form MILP. Any disagreement beyond tolerance fails the experiment.

## 8. Finite convergence scope

With an exact binary master, exact LP subproblems, valid cuts, and complete recourse, classical Benders decomposition converges finitely on this finite first-stage domain. The safety-completion phase restores that classical algorithm even when the learned phase behaves poorly.

This statement applies to the implemented finite stochastic facility-location model. It is not a generic convergence proof for nonlinear generalized Benders decomposition, inexact subproblems, approximate duals, or asynchronous algorithms.

## 9. Numerical scope

SciPy/HiGHS uses floating-point arithmetic. The repository uses explicit tolerances, independent primal/dual solves, exhaustive tiny-instance tests, and an extensive-form reference. These controls reduce implementation risk but do not constitute exact rational arithmetic.
