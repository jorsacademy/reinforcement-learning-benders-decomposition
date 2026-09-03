# Experiment protocol

## Objective

The experiment tests whether sequential cut-batch control changes Benders work without weakening the exact certificate.

## Data generation

Training and evaluation instances use deterministic synthetic generators with disjoint seed ranges. The generator controls:

- number of facilities, customers, and scenarios;
- spatial regime: uniform or clustered;
- demand regime: stable or volatile;
- capacity factor;
- fixed opening costs;
- shipping and shortage costs.

The generated model always has complete recourse.

## Training

The Q policy is trained only on uniform-location, stable-demand instances. Every episode runs a fresh Benders environment selected from the declared training pool.

The default protocol uses:

```text
24 training instances
5 facilities
8 customers
8 scenarios
240 episodes
```

These defaults are a compact research setting, not a power calculation or a claim of statistical sufficiency.

## Evaluation scenarios

The same frozen policy is evaluated on disjoint seeds under:

1. `in_distribution`;
2. `demand_volatility_shift`;
3. `spatial_cluster_shift`;
4. `scenario_count_shift`;
5. `problem_size_shift`.

Results are not pooled into one opaque score. Each shift is reported separately.

## Baselines

Every instance is solved by:

- all violated cuts;
- one violated cut;
- half of violated cuts;
- an adaptive gap/violation heuristic;
- a seeded random-action controller;
- the trained Q policy.

All methods use the same exact master, recourse solver, cut ranking, safety overrides, and completion contract.

## Reference

Each test instance is solved as a monolithic extensive-form MILP. A benchmark run fails if any certified Benders result disagrees with the reference objective beyond tolerance.

Tiny tests also compare the extensive form against complete enumeration of all binary first-stage vectors.

## Metrics

Report at least:

- objective and objective gap to the extensive form;
- certification rate;
- completion-phase usage;
- policy decisions and completion iterations;
- master and subproblem solve counts;
- cut count;
- master, subproblem, and total time;
- action distribution;
- cumulative reward.

For multiple instances, report means and a descriptive normal-approximation 95% confidence-interval half-width for total runtime. The interval is descriptive and does not replace a paired statistical test.

## Fair-comparison rules

- Run all policies on identical instances.
- Keep solver settings fixed.
- Keep the safety controller fixed.
- Do not tune on evaluation seeds.
- Do not remove unfavorable policies or distribution shifts after observing results.
- Retain negative results, including policies that increase runtime.
- Separate RL training cost from solve-time evaluation.
- Do not call a policy superior based only on reward or action-classification behavior.

## Reproducibility

Store:

- package version and commit SHA;
- frozen JSON configuration;
- training and evaluation seed ranges;
- Q checkpoint;
- raw per-instance rows;
- aggregate report;
- Python, NumPy, SciPy, and HiGHS versions.

The current CLI stores the first five items directly. Environment package versions should be captured by the experiment runner or archival workflow.
