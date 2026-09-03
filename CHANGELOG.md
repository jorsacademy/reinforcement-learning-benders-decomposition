# Changelog

All notable changes will be documented in this file.

## [0.1.0] - 2026-09-03

### Added

- Typed two-stage stochastic capacitated facility-location benchmark with complete recourse.
- Deterministic uniform/clustered and stable/volatile instance generation.
- Exact primal and explicit dual scenario LPs with strong-duality validation.
- Scenario-specific multicut Benders master and duplicate-safe cut pool.
- Extensive-form MILP and complete first-stage-enumeration oracles.
- Safe policy-controlled cut batching with one, quarter, half, and all actions.
- Gap, violation, cut-load, concentration, and stall state encoding.
- Seeded epsilon-greedy tabular Q-learning and versioned NumPy checkpoints.
- Fixed-action, adaptive-heuristic, and random-action baselines.
- Mandatory safety overrides and exact all-cut completion after RL truncation.
- Paired benchmark reporting against the extensive-form optimum.
- Frozen in-distribution and distribution-shift research protocol.
- CLI commands for generation, solving, training, benchmarking, and research runs.
- Python 3.11/3.12 CI with Ruff, strict mypy, branch coverage, and end-to-end smoke tests.
- Exactness, RL formulation, experiment, architecture, research, contribution, security, citation, and licensing documentation.
