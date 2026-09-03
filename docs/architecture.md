# Architecture

## Trusted optimization layer

- `domain.py`: validated finite stochastic model.
- `subproblem.py`: primal and explicit dual recourse LPs.
- `cuts.py`: immutable cut representation and fingerprints.
- `master.py`: exact binary Benders master.
- `oracle.py`: monolithic and enumeration references.

## Control layer

- `control.py`: action enumeration, state encoding, and transparent baselines.
- `benders.py`: environment, safety overrides, bounds, certification, and completion.
- `rl.py`: Q table, epsilon-greedy selection, updates, and checkpoints.

## Evaluation layer

- `benchmark.py`: paired per-instance comparisons and machine-readable reports.
- `experiment.py`: frozen training and distribution-shift protocol.
- `cli.py`: user-facing commands.

The dependency direction is one-way: the optimization layer does not import or trust the RL layer.
