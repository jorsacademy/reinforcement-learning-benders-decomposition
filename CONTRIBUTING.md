# Contributing

Contributions should improve mathematical correctness, auditability, reproducibility, or experimental validity.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the complete quality gate:

```bash
ruff check .
ruff format --check .
mypy src
pytest
```

## Pull-request requirements

- Add regression tests for every change to cut generation, master compilation, state encoding, reward, or safety behavior.
- Preserve primal/dual strong-duality checks.
- Preserve the extensive-form objective comparison for certified benchmark rows.
- Never allow a learned component to issue an optimality certificate.
- Do not weaken exact completion or silently increase numerical tolerances.
- Keep training and evaluation seed ranges disjoint.
- Report negative results and RL overhead.
- Document any new action, state feature, reward term, solver option, or stopping criterion.
- Do not commit generated checkpoints, large benchmark artifacts, credentials, or proprietary data.
- Update `CHANGELOG.md` for user-visible changes.

## Scope changes

Adding feasibility cuts, integer recourse, nonlinear subproblems, inexact master solves, or adaptive scenario selection changes the proof obligations. Such changes must include a revised exactness document and independent regression oracle.
