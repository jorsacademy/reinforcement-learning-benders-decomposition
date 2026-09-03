# Reinforcement Learning for Benders Decomposition

[![CI](https://github.com/jorsacademy/reinforcement-learning-benders-decomposition/actions/workflows/ci.yml/badge.svg)](https://github.com/jorsacademy/reinforcement-learning-benders-decomposition/actions/workflows/ci.yml)
[![Python 3.11–3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-orange)](LICENSE)

A verification-first research benchmark for using **reinforcement learning to control cut batching inside Benders decomposition** for a two-stage stochastic capacitated facility-location problem.

The learned controller is deliberately subordinate to the optimizer:

> Reinforcement learning may choose how aggressively valid Benders cuts are added, but it may not certify optimality, alter a cut, skip the final exact checks, or replace the master and recourse solvers.

Every candidate cut is generated from an exact scenario LP. A fixed safety controller forces larger batches when progress stalls or the optimality gap becomes small. If the learned policy reaches its decision horizon before convergence, classical all-cut Benders decomposition completes the solve. A result is reported as globally certified only when the exact master lower bound and a feasible incumbent upper bound coincide within tolerance and no evaluated scenario cut remains violated.

## Research question

Can a sequential policy learn when to add one, a fraction, or all currently violated scenario cuts so that Benders decomposition balances:

- rapid lower-bound improvement;
- master-problem growth;
- number of master re-solves;
- number of generated cuts;
- and final exactness?

The project does not assume that fewer cuts are always better. Adding too few cuts can increase the number of iterations; adding all cuts can make the master unnecessarily large. The RL agent observes solver progress and selects a batch size at every iteration.

## Problem class

The benchmark uses a finite two-stage stochastic capacitated facility-location problem with complete recourse.

First-stage variables:

\[
y_j \in \{0,1\}
\]

indicate whether candidate facility \(j\) is opened. Opening costs are \(f_j\), and capacities are \(K_j\).

For demand scenario \(s\), recourse variables are:

- \(x_{ijs}\ge 0\): quantity shipped from facility \(j\) to customer \(i\);
- \(u_{is}\ge 0\): unmet demand at customer \(i\).

The deterministic equivalent is

\[
\min_y\quad \sum_j f_j y_j + \sum_s p_s Q_s(y),
\]

where

\[
Q_s(y)=\min_{x,u}
\left\{
\sum_{i,j} c_{ij}x_{ijs}+\sum_i q_i u_{is}
\right\}
\]

subject to

\[
\sum_j x_{ijs}+u_{is}\ge d_{is}\qquad \forall i,
\]

\[
\sum_i x_{ijs}\le K_j y_j\qquad \forall j.
\]

The shortage variables provide complete recourse: every binary first-stage vector has a feasible second stage. Shortage penalties exceed available shipping costs so that serving demand is economically meaningful whenever capacity is opened.

## Benders cuts

For one scenario, the recourse dual is

\[
\max_{\pi,\lambda}\quad
\sum_i d_{is}\pi_i-\sum_j K_j y_j\lambda_j
\]

subject to

\[
\pi_i-\lambda_j\le c_{ij},
\qquad
0\le \pi_i\le q_i,
\qquad
\lambda_j\ge 0.
\]

A dual solution produces the globally valid optimality cut

\[
\theta_s \ge
\sum_i d_{is}\pi_i-
\sum_j K_j\lambda_j y_j.
\]

The implementation solves both the primal and dual recourse LPs and checks strong duality before accepting a cut. Tiny regression instances enumerate every binary first-stage vector and verify that every generated cut underestimates the scenario recourse function everywhere on that domain.

## RL control problem

The agent does not choose cut coefficients or scenario duals. Exact recourse evaluation first constructs all currently violated cuts. The agent then chooses one action:

| Action | Added cuts |
| --- | ---: |
| `one` | highest normalized violation |
| `quarter` | top 25% of violated cuts |
| `half` | top 50% of violated cuts |
| `all` | every violated cut |

Cuts are ranked by normalized violation with deterministic tie breaking.

### State

The tabular state is a five-dimensional discretization of:

1. current relative optimality gap;
2. fraction of scenarios with violated cuts;
3. concentration of total violation in the most violated scenario;
4. current cut load per scenario;
5. number of consecutive low-progress iterations.

This creates 432 possible states. The compact state space keeps the policy auditable and makes every learned decision inspectable.

### Reward

The training reward is a deterministic work proxy, not a claim about CPU time:

\[
r_t =
 w_g\frac{g_t-g_{t+1}}{\max(g_t,\epsilon)}
 -w_i
 -w_b|B_t|
 -w_m|C_{t+1}|
 +r_{\text{terminal}}.
\]

Here \(g_t\) is the relative gap, \(B_t\) is the selected cut batch, and \(C_{t+1}\) is the cumulative cut pool. Truncated episodes receive a penalty. Wall-clock time, master nodes, cuts, and solver calls are reported separately during evaluation.

The initial implementation uses seeded epsilon-greedy tabular Q-learning. It is a transparent baseline, not a reproduction of a particular deep-RL paper.

## Safety and exactness contract

The following rules are enforced in code and regression tests:

1. Every scenario is solved exactly before the policy sees the iteration state.
2. The policy selects only among cuts already derived from exact dual-feasible solutions.
3. A cut is never modified by the RL agent.
4. Duplicate cuts are rejected deterministically.
5. If the selected batch adds no new cut, the safety controller attempts all violated cuts.
6. Stalling or a near-closed gap forces the `all` action.
7. Hitting the RL decision horizon does not terminate the optimization algorithm; exact all-cut completion takes over.
8. Global certification requires no violated scenario cut and a closed master–incumbent gap.
9. Every certified result is checked against the monolithic extensive-form MILP in the benchmark harness.

Therefore, an ineffective policy can increase work, use the completion phase, or receive poor reward. It cannot create a false optimality certificate under the declared finite model and solver tolerances.

See [`docs/exactness.md`](docs/exactness.md) for the proof obligations and scope boundary.

## Architecture

```text
stochastic facility-location instance
                 │
                 ▼
        binary Benders master
          y variables + theta_s
                 │
                 ├── exact lower bound
                 │
                 ▼
     exact scenario recourse LPs
       primal + explicit dual
                 │
                 ├── feasible incumbent / upper bound
                 └── valid violated cuts
                              │
                              ▼
                state discretization
                              │
                              ▼
            epsilon-greedy Q policy
             one / quarter / half / all
                              │
                              ▼
                  add selected cuts
                              │
                              ├── safety override on stall
                              └── exact completion on horizon
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Core runtime dependencies are NumPy and SciPy. No commercial solver, external API, or network access is required.

## Quick start

Solve one generated instance with classical all-cut Benders decomposition:

```bash
rl-benders solve \
  --facilities 5 \
  --customers 8 \
  --scenarios 8 \
  --seed 42 \
  --mode all
```

Compare a conservative one-cut policy that can fall back to exact completion:

```bash
rl-benders solve \
  --facilities 5 \
  --customers 8 \
  --scenarios 8 \
  --seed 42 \
  --mode one \
  --max-policy-decisions 20 \
  --include-records
```

Generate and save an instance:

```bash
rl-benders generate \
  --facilities 5 \
  --customers 8 \
  --scenarios 8 \
  --seed 42 \
  --output artifacts/instance.json
```

## Train a Q-learning policy

```bash
rl-benders train \
  --instances 24 \
  --facilities 5 \
  --customers 8 \
  --scenarios 8 \
  --episodes 240 \
  --seed 2026 \
  --checkpoint artifacts/q-policy.npz \
  --output-report artifacts/training.json
```

Training instances use a dedicated seed range. Evaluation should use disjoint seeds.

## Benchmark policies

```bash
rl-benders benchmark \
  --instances 8 \
  --facilities 5 \
  --customers 8 \
  --scenarios 8 \
  --seed 50000 \
  --checkpoint artifacts/q-policy.npz \
  --output-json artifacts/benchmark.json \
  --output-csv artifacts/benchmark.csv
```

The benchmark compares:

- classical all-cut Benders;
- one-cut batching;
- half-cut batching;
- a transparent adaptive heuristic;
- a seeded random-action control;
- the trained Q-learning policy.

Every policy runs on identical instances and is checked against the extensive-form optimum. The harness fails closed if a supposedly certified result disagrees with the monolithic reference.

## Frozen distribution-shift protocol

```bash
rl-benders research \
  --train-instances 24 \
  --evaluation-instances 8 \
  --facilities 5 \
  --customers 8 \
  --scenarios 8 \
  --episodes 240 \
  --seed 2026 \
  --checkpoint artifacts/research-q-policy.npz \
  --output-report artifacts/research-report.json
```

The policy is trained on uniform locations with stable demand and evaluated on five disjoint scenarios:

1. in-distribution instances;
2. higher demand volatility;
3. clustered spatial structure;
4. a larger scenario set;
5. a larger facility/customer problem.

Defaults are frozen in [`configs/research_v1.json`](configs/research_v1.json). See [`docs/experiment_protocol.md`](docs/experiment_protocol.md).

## Reported metrics

Optimization metrics:

- exact extensive-form objective;
- Benders lower and upper bounds;
- relative objective gap;
- certification status;
- first-stage decision vector;
- master solves and scenario-subproblem solves;
- policy decisions and completion iterations;
- total generated cuts;
- master, subproblem, and end-to-end runtime;
- master branch-and-bound nodes where exposed by HiGHS.

RL/control metrics:

- action counts;
- cumulative deterministic reward;
- certification within the policy horizon;
- exact-completion usage rate;
- performance by distribution-shift scenario.

No single weighted leaderboard score hides the distinction between correctness, solver work, and runtime.

## Repository structure

```text
src/rl_benders/
├── domain.py       typed stochastic facility-location model and JSON I/O
├── generator.py    deterministic synthetic instance generator
├── subproblem.py   primal/dual scenario LPs and valid Benders cuts
├── cuts.py         cut representation and fingerprints
├── master.py       binary restricted Benders master
├── oracle.py       extensive-form and first-stage-enumeration references
├── control.py      actions, state encoding, and non-learning policies
├── benders.py      certified environment, safety layer, and solve loop
├── rl.py           tabular Q-learning, training, and checkpoints
├── benchmark.py    repeated-policy comparison and JSON/CSV reporting
├── experiment.py   frozen train/shift-evaluation protocol
└── cli.py          command-line interface
```

## Tests and CI

```bash
ruff check .
ruff format --check .
mypy src
pytest
```

The test suite covers:

- deterministic instance generation and JSON round trips;
- primal/dual recourse strong duality;
- global validity of generated cuts over every binary first-stage vector on a tiny instance;
- extensive-form MILP versus complete first-stage enumeration;
- classical Benders versus the exact oracle;
- safe completion after a deliberately short RL horizon;
- rejection of uncertified truncated results;
- state-space bounds and Q-learning updates;
- checkpoint round trips;
- policy benchmarking and the distribution-shift protocol;
- CLI generate, solve, train, and benchmark paths.

GitHub Actions runs linting, formatting, strict type checking, branch-aware coverage, and an end-to-end train/solve smoke test on Python 3.11 and 3.12.

## Methodological boundaries

This repository does **not** claim:

- industrial-scale stochastic programming performance;
- that the tabular controller is state of the art;
- that the deterministic reward proxy predicts wall-clock time;
- that a policy trained on synthetic instances transfers to production data;
- that cut-batch control dominates cut classification, subproblem selection, stabilization, or inexact-master control;
- that observed runtime differences on small HiGHS models generalize to commercial solvers;
- convergence guarantees for arbitrary nonlinear generalized Benders decomposition;
- asynchronous or parallel scenario solution;
- branch-and-Benders-cut integration.

The project isolates one controlled question: sequential selection of valid cut-batch sizes under an exact safety envelope.

## Research context

The repository is motivated by several modern lines of work:

- supervised classification of valuable Benders cuts;
- ML-assisted scenario and cut selection;
- reinforcement learning for controlling inexact Benders master solves;
- graph-based imitation and reinforcement learning for Benders master/subproblem acceleration.

This implementation does not reproduce those architectures. It provides a smaller, auditable benchmark where learning cannot silently weaken the mathematical certificate. See [`docs/research_context.md`](docs/research_context.md).

## References

1. Benders, J. F. (1962). Partitioning procedures for solving mixed-variables programming problems. *Numerische Mathematik*, 4, 238–252. https://doi.org/10.1007/BF01580645
2. Jia, H., & Shen, S. (2021). Benders Cut Classification via Support Vector Machines for Solving Two-Stage Stochastic Programs. *INFORMS Journal on Optimization*, 3(3), 278–297. https://doi.org/10.1287/ijoo.2019.0050
3. Li, Z., Agyeman, B. T., Mitrai, I., & Daoutidis, P. (2026). Learning to control inexact Benders decomposition via reinforcement learning. *Computers & Chemical Engineering*, 205, 109461. https://doi.org/10.1016/j.compchemeng.2025.109461
4. Agyeman, B. T., Li, Z., Mitrai, I., & Daoutidis, P. (2025). Graph-Based Imitation and Reinforcement Learning for Efficient Benders Decomposition. https://arxiv.org/abs/2511.11870
5. Donkiewicz, T. (2026). Adaptive Subproblem Selection in Benders Decomposition for Survivable Network Design Problems. https://arxiv.org/abs/2604.09031

## License

This project is source-available under the **PolyForm Noncommercial License 1.0.0**. Commercial use is not granted. It is not OSI Open Source. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
