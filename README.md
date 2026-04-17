# Low-Thrust Trajectory Optimization: Earth-to-Mars SEP Transfers

## Project Overview

This project develops and analyzes continuous low-thrust electric propulsion (EP)
trajectories for Earth-to-Mars transfers using solar electric propulsion (SEP).
Unlike impulsive maneuvers, SEP systems operate at high specific impulse over
extended burn arcs, requiring trajectory optimization techniques that account for
continuous thrust throughout the transfer.

## Objectives

Two optimization problems are formulated and solved:

- **Time-Optimal**: Minimize total transfer time subject to propellant and thrust
  constraints. The thrust direction is the primary control variable, and the
  optimizer determines the steering law that achieves the shortest feasible
  transfer.

- **Mass-Optimal**: Minimize propellant mass consumption (equivalently, maximize
  final spacecraft mass) for a fixed transfer time. The optimizer determines the
  combined thrust magnitude and direction history that delivers the spacecraft
  with the greatest possible mass margin.

## Approach

### Coordinate System

Trajectories are modeled in a **2D heliocentric polar coordinate system**
(r, θ) centered on the Sun. The spacecraft state is described by radial
distance, polar angle, radial velocity, and tangential velocity. This
formulation is natural for interplanetary transfers and simplifies the equations
of motion for planar orbits.

### Transcription Method

A **direct single-shooting** transcription is used. The continuous trajectory
is parameterized by a finite set of control variables (thrust angles and/or
magnitudes at discrete nodes). The equations of motion are integrated forward
from the departure state using these controls, and boundary conditions at Mars
arrival are enforced as nonlinear equality constraints.

### NLP Solver

The resulting finite-dimensional nonlinear programming (NLP) problem is solved
with **SLSQP** (Sequential Least Squares Programming) via `scipy.optimize.minimize`.
SLSQP handles both equality constraints (boundary conditions) and inequality
constraints (thrust limits, propellant budget) and is well-suited to
moderate-dimensional trajectory optimization problems.

## Modules

| Module | Description |
|--------|-------------|
| `baseline_impulsive.py` | Computes the classical two-impulse Hohmann-like transfer between Earth and Mars orbits. Provides delta-V and time-of-flight baselines against which low-thrust solutions are compared. |
| `time_optimal.py` | Implements the time-optimal low-thrust solver. Optimizes thrust steering history to minimize transfer time while satisfying terminal boundary conditions and propellant constraints. |
| `mass_optimal.py` | Implements the mass-optimal low-thrust solver. Optimizes thrust magnitude and steering to minimize propellant consumption over a fixed time horizon, maximizing delivered mass. |
| `nonplanar_extension.py` | Extends the 2D planar model to 3D by incorporating an out-of-plane thrust component. Accounts for orbital inclination differences and enables more realistic Earth-Mars transfers with nonzero relative inclination. |

## Milestone Structure

| Week | Milestone |
|------|-----------|
| 1 | Set up environment, implement heliocentric equations of motion in polar coordinates, validate against analytical two-body propagation |
| 2 | Implement baseline impulsive model; compute Hohmann transfer delta-V and time-of-flight as performance benchmarks |
| 3 | Implement direct single-shooting transcription; integrate state and co-state equations; set up SLSQP problem structure |
| 4 | Solve time-optimal problem; analyze thrust steering angles and trajectory shape; compare transfer time to impulsive baseline |
| 5 | Solve mass-optimal problem; analyze throttle and steering history; compare propellant consumption across objectives |
| 6 | Implement nonplanar extension; study effect of inclination change on optimal solutions; produce final plots and report |

## Dependencies

Install all required packages with:

```bash
pip install -r requirements.txt
```

- `numpy` — numerical arrays and linear algebra
- `scipy` — ODE integration (`solve_ivp`) and NLP solver (`minimize` with SLSQP)
- `matplotlib` — trajectory and control history plots
- `astropy` — planetary ephemeris data and unit conversions

## Results

Output figures and data files are saved to the `results/` directory.
