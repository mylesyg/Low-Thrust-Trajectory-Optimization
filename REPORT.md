# Low-Thrust Earth-to-Mars Trajectory Optimization: A Comprehensive Guide

**Course:** AEROSP 548 — Astrodynamics, Spring 2026  
**Project 2**

---

## Table of Contents

1. [Introduction and Motivation](#1-introduction-and-motivation)
2. [Background: Astrodynamics Concepts for Beginners](#2-background-astrodynamics-concepts-for-beginners)
   - 2.1 [The Solar System as a Physics Problem](#21-the-solar-system-as-a-physics-problem)
   - 2.2 [Orbital Mechanics Fundamentals](#22-orbital-mechanics-fundamentals)
   - 2.3 [Spacecraft Propulsion: Rockets 101](#23-spacecraft-propulsion-rockets-101)
   - 2.4 [Impulsive vs. Low-Thrust Propulsion](#24-impulsive-vs-low-thrust-propulsion)
   - 2.5 [What Does "Optimal" Mean?](#25-what-does-optimal-mean)
3. [Mathematical Foundations](#3-mathematical-foundations)
   - 3.1 [Coordinate Systems](#31-coordinate-systems)
   - 3.2 [The 2D Equations of Motion](#32-the-2d-equations-of-motion)
   - 3.3 [The 3D Equations of Motion](#33-the-3d-equations-of-motion)
   - 3.4 [Non-Dimensional Scaling](#34-non-dimensional-scaling)
   - 3.5 [Hohmann Transfer: The Impulsive Baseline](#35-hohmann-transfer-the-impulsive-baseline)
   - 3.6 [Optimal Control and the NLP Formulation](#36-optimal-control-and-the-nlp-formulation)
4. [Code Architecture](#4-code-architecture)
   - 4.1 [Directory Structure](#41-directory-structure)
   - 4.2 [Data Flow Through the Pipeline](#42-data-flow-through-the-pipeline)
5. [Module-by-Module Reference](#5-module-by-module-reference)
   - 5.1 [core/constants.py](#51-coreconstantspy)
   - 5.2 [core/nondimensional.py](#52-corenondimensionalpy)
   - 5.3 [core/equations_of_motion.py](#53-coreequations_of_motionpy)
   - 5.4 [config/mission_config.py](#54-configmission_configpy)
   - 5.5 [ephemeris/planetary_states.py](#55-ephemerisplanetary_statespy)
   - 5.6 [baseline/impulsive_transfer.py](#56-baselineimpulsive_transferpy)
   - 5.7 [baseline/lambert_solver.py](#57-baselinelambert_solverpy)
   - 5.8 [propagator/integrator.py](#58-propagatorintegratorpy)
   - 5.9 [optimization/shooting.py](#59-optimizationshootingpy)
   - 5.10 [optimization/time_optimal.py](#510-optimizationtime_optimalpy)
   - 5.11 [optimization/mass_optimal.py](#511-optimizationmass_optimalpy)
   - 5.12 [analysis/pareto.py](#512-analysisparetospy)
   - 5.13 [nonplanar/eom_3d.py](#513-nonplanareom_3dpy)
   - 5.14 [nonplanar/nonplanar_optimizer.py](#514-nonplanarnonplanar_optimizerpy)
   - 5.15 [visualization/](#515-visualization)
     - [trajectory_plot.py — 2D path, thrust arrows](#trajectory_plotpy)
     - [porkchop_plot.py — launch window C3 and arrival DeltaV](#porkchop_plotpy)
     - [control_plot.py — steering angle and mass history](#control_plotpy)
     - [pareto_plot.py — Pareto frontier and power sweep](#pareto_plotpy)
   - 5.16 [main.py](#516-mainpy)
6. [How to Run the Code](#6-how-to-run-the-code)
   - 6.1 [Prerequisites and Installation](#61-prerequisites-and-installation)
   - 6.2 [Running Each Week's Milestone](#62-running-each-weeks-milestone)
   - 6.3 [Quick Mode for Fast Testing](#63-quick-mode-for-fast-testing)
7. [Expected Results and How to Interpret Them](#7-expected-results-and-how-to-interpret-them)
   - 7.1 [Week 1: The Hohmann Baseline](#71-week-1-the-hohmann-baseline)
   - 7.2 [Week 2: Time-Optimal Transfer](#72-week-2-time-optimal-transfer)
   - 7.3 [Week 3: Mass-Optimal and Pareto Analysis](#73-week-3-mass-optimal-and-pareto-analysis)
   - 7.4 [Week 4: 3D Nonplanar Extension](#74-week-4-3d-nonplanar-extension)
8. [Key Design Decisions and Engineering Trade-offs](#8-key-design-decisions-and-engineering-trade-offs)
9. [Glossary](#9-glossary)
10. [Low-Thrust Spiral Descent in Mars Orbit](#10-low-thrust-spiral-descent-in-mars-orbit)
    - 10.1 [Motivation and Physical Setup](#101-motivation-and-physical-setup)
    - 10.2 [State Representation: Modified Equinoctial Elements](#102-state-representation-modified-equinoctial-elements)
    - 10.3 [Gauss Variational Equations for Tangential Thrust](#103-gauss-variational-equations-for-tangential-thrust)
    - 10.4 [Time-Optimal Control Law](#104-time-optimal-control-law)
    - 10.5 [Edelbaum Analytical Solution for Circular Spirals](#105-edelbaum-analytical-solution-for-circular-spirals)
    - 10.6 [Mars-Centric Non-Dimensional Scaling](#106-mars-centric-non-dimensional-scaling)
    - 10.7 [Interpretation of the Demonstration Trajectory](#107-interpretation-of-the-demonstration-trajectory)
11. [Low-Thrust Inclination Change in Mars Orbit](#11-low-thrust-inclination-change-in-mars-orbit)
    - 11.1 [Motivation and Physical Setup](#111-motivation-and-physical-setup)
    - 11.2 [State Representation: Modified Equinoctial Elements](#112-state-representation-modified-equinoctial-elements)
    - 11.3 [Gauss Variational Equations for Out-of-Plane Thrust](#113-gauss-variational-equations-for-out-of-plane-thrust)
    - 11.4 [Bang-Bang Normal Thrust Control Law](#114-bang-bang-normal-thrust-control-law)
    - 11.5 [Time-of-Flight Estimate and Impulsive Baseline](#115-time-of-flight-estimate-and-impulsive-baseline)
    - 11.6 [Mars-Centric Non-Dimensional Scaling](#116-mars-centric-non-dimensional-scaling)
    - 11.7 [Interpretation of the Demonstration Trajectory](#117-interpretation-of-the-demonstration-trajectory)

---

## 1. Introduction and Motivation

Sending a spacecraft from Earth to Mars is one of the most technically challenging missions in the solar system. Unlike science-fiction starships that can thrust continuously at enormous accelerations, real spacecraft must work within tight constraints of fuel mass, engine power, and orbital mechanics.

This project focuses on **low-thrust electric propulsion** — specifically Solar Electric Propulsion (SEP). Instead of a massive chemical rocket burn lasting only minutes, SEP thrusters fire continuously for months, gently nudging the spacecraft onto an ever-widening spiral path from Earth's orbit to Mars's orbit. The trade-off is significant:

- **Chemical rockets** are powerful but inefficient — they burn lots of propellant per unit of speed change.
- **Electric thrusters** are weak but extremely efficient — they extract far more speed change per kilogram of propellant, at the cost of requiring many months of continuous operation.

The core mathematical question this project answers is: **given a fixed spacecraft and thruster, what is the best way to point the engine over the entire journey?** "Best" can mean different things — maybe you want to get there as fast as possible (time-optimal), or maybe you want to arrive with as much propellant leftover as possible (mass-optimal). Both problems are formulated and solved here.

---

## 2. Background: Astrodynamics Concepts for Beginners

### 2.1 The Solar System as a Physics Problem

Think of the Sun as a bowling ball sitting on a rubber sheet. Each planet rolls in a curved groove carved by the Sun's gravity, tracing out a nearly circular path called an **orbit**. Earth is closer to the Sun (about 150 million km, or 1 Astronomical Unit, AU), so it moves faster — completing one lap every 365 days. Mars is farther away (about 228 million km, or 1.52 AU), so it moves more slowly — taking about 687 days per lap.

The key insight from Isaac Newton and later Johannes Kepler is that the Sun's gravity alone determines how fast each planet moves. There is no engine, no air resistance, no friction in space. An object at distance *r* from the Sun moves at the **circular orbital speed**:

```
v_circ = sqrt(mu_sun / r)
```

where `mu_sun` is the Sun's gravitational parameter (≈ 1.327 × 10¹¹ km³/s²). At Earth's distance, this is about 29.8 km/s. At Mars's distance, it is about 24.1 km/s.

### 2.2 Orbital Mechanics Fundamentals

#### Kepler's Laws

1. **Every orbit is an ellipse** with the Sun at one focus. Circles are just special ellipses.
2. **A line from the Sun to the spacecraft sweeps equal areas in equal times.** This means spacecraft move faster when closer to the Sun and slower when farther away.
3. **The square of the orbital period is proportional to the cube of the semi-major axis:** T² ∝ a³.

#### Orbital Elements

A Keplerian orbit in 3D space is completely described by six **orbital elements**:

| Element | Symbol | Description |
|---|---|---|
| Semi-major axis | a | Size of the orbit (average of closest and farthest distance) |
| Eccentricity | e | Shape: 0 = circle, 0–1 = ellipse, 1 = parabola |
| Inclination | i | Tilt of the orbit plane relative to the ecliptic (Earth-Sun plane) |
| RAAN | Ω | Right Ascension of Ascending Node — where the orbit crosses the ecliptic going north |
| Argument of periapsis | ω | Orientation of the ellipse within the orbit plane |
| True anomaly | ν | Current position of the spacecraft on the orbit |

For this project, we mostly care about **a** (orbit size), **i** (inclination), and **Ω** (RAAN) when dealing with the 3D extension.

#### The Vis-Viva Equation

This equation tells you the speed of a spacecraft at any point on its orbit:

```
v² = mu * (2/r - 1/a)
```

For a circular orbit (r = a), this simplifies to `v = sqrt(mu/r)`. This equation is used extensively in the Hohmann transfer calculations.

#### Specific Angular Momentum

The specific angular momentum vector **h** = **r** × **v** is perpendicular to the orbit plane and has magnitude:

```
h = |r × v| = r * v * sin(angle between r and v)
```

For a circular orbit, **r** and **v** are perpendicular, so h = r × v. The direction of **h** determines the orbit's inclination and RAAN — this is heavily used in the 3D constraint formulation.

### 2.3 Spacecraft Propulsion: Rockets 101

Every rocket works by throwing mass backward and using Newton's third law to push forward. The efficiency of a rocket engine is measured by **specific impulse** (Isp), which is the thrust you get per unit weight of propellant consumed per second. Higher Isp = more efficient engine.

| Propulsion Type | Thrust | Isp (s) | Example |
|---|---|---|---|
| Chemical (liquid) | 100 kN – 1 MN | 300–450 | SpaceX Merlin |
| Ion thruster (SEP) | 0.1–500 mN | 1500–10000 | Dawn mission Hall thruster |

The fundamental **Tsiolkovsky rocket equation** relates the speed change (ΔV) a rocket can achieve to its propellant consumption:

```
ΔV = Isp * g₀ * ln(m₀ / m_f)
```

where:
- `g₀ = 9.80665 m/s²` (standard gravity — used to give Isp consistent units)
- `m₀` = initial (wet) mass including propellant
- `m_f` = final (dry) mass after burning

Rearranging: `m_f / m₀ = exp(-ΔV / (Isp * g₀))`

A higher Isp means you need a smaller mass fraction to achieve the same ΔV. For Isp = 3000 s and ΔV = 5000 m/s:
```
m_f/m₀ = exp(-5000 / (3000 × 9.81)) ≈ 0.842
```
So only about 15.8% of the initial mass is consumed — very efficient compared to chemical rockets which might use 80%+ for the same ΔV.

The **mass flow rate** (propellant consumption rate) for a thruster is:

```
ṁ = T / (g₀ × Isp)
```

where T is thrust in Newtons. For T = 3.5 N and Isp = 3000 s: ṁ ≈ 0.000119 kg/s ≈ 10.3 kg/day.

### 2.4 Impulsive vs. Low-Thrust Propulsion

**Impulsive maneuvers** assume the burn happens instantaneously — the spacecraft jumps from one velocity to another in zero time. This is a useful idealization for chemical rockets (burns last minutes compared to months-long coast arcs). The Hohmann transfer uses two impulsive burns.

**Low-thrust transfers** cannot use this approximation. The engine fires continuously for months, and the trajectory is a continuous spiral rather than a sequence of elliptic arcs with instantaneous jumps. This makes the mathematics significantly more complex — you cannot solve it analytically and must use numerical optimization.

The key consequence: in the equations of motion, the spacecraft mass changes continuously as propellant is consumed. At time t, if a fraction `mp` of the initial mass has been consumed, the remaining mass is `m(t) = m₀ × (1 - mp)`, and the current thrust acceleration is:

```
a_m = T / m(t) = (T / m₀) / (1 - mp) = a_thrust_ref / (1 - mp)
```

As propellant depletes, the thrust-to-mass ratio increases slightly, meaning the spacecraft actually accelerates slightly faster toward the end.

### 2.5 What Does "Optimal" Mean?

An **optimization problem** has three ingredients:

1. **Decision variables** — what you can control (here: the direction and magnitude of thrust at each moment in time)
2. **Objective function** — what you want to minimize or maximize (transfer time, or propellant consumed)
3. **Constraints** — conditions that must be satisfied (must arrive at Mars's orbit with the right speed)

This project solves two versions:

- **Time-optimal:** Minimize total transfer time. The spacecraft thrusts at full power continuously (throttle = 1), and the optimizer finds the best steering angle history.
- **Mass-optimal:** For a fixed transfer time, minimize the amount of propellant burned. The optimizer adjusts both the steering angles and a throttle setting (how hard the engine fires).

---

## 3. Mathematical Foundations

### 3.1 Coordinate Systems

#### 2D Heliocentric Polar Coordinates

The 2D model places the Sun at the origin and describes the spacecraft position by:
- **r**: radial distance from Sun [AU]
- **θ**: polar angle (longitude) [rad]

The spacecraft's velocity is decomposed into:
- **u**: radial velocity component (positive = moving away from Sun) [AU/day]
- **v**: transverse (tangential) velocity component (positive = moving in direction of increasing θ, i.e., prograde) [AU/day]

This is a natural coordinate system for circular-orbit problems because:
- A circular orbit has u = 0 and v = sqrt(mu/r) = constant
- Earth starts at r = 1 AU, u = 0, v = v_Earth
- Mars target is at r = 1.52368 AU, u = 0, v = sqrt(mu/r_Mars)

The **control variable** is the **steering angle** α (alpha), measured from the transverse (prograde) direction toward the outward radial direction:

```
α = 0      → pure prograde thrust (tangential, in direction of motion)
α = +π/2   → pure outward radial thrust (away from Sun)
α = -π/2   → pure inward radial thrust (toward Sun)
```

The thrust vector components in polar coordinates are:
```
F_r  = a_m × sin(α)    [radial component]
F_θ  = a_m × cos(α)    [transverse component]
```

#### 3D Cartesian Heliocentric Coordinates

For the 3D extension, the state is expressed in inertial Cartesian coordinates:
- Position: (x, y, z) with x-y being the ecliptic plane and z pointing north
- Velocity: (vx, vy, vz)

Two control angles are used:
- **α (alpha)**: in-plane azimuth angle, measured from the +x axis in the ecliptic plane
- **β (beta)**: out-of-plane declination angle, in [-π/2, π/2]; positive = thrust toward ecliptic north

The thrust unit vector in Cartesian coordinates is:
```
û = [cos(β)cos(α),  cos(β)sin(α),  sin(β)]
```

### 3.2 The 2D Equations of Motion

The **state vector** for the 2D problem is a 6-element array:

```
y = [r, u, v, θ, mp, acc_dv]
```

where:
- `r` : radial distance [AU, ND]
- `u` : radial velocity [AU/t_cf, ND]
- `v` : transverse velocity [AU/t_cf, ND]
- `θ` : polar angle [rad]
- `mp` : accumulated propellant mass fraction — starts at 0, increases as fuel burns
- `acc_dv` : accumulated ΔV (integral of thrust acceleration magnitude over time)

The **equations of motion** (derivatives of the state with respect to time) are:

```
ṙ     = u

u̇     = v²/r  -  μ*/r²  +  a_m × sin(α)
         [centrifugal]  [gravity]  [radial thrust]

v̇     = -u×v/r          +  a_m × cos(α)
         [Coriolis]         [tangential thrust]

θ̇     = v / r

ṁp    = throttle × ṁp_ref

acc_dv̇ = a_m
```

where the **instantaneous thrust acceleration** is:

```
a_m = throttle × a_thrust_ref / (1 - mp)
```

The denominator `(1 - mp)` correctly accounts for the decreasing spacecraft mass as propellant is consumed. Without this term, the EOM would underestimate thrust late in the mission.

The first two terms in the u̇ equation deserve explanation:
- `v²/r`: **centrifugal acceleration** — in polar coordinates, a body moving tangentially creates an apparent outward force
- `-μ*/r²`: **gravitational acceleration** — the Sun pulls the spacecraft inward. With μ* = 1 (non-dimensional), this simplifies to just `-1/r²`

The first term in v̇:
- `-u×v/r`: **Coriolis acceleration** — a radially-moving body in a rotating frame also changes its tangential velocity

These three "fictitious" terms arise because we are using a non-inertial rotating polar coordinate frame. They are not actual forces — they are mathematical corrections needed when working in polar coordinates.

### 3.3 The 3D Equations of Motion

The 3D state vector is length 8:

```
y = [x, y, z, vx, vy, vz, mp, acc_dv]
```

The equations of motion are much simpler in Cartesian coordinates because there are no centrifugal or Coriolis corrections — inertial Cartesian frames are non-rotating:

```
ẋ  = vx
ẏ  = vy
ż  = vz

v̇x = -μ* × x / r³  +  a_m × cos(β)cos(α)
v̇y = -μ* × y / r³  +  a_m × cos(β)sin(α)
v̇z = -μ* × z / r³  +  a_m × sin(β)

ṁp = throttle × ṁp_ref

acc_dv̇ = a_m
```

where `r = sqrt(x² + y² + z²)` and `r³ = (x² + y² + z²)^(3/2)`.

The three velocity equations each have: (gravity term) + (thrust term). The gravity term has the Sun's gravitational pull split among the three Cartesian directions proportional to the displacement from the Sun.

#### Terminal Constraints for the 3D Problem

The 3D optimizer enforces **five terminal orbital-element constraints** at Mars arrival:

```
c1: |r_f| / r_Mars  - 1              = 0    (correct orbital radius)
c2: (r_f · v_f) / (r_Mars × v_circ) = 0    (zero radial velocity = circular orbit)
c3: (|v_f| - v_circ) / v_circ       = 0    (correct speed for circular orbit)
c4: h_fz / |h_f| - cos(i₂)          = 0    (correct inclination)
c5: (h_fx×cos(Ω₂) + h_fy×sin(Ω₂)) / |h_f| = 0    (correct RAAN)
```

where **h_f = r_f × v_f** is the specific angular momentum at arrival.

Constraints c1–c3 enforce the geometry of a circular orbit at Mars's distance. Constraints c4–c5 enforce the 3D orientation of that orbit (inclination and the ascending node direction).

### 3.4 Non-Dimensional Scaling

Raw physical quantities span many orders of magnitude and can cause numerical precision issues in solvers. The code **non-dimensionalizes** all quantities using a consistent scaling scheme:

| Quantity | Dimensional Unit | Non-Dimensional Unit |
|---|---|---|
| Distance | AU (1 AU = 149,597,870 km) | 1 AU (r* = 1 AU) |
| Time | day | t_cf = sqrt(1/μ_sun) ≈ 58.13 days |
| Velocity | AU/day | AU/t_cf |
| Acceleration | AU/day² | AU/t_cf² = μ_sun/AU² |

The key consequence of this normalization is that the **non-dimensional gravitational parameter μ* = 1 exactly**. This means:
- Earth's circular orbital radius: r_Earth = 1.0 [ND]
- Earth's circular orbital speed: v_Earth = sqrt(μ*/r) = sqrt(1/1) = 1.0 [ND]
- Mars's circular orbital radius: r_Mars = 1.52368 [ND] (same numerical value as AU)
- Mars's circular orbital speed: v_Mars = sqrt(1/1.52368) ≈ 0.8101 [ND]

**Converting thrust acceleration to ND:**

Starting from the dimensional thrust acceleration a = T/m₀ [m/s²]:
```
Step 1: a [m/s²] → a [AU/day²]: multiply by (86400)²/AU_m
Step 2: a [AU/day²] → a [ND]: divide by (μ_sun [AU³/day²] / 1 AU²) = μ_sun [AU/day²]
```

So: `acc_thrust_ND = (T/m₀) × (86400)² / AU_m / μ_sun`

In code: `a_si = T/m0; a_au_day2 = a_si * 86400² / AU_m; acc_nd = a_au_day2 / mu_sun`

**Converting mass flow rate to ND:**
```
ṁ_dim = T / (g₀ × Isp)   [kg/s]
ṁ_dim × 86400 / m₀        [1/day]  (specific mass flow per day)
×  t_cf                    [ND]     (multiply by time unit)
```

So: `mp_dot_ND = (T / (g₀ × Isp)) × 86400 / m₀ × t_cf`

One non-dimensional time unit = t_cf ≈ 58.13 days. One non-dimensional year ≈ 2π ND time units (exactly, because v_Earth = 1 and r_Earth = 1, so the orbital period is 2π/1 = 2π in ND time).

### 3.5 Hohmann Transfer: The Impulsive Baseline

The **Hohmann transfer** is the energy-optimal two-burn trajectory between two coplanar, circular orbits. It is the gold standard baseline against which low-thrust trajectories are compared.

**Geometry:** The transfer orbit is an ellipse with periapsis at Earth's orbit and apoapsis at Mars's orbit:

```
a_transfer = (r_Earth + r_Mars) / 2 = (1.0 + 1.52368) / 2 = 1.26184 AU
```

**Transfer time** (half the elliptical period):

```
t_Hohmann = π × sqrt(a_transfer³ / μ_sun) ≈ 259 days
```

**Delta-V calculations** using the vis-viva equation:

At departure (perigee of transfer ellipse, still at r = r_Earth):
```
v_transfer_peri = sqrt(μ × (2/r_Earth - 1/a_transfer))
ΔV₁ = v_transfer_peri - v_Earth_circ
    ≈ 2945 m/s  (prograde burn to raise apoapsis to Mars)
```

At arrival (apogee of transfer ellipse, at r = r_Mars):
```
v_transfer_apo = sqrt(μ × (2/r_Mars - 1/a_transfer))
ΔV₂ = v_Mars_circ - v_transfer_apo
    ≈ 2648 m/s  (prograde burn to circularize at Mars)
```

**Total ΔV ≈ 5593 m/s** for an Earth-to-Mars Hohmann transfer.

**Propellant mass consumed** (sequential Tsiolkovsky):
```
After burn 1: m₁ = m₀ × exp(-ΔV₁ / (g₀ × Isp))
After burn 2: m_f = m₁ × exp(-ΔV₂ / (g₀ × Isp))
Propellant = m₀ - m_f
```

For Isp = 3000 s and m₀ = 5000 kg: propellant ≈ 476 kg (about 9.5% of initial mass).

The low-thrust optimizer must do better than this on mass (fewer kg of propellant per kg of spacecraft), typically at the cost of more time.

### 3.6 Optimal Control and the NLP Formulation

#### The Control Problem

We want to find the **control history** (steering angle as a function of time) that minimizes an objective while satisfying constraints. This is called an **optimal control problem**.

The exact mathematical solution uses **Pontryagin's Maximum Principle** and **Hamiltonian mechanics** — very powerful but complex. Instead, this code uses a simpler **direct transcription** approach that converts the infinite-dimensional optimal control problem into a finite-dimensional **Nonlinear Program (NLP)**.

#### Direct Single-Shooting Transcription

The idea is simple:

1. **Divide time** into N equal segments of duration Δt = TOF/N
2. **Hold the control constant** within each segment (piecewise-constant approximation)
3. **Integrate the EOM** forward through all N segments to get the final state
4. **Enforce terminal constraints** as NLP equality constraints
5. **Feed everything into a standard NLP solver** (SLSQP)

The **decision vector** (the N+1 numbers the optimizer adjusts) for the time-optimal case is:

```
x = [tof_nd,  α₁,  α₂,  ...,  α_N]
     ↑ time    ↑ steering angles per segment
```

For mass-optimal:
```
x = [throttle,  α₁,  α₂,  ...,  α_N]
     ↑ uniform   ↑ steering angles per segment
       throttle
```

#### SLSQP: Sequential Least Squares Programming

SLSQP is a gradient-based optimizer that:
1. Linearizes the constraints around the current guess
2. Solves a **quadratic programming** (QP) subproblem to find the next step
3. Updates the guess and repeats until constraints are satisfied and the objective stops improving

The constraint Jacobian (how constraints change with each decision variable) is approximated using **finite differences**:

```
∂c/∂x_k ≈ (c(x + ε × e_k) - c(x)) / ε
```

where `e_k` is the unit vector in direction k and `ε` is a small step size.

A key optimization in this code: the **shared Jacobian**. Normally, computing the Jacobian of 4 quantities (1 objective + 3 constraints) over N+1 variables would require 4(N+1) trajectory integrations per iteration. By noting that a single perturbed integration `shooting(x + ε × e_k)` gives values for all 4 quantities simultaneously, we need only N+1 integrations per iteration — a 4× speedup.

---

## 4. Code Architecture

### 4.1 Directory Structure

```
low_thrust_optimization/
├── analysis/
│   ├── __init__.py
│   └── pareto.py                   # Pareto frontier and power sweep analysis
├── baseline/
│   ├── __init__.py
│   ├── impulsive_transfer.py       # Hohmann transfer analytical solution
│   └── lambert_solver.py           # Lambert's problem solver
├── config/
│   ├── __init__.py
│   └── mission_config.py           # Mission parameters dataclass
├── core/
│   ├── __init__.py
│   ├── constants.py                # Physical constants (μ_sun, AU, g₀, t_cf)
│   ├── equations_of_motion.py      # 2D polar heliocentric EOM
│   └── nondimensional.py           # Unit conversion utilities
├── ephemeris/
│   ├── __init__.py
│   └── planetary_states.py         # Planet positions, departure date estimation
├── nonplanar/
│   ├── __init__.py
│   ├── eom_3d.py                   # 3D Cartesian EOM + orbital element conversions
│   └── nonplanar_optimizer.py      # 3D time-optimal NLP solver
├── optimization/
│   ├── __init__.py
│   ├── shooting.py                 # Core constraint evaluation (shooting function)
│   ├── time_optimal.py             # Time-optimal SLSQP solver
│   └── mass_optimal.py             # Mass-optimal SLSQP solver
├── propagator/
│   ├── __init__.py
│   └── integrator.py               # Multi-segment trajectory propagator
├── visualization/
│   ├── __init__.py
│   ├── control_plot.py             # Steering angle and mass history plots
│   ├── pareto_plot.py              # Pareto frontier and power sweep plots
│   ├── porkchop_plot.py            # Launch-window C3 and arrival DeltaV pork chop plots
│   └── trajectory_plot.py          # 2D transfer path, state history, and thrust arrow overlay
├── main.py                         # Top-level CLI pipeline driver
├── README.md                       # Project overview
└── requirements.txt                # Python package dependencies
```

### 4.2 Data Flow Through the Pipeline

Here is how data flows when you run the time-optimal solver:

```
User runs: python main.py --week 2
                    │
                    ▼
            main.py::run_week2()
                    │
       Creates MissionConfig (spacecraft params)
                    │
                    ▼
       solve_time_optimal(config)          ← time_optimal.py
                    │
          Coarse warm-start (N=25)
          → SLSQP solve
          → Interpolate to fine grid (N=400)
          → Fine SLSQP solve
                    │
         At each SLSQP function call:
                    ▼
         shooting_function(x, nd, mode)    ← shooting.py
                    │
          Decode x = [tof, α₁, ..., α_N]
          Get initial state (Earth departure)
                    │
                    ▼
         propagate_trajectory(y0, tof, α)  ← integrator.py
                    │
          Loop over N segments:
            propagate_segment()            ← integrator.py
              calls solve_ivp(eom_2d_ivp)  ← equations_of_motion.py
                    │
          Returns final state y_f
                    │
         Compute constraints:
           c0 = r_f - r_Mars
           c1 = u_f - 0
           c2 = v_f - v_Mars_circ
                    │
                    ▼
         SLSQP adjusts x and iterates until constraints ≈ 0
                    │
                    ▼
         Post-process: convert ND → days, kg
         Generate trajectory and control plots
```

---

## 5. Module-by-Module Reference

### 5.1 core/constants.py

**Purpose:** Defines all physical constants as plain Python floats so they can be used inside numerical solvers without incurring overhead from unit-aware libraries.

**Key constants:**

| Name | Value | Units | Description |
|---|---|---|---|
| `mu_sun_au3day2` | ≈ 2.959 × 10⁻⁴ | AU³/day² | Sun's gravitational parameter |
| `mu_sun_km3s2` | 1.3274 × 10¹¹ | km³/s² | Same, in km/s units |
| `AU_km` | 149,597,870.691 | km | 1 Astronomical Unit |
| `AU_m` | 1.496 × 10¹¹ | m | Same, in meters |
| `r_earth` | 1.0 | AU | Earth's mean orbital radius |
| `r_mars` | 1.52368 | AU | Mars's mean orbital radius |
| `g0` | 9.80665 | m/s² | Standard gravity (exact, SI definition) |
| `t_cf` | ≈ 58.13 | days | Non-dimensional time unit |

**How t_cf is computed:**
```python
t_cf = sqrt(1.0 / mu_sun_au3day2)
```
This is the time T* such that in ND units, the period of a circular orbit at r* = 1 AU is exactly 2π, and the circular speed is exactly 1. It can be thought of as the "natural clock rate" of the solar system.

**Dependencies:** Uses `astropy.constants` and `astropy.units` to get accurate IAU 2015 values at import time, then caches them as plain floats.

---

### 5.2 core/nondimensional.py

**Purpose:** Provides the unit-conversion functions to go between dimensional (meters, seconds, kilograms) and non-dimensional quantities used throughout the integrators and optimizers.

**Class:** `NonDimensional` — a collection of static methods.

**Key methods:**

#### `time_to_nd(t_days)` and `time_to_dim(t_nd)`
```python
t_nd = t_days / t_cf      # days → ND time
t_days = t_nd * t_cf      # ND time → days
```

#### `velocity_to_nd(v_au_per_day)`
```python
v_nd = v_au_per_day * t_cf
```
Sanity check: Earth's circular speed ≈ 0.01720 AU/day → 0.01720 × 58.13 ≈ 1.0 ND.

#### `compute_thrust_acc(thrust_N, mass_kg)`
Converts SI thrust acceleration to non-dimensional:
```python
a_si = thrust_N / mass_kg           # m/s²
a_au_day2 = a_si * 86400² / AU_m   # AU/day²
acc_nd = a_au_day2 / mu_sun         # ND
```
For T = 3.5 N and m₀ = 5000 kg: `acc_thrust_ND ≈ 0.118`

#### `compute_mass_flow(thrust_N, isp_s, mass_kg)`
Converts the propellant mass flow rate to non-dimensional:
```python
m_dot_kg_s = thrust_N / (g0 * isp_s)        # kg/s (dimensional)
mp_dot_nd = m_dot_kg_s * 86400 / mass_kg * t_cf  # ND
```
For T = 3.5 N, Isp = 3000 s, m₀ = 5000 kg: `mp_dot_ND ≈ 0.118`

Note: `acc_thrust_ND ≈ mp_dot_ND` for these parameters. This is not a coincidence — they are both proportional to T/m₀ and have similar unit conversion factors. The ratio differs when Isp changes.

---

### 5.3 core/equations_of_motion.py

**Purpose:** Implements the 2D heliocentric polar equations of motion — the heart of the dynamical model.

#### `eom_2d(t, y, alpha, acc_thrust, mp_dot, throttle)`

This is the right-hand side of the ODE system. Given the current state `y` and control `alpha`, it returns the time derivatives `dy/dt`.

**Inputs:**
- `t`: current time (not explicitly used in the EOM, but required by scipy's ODE interface)
- `y = [r, u, v, θ, mp, acc_dv]`: current state vector (6 elements)
- `alpha`: steering angle [rad]
- `acc_thrust`: reference ND thrust acceleration (T/m₀ normalized)
- `mp_dot`: reference ND mass flow rate
- `throttle`: throttle in [0, 1]

**Core computation (annotated):**
```python
mass_frac = max(1.0 - mp, 1e-6)        # remaining mass fraction; guard against division by zero
a_m = throttle * acc_thrust / mass_frac  # current thrust acceleration magnitude

r_dot     = u                                             # velocity of radial position
u_dot     = (v*v)/r  -  1.0/(r*r)  +  a_m*sin(alpha)   # radial accel: centrifugal - gravity + thrust_r
v_dot     = -(u*v)/r              +  a_m*cos(alpha)     # tangential accel: Coriolis + thrust_θ
theta_dot = v / r                                        # angular velocity
mp_dot_val= throttle * mp_dot                           # propellant consumption rate
dvdt      = a_m                                          # delta-V accumulation rate
```

#### `eom_2d_ivp(t, y, params)`

A thin wrapper around `eom_2d` that unpacks a parameter dictionary. This exists because `scipy.integrate.solve_ivp` requires the function signature to be `f(t, y)` — you cannot pass extra arguments directly. The wrapper closes over the `params` dict.

#### `terminal_conditions_mars()`

Returns the target state at Mars arrival:
```python
r_f = r_mars = 1.52368  [ND]
u_f = 0.0               [ND]  (no radial motion in circular orbit)
v_f = sqrt(1.0/r_mars)  [ND]  (circular speed at Mars = sqrt(μ*/r_Mars))
```

The polar angle θ_f is **unconstrained** (free final angle). This means the optimizer does not care where in its orbit Mars is when the spacecraft arrives — only that the spacecraft is in a circular orbit at the right radius with the right speed. This is physically reasonable: we can design the launch date so Mars is at the right place, but the optimizer works independently of calendar dates.

---

### 5.4 config/mission_config.py

**Purpose:** A central configuration object (Python dataclass) that holds all mission parameters in one place. Think of it as the "settings file" for a simulation run.

**Class:** `MissionConfig`

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `initial_mass_kg` | 5000 kg | Wet mass (including propellant) |
| `thrust_N` | 3.5 N | Maximum SEP thrust |
| `isp_s` | 3000 s | Engine specific impulse |
| `opt_mode` | 'time_optimal' | Optimization objective |
| `n_segments` | 400 | Number of shooting segments N |
| `time_guess_days` | 200 days | Initial guess for transfer time |
| `time_lb_days` | 175 days | Lower bound on transfer time |
| `time_ub_days` | 225 days | Upper bound on transfer time |
| `throttle_guess` | 1.0 | Initial throttle guess |
| `alpha_guess_deg` | 20 deg | Initial steering angle guess |

**Why 3.5 N thrust on a 5000 kg spacecraft?** The thrust-to-mass ratio is 3.5/5000 = 0.0007 m/s² = 0.7 mm/s². Compare this to Earth's gravity at sea level: 9.81 m/s². This is an incredibly weak thruster — roughly 1/14,000th of Earth's surface gravity. The spacecraft would barely feel it if you held it in your hand, but over 200 days of continuous operation, it accumulates enough ΔV to spiral from Earth's orbit to Mars's orbit.

**Factory methods:**

`MissionConfig.default_time_optimal()` — returns a configuration appropriate for minimizing transfer time (full throttle, TOF free variable, N=400 segments).

`MissionConfig.default_mass_optimal()` — returns a configuration for minimizing propellant (variable throttle, fixed TOF=200 days, N=400 segments).

**Method `to_nd_params()`** converts the dimensional parameters to ND quantities for use in the integrator and optimizer:
```python
nd = config.to_nd_params()
# Returns dict with: acc_thrust, mp_dot, tof_nd, tof_lb_nd, tof_ub_nd
```

---

### 5.5 ephemeris/planetary_states.py

**Purpose:** Provides planetary positions and departure date estimates. "Ephemeris" is the astronomical term for a table of planetary positions as a function of time.

#### `get_planet_longitude(planet_name, jd)`

Returns the heliocentric ecliptic longitude of Earth or Mars at a given Julian Date, using low-precision polynomial expressions from Meeus's "Astronomical Algorithms":

```
Earth: L = 100.466449 + 35999.3728519 × T - 0.00000568 × T²   [deg]
Mars:  L = 355.433275 + 19140.2993313 × T + 0.00000261 × T²   [deg]
```

where T is Julian centuries from J2000.0 (January 1, 2000 at noon): `T = (JD - 2451545.0) / 36525.0`

These are accurate to about 1 degree over 1800–2050.

#### `compute_departure_date(jd_guess, transfer_time_days, transfer_angle_deg)`

Uses a **phasing equation** to find when Earth and Mars are in the right relative position for a transfer of given duration and arc angle.

The logic: At departure, Earth is at longitude θ_E. After `t_T` days, the spacecraft has swept `θ_T` degrees. For Mars to be at the arrival point, we need:

```
θ_Mars(departure) + ω_Mars × t_T = θ_Earth(departure) + θ_T
```

Linearizing around the initial guess gives:
```
Δt = (θ_Mars + ω_Mars×t_T - θ_Earth - θ_T) / (ω_Earth - ω_Mars)
```

This is the correction to add to the initial Julian Date guess to find the actual launch window.

#### `compute_synodic_period()`

The **synodic period** is the time between successive Earth-Mars launch windows:

```
1/T_syn = 1/T_Earth - 1/T_Mars
T_syn = T_Earth × T_Mars / (T_Mars - T_Earth) ≈ 779.95 days ≈ 2.135 years
```

This is why Mars missions launch roughly every 26 months.

#### `get_initial_state_earth()`

Returns the non-dimensional initial state for Earth departure:
```python
y0 = [1.0,   # r = 1 AU (Earth's orbital radius, ND)
      0.0,   # u = 0 (no radial velocity in circular orbit)
      1.0,   # v = 1 (Earth's circular speed, ND — this equals 1 by construction)
      0.0,   # θ = 0 (reference departure longitude)
      0.0,   # mp = 0 (no propellant consumed yet)
      0.0]   # acc_dv = 0 (no ΔV accumulated yet)
```

#### `get_final_state_mars()`

Returns the target final state (NaN for unconstrained elements):
```python
yf = [1.52368,    # r = r_Mars
      0.0,        # u = 0 (circular orbit)
      0.8101,     # v = sqrt(1/r_Mars) (Mars circular speed, ND)
      nan,        # θ_f = free
      nan,        # mp_f = free (result of optimization)
      nan]        # acc_dv_f = free (result of optimization)
```

---

### 5.6 baseline/impulsive_transfer.py

**Purpose:** Computes the two-impulse Hohmann transfer for use as a performance baseline.

#### `hohmann_transfer(r1_au, r2_au)`

Given departure and arrival orbit radii, analytically computes:

```python
a_t = (r1 + r2) / 2                              # transfer SMA
v_peri = sqrt(mu * (2/r1 - 1/a_t))              # speed at periapsis (departure point)
v_apo  = sqrt(mu * (2/r2 - 1/a_t))              # speed at apoapsis (arrival point)
delta_v1 = abs(v_peri - sqrt(mu/r1))            # departure burn
delta_v2 = abs(sqrt(mu/r2) - v_apo)             # arrival burn
tof = pi * sqrt(a_t³ / mu)                      # transfer time (half period)
```

Returns a dictionary with `delta_v1_auday`, `delta_v2_auday`, `total_delta_v_auday`, `tof_days`.

#### `compute_baseline(config)`

Wraps `hohmann_transfer` and adds Tsiolkovsky propellant computation using the mission's Isp. The sequential Tsiolkovsky correctly accounts for the fact that the spacecraft has already consumed propellant in the first burn before performing the second:

```python
ve = g0 * isp                    # effective exhaust speed [m/s]
m1 = m0 * exp(-dv1/ve)          # mass after departure burn
mf = m1 * exp(-dv2/ve)          # mass after arrival burn
propellant = m0 - mf            # total propellant consumed
```

**Expected results for the default configuration:**
- Transfer time: ~259 days
- ΔV₁ (departure): ~2945 m/s
- ΔV₂ (arrival): ~2648 m/s
- Total ΔV: ~5593 m/s
- Propellant consumed: ~476 kg (9.5% of 5000 kg) for Isp=3000 s

---

### 5.7 baseline/lambert_solver.py

**Purpose:** Solves Lambert's problem — given two position vectors and a transfer time, find the velocity vectors that connect them. This is useful for more realistic trajectory design that accounts for actual planetary positions.

The implementation uses the **Battin-Mueller-White (BMW) universal variable formulation** with Stumpff functions C(z) and S(z) for numerical robustness. The solver uses Newton-Raphson iteration to find the universal variable that satisfies the transfer time equation, then computes departure and arrival velocities using Lagrange coefficients.

This module is available for advanced analysis but is not in the main optimization pipeline for the weekly milestones.

---

### 5.8 propagator/integrator.py

**Purpose:** Integrates the equations of motion over multiple segments to propagate a full trajectory.

#### `propagate_segment(y0, t_start, t_end, alpha, acc_thrust, mp_dot, throttle, tol)`

Integrates the EOM over a single segment [t_start, t_end] with constant steering angle `alpha`. Uses `scipy.integrate.solve_ivp` with the **DOP853** method (Dormand-Prince 8th order Runge-Kutta) — a high-accuracy adaptive integrator.

Key settings:
- `rtol = atol = tol = 1e-10` (relative and absolute tolerance — very tight)
- `dense_output = False` (no need for intermediate values, only the endpoint)

If the integrator fails (e.g., step-size becomes too small due to ill-conditioned dynamics), it raises a `RuntimeError`, which is caught by the shooting function and converted into a penalty result.

#### `propagate_trajectory(y0, tof_nd, alphas, acc_thrust, mp_dot, throttle, tol)`

Loops over all N segments in sequence:
```python
delta_t = tof_nd / N
time_history = linspace(0, tof_nd, N+1)   # avoids float accumulation errors
state_history = zeros((N+1, 6))
state_history[0] = y0

for k in range(N):
    y_current = propagate_segment(y_current, time_history[k], time_history[k+1], 
                                   alphas[k], ...)
    state_history[k+1] = y_current
```

Returns a dictionary with:
- `'final_state'`: the state at the end of the last segment — this is what gets compared to Mars terminal conditions
- `'state_history'`: all N+1 state vectors (including initial state) — used for plotting
- `'time_history'`: all N+1 time values — used for plotting

**Why DOP853 instead of the default RK45?** DOP853 is an 8th-order method compared to RK45's 4/5th-order. For smooth right-hand sides (like gravitational + smooth thrust), higher-order methods achieve the same accuracy with fewer function evaluations (fewer calls to `eom_2d`), making the integrator faster overall.

---

### 5.9 optimization/shooting.py

**Purpose:** The "shooting function" — evaluates the objective and all constraints for a given decision vector. This is the function called by SLSQP at every iteration.

The name "shooting" comes from the analogy of shooting a bullet: you specify the initial conditions and "shoot" forward to see where you land. If you don't land at the target (Mars), you adjust the angle (steering) and try again.

#### `shooting_function(x, config_nd, mode, tol)`

**Input:** Decision vector `x`, plus configuration and mode.

**Processing:**
1. Decode `x`:
   - Time-optimal: `tof_nd = x[0]`, `alphas = x[1:]`, `throttle = 1.0`
   - Mass-optimal: `throttle = x[0]`, `alphas = x[1:]`, `tof_nd` from config
2. Guard against invalid inputs (negative TOF → return penalty)
3. Get initial state: `y0 = get_initial_state_earth()`
4. Call `propagate_trajectory(y0, tof_nd, alphas, ...)`
5. Extract final state `y_f = [r_f, u_f, v_f, ...]`
6. Compute terminal constraint residuals:
   ```python
   constraints = [r_f - r_Mars,    # should be 0
                  u_f - 0.0,       # should be 0
                  v_f - v_Mars]    # should be 0
   ```
7. Compute objective:
   - Time-optimal: `objective = acc_dv` (accumulated ΔV, monotonically related to transfer time)
   - Mass-optimal: `objective = throttle` (the throttle setting being minimized)

**Penalty handling:** If integration fails or produces NaN, returns `objective = 1e6` and `constraints = [1e6, 1e6, 1e6]`. This large penalty value steers SLSQP away from the numerically problematic region rather than crashing.

---

### 5.10 optimization/time_optimal.py

**Purpose:** Top-level solver for the time-optimal problem. Wraps SLSQP with a multi-resolution warm-start strategy.

#### Problem Formulation

```
Minimize:   J = acc_dv   (accumulated ΔV ≡ transfer time at full throttle)
Subject to: r_f = r_Mars          (arrive at Mars radius)
            u_f = 0               (circular orbit)
            v_f = v_Mars_circ     (correct circular speed)
            175 ≤ TOF ≤ 225 days  (time bounds)
            -π ≤ αₖ ≤ π           (steering angle bounds, k=1..N)
```

Decision vector: `x = [tof_nd, α₁, α₂, ..., α_N]` (length N+1)

#### Multi-Resolution Warm-Start

For N=400, a cold start (all αₖ = 20°) may take 80–100 SLSQP iterations because the optimizer is far from the solution basin. The warm-start strategy:

1. **Solve coarse problem:** Use N=25 segments (same TOF bounds and spacecraft). This is ~(400/25)² = 256 times cheaper in Jacobian computation cost.
2. **Interpolate:** Linearly interpolate the 25 optimized steering angles onto the 400-segment grid.
3. **Solve fine problem:** Start from the interpolated guess, which is already near the solution.

The fine-grid solve typically takes only 10–20 iterations with a warm start vs. 80–100 from cold.

#### Shared Jacobian Computation

The `_run_slsqp` function implements the shared Jacobian. The key insight: to compute the gradient of all 4 quantities (1 objective + 3 constraints) with respect to all N+1 decision variables, we need N+1 perturbed trajectory integrations — not 4(N+1).

```python
f0 = shooting_function(x, ...)          # baseline evaluation
J = zeros((4, N+1))                      # 4 rows: obj + 3 constraints

for k in range(N+1):
    xp = x.copy()
    xp[k] += eps                        # perturb one variable
    fp = shooting_function(xp, ...)     # one propagation gives ALL 4 values
    J[:, k] = (fp - f0) / eps          # fill entire column of Jacobian
```

#### Parallel Jacobian Evaluation

When `n_jobs > 1`, the N+1 perturbed propagations can run simultaneously on different CPU cores using `ProcessPoolExecutor`. This gives near-linear speedup on multi-core machines.

The worker function `_jacobian_worker` must be a module-level function (not a closure/lambda) so it can be serialized (pickled) and sent to worker processes.

#### `solve_time_optimal(config, x0_override, integ_tol, maxiter, n_jobs)`

**Returns dictionary with:**
- `'success'`: True if SLSQP converged AND all constraints |cᵢ| < 1e-6
- `'tof_days'`: optimal transfer time in days
- `'alphas_rad'`: optimal steering angles for each segment [rad]
- `'propellant_mass_kg'`: propellant consumed = `yfinal[4] × m₀`
- `'final_mass_kg'`: remaining mass = `m₀ × (1 - yfinal[4])`
- `'state_history'`: (N+1, 6) array of state vectors at each node
- `'time_history_days'`: (N+1,) array of times in days
- `'solver_message'`: human-readable SLSQP exit status
- `'n_iterations'`: number of SLSQP major iterations
- `'wall_time_s'`: total elapsed time in seconds

---

### 5.11 optimization/mass_optimal.py

**Purpose:** Solver for the mass-optimal problem — find the minimum throttle that allows the spacecraft to reach Mars in a fixed transfer time.

#### Problem Formulation

```
Minimize:   J = throttle   (uniform throttle setting ≡ propellant consumed)
Subject to: r_f = r_Mars          (arrive at Mars radius)
            u_f = 0               (circular orbit)
            v_f = v_Mars_circ     (correct circular speed)
            TOF = 200 days (fixed)
            0 ≤ throttle ≤ 1
            -π ≤ αₖ ≤ π
```

Decision vector: `x = [throttle, α₁, α₂, ..., α_N]` (length N+1)

**Why is minimizing throttle the same as minimizing propellant?**

The propellant consumed is:
```
Δm = (T/(g₀ × Isp)) × throttle × TOF
```
Since T, g₀, Isp, and TOF are all constants in this problem, minimizing `throttle` is exactly equivalent to minimizing `Δm`.

#### Analytic Objective Gradient

For the mass-optimal problem, the objective is simply `J = x[0] = throttle`. Its gradient is exactly:
```
∇J = [1, 0, 0, ..., 0]   (1 in the throttle direction, 0 elsewhere)
```

This is provided analytically to SLSQP, eliminating the need for a finite-difference evaluation of the objective gradient. Only the constraint Jacobian requires finite differences — a further computational savings over the time-optimal case.

#### `solve_mass_optimal(config, x0_override, integ_tol, maxiter, n_jobs)`

**Returns dictionary with:**
- `'success'`: True if converged
- `'tof_days'`: the fixed transfer time (from config)
- `'throttle'`: optimal uniform throttle setting (the minimized quantity)
- `'alphas_rad'`: optimal steering angles
- `'propellant_mass_kg'`: propellant consumed
- `'final_mass_kg'`: remaining mass
- `'state_history'`, `'time_history_days'`, `'solver_message'`, `'n_iterations'`, `'wall_time_s'`

**Key physics:** If you give the spacecraft more time (larger TOF), it can achieve the same orbital transfer with less thrust (lower throttle), thus burning less propellant. This is the fundamental trade-off between transfer time and propellant efficiency.

---

### 5.12 analysis/pareto.py

**Purpose:** Conducts systematic sweeps across design space to trace trade-off curves.

#### Pareto Frontier: TOF vs. Propellant

`generate_pareto_frontier(base_config, tof_range_days, n_points, ...)` sweeps through a range of transfer times and solves the mass-optimal problem at each point.

- Sweeps from longest to shortest TOF (sequential mode): easier problems (high TOF, low throttle) first, using the converged solution as a warm start for the next harder problem.
- Each converged point is stored as a dictionary containing summary statistics and the **full trajectory data** needed for visualization: `tof_days`, `propellant_fraction`, `throttle`, `state_history` (N+1, 6), `alphas_rad` (N,), and `time_history_days` (N+1,). Storing trajectory data alongside each Pareto point avoids a second optimizer run when a specific point is selected for plotting.
- The resulting curve shows the Pareto frontier: the minimum propellant for any given transfer time. Points above the curve are suboptimal; points below are physically infeasible.

**Expected shape:** Monotonically decreasing — longer transfer time → lower propellant fraction. The minimum time point (left end) is at full throttle and minimum propellant for that time. As TOF increases, throttle decreases, and less propellant is burned.

#### Power Sweep: Thrust vs. Transfer Time

`generate_power_sweep(base_config, thrust_range_N, n_points, ...)` sweeps through different SEP thrust levels (corresponding to different electrical power systems) and solves the time-optimal problem at each.

For each thrust level, the **specific power** is computed:
```
P_electric = T × Isp × g₀ / (2 × η)   [W]
P_sp = P_electric / m₀                  [W/kg]
```

where η = 0.65 is the assumed thruster efficiency. This formula comes from the jet power of an ion thruster: P = (1/2) × ṁ × v_exhaust² = T × v_exhaust / 2 = T × Isp × g₀ / 2. Dividing by m₀ gives specific power.

**Expected shape:** Higher thrust (more power) → shorter transfer time, but requires a heavier power system (more W/kg).

---

### 5.13 nonplanar/eom_3d.py

**Purpose:** Implements the 3D Cartesian equations of motion for the nonplanar extension.

#### `eom_3d(t, y, alpha, beta, acc_thrust, mp_dot, throttle)`

8-state EOM in Cartesian coordinates. The key difference from 2D is:
- No centrifugal or Coriolis terms (inertial Cartesian frame is non-rotating)
- Two control angles: alpha (in-plane azimuth) and beta (out-of-plane declination)
- Gravity in all three directions: `-μ* × [x, y, z] / r³`

#### `cartesian_to_orbital_elements(r_vec, v_vec, mu)`

Converts a Cartesian state `(r_vec, v_vec)` to classical orbital elements. This is used to check constraints at the terminal state.

The algorithm follows Curtis (2020) "Orbital Mechanics for Engineering Students":

1. **Specific angular momentum:** `h = r × v`
2. **Orbital energy:** `ε = v²/2 - μ/r`
3. **Semi-major axis:** `a = -μ/(2ε)`
4. **Eccentricity vector:** `e = (v × h)/μ - r̂`
5. **Inclination:** `i = arccos(h_z / |h|)`
6. **Node vector:** `N = ẑ × h`
7. **RAAN:** `Ω = arccos(N_x / |N|)`, with quadrant check
8. **Argument of periapsis:** `ω = arccos(N·e / (|N||e|))`, with quadrant check
9. **True anomaly:** `ν = arccos(e·r / (|e||r|))`, with quadrant check

Handles degenerate cases: equatorial orbits (RAAN undefined → NaN), circular orbits (ω undefined → NaN).

#### `propagate_3d_trajectory(y0, tof_nd, alphas, betas, ...)`

Same structure as the 2D propagator but with 8-element state and two control angle arrays per segment.

---

### 5.14 nonplanar/nonplanar_optimizer.py

**Purpose:** Extends the time-optimal NLP to 3D with five orbital element constraints.

#### `_initial_state_from_elements(a_au, i_deg, raan_deg)`

Builds the 8-element initial state for a circular orbit at the ascending node:
```
r_vec0 = r1 × [cos(Ω), sin(Ω), 0]
v_vec0 = vc1 × [-cos(i)×sin(Ω), cos(i)×cos(Ω), sin(i)]
```

When i=0 and Ω=0 (equatorial orbit, same as 2D Earth case), this reduces exactly to the 2D initial state [1, 0, 1] in the x-y plane.

#### Decision Vector for 3D

```
x = [tof_nd, α₁, ..., α_N, β₁, ..., β_N]   length = 2N+1
```

Both steering angles (alpha for in-plane, beta for out-of-plane) are decision variables for every segment.

#### Terminal Constraints (5 equations)

```python
r_f = norm(y_f[:3])           # distance from Sun
v_f_vec = y_f[3:6]           # velocity vector
h_f = cross(r_f_vec, v_f_vec) # angular momentum

c1 = r_f / r2 - 1.0                              # radius constraint
c2 = dot(r_f_vec, v_f_vec) / (r2 * vc2)         # circular (zero radial v)
c3 = (norm(v_f_vec) - vc2) / vc2                 # circular speed
c4 = h_f[2] / norm(h_f) - cos(i2)               # inclination
c5 = (h_f[0]*cos(Ω2) + h_f[1]*sin(Ω2)) / norm(h_f)  # RAAN
```

The RAAN constraint (c5) is derived from the condition that the orbit-normal vector `h/|h|` must be perpendicular to the RAAN direction. If RAAN is defined as the angle from the +x axis to the ascending node, then the orbit normal has components `[sin(i)sin(Ω), -sin(i)cos(Ω), cos(i)]`, and requiring h to be parallel to this gives the formula for c5.

---

### 5.15 visualization/

Three visualization modules produce publication-quality figures.

#### trajectory_plot.py

##### `plot_2d_trajectory(result_dict, config, n_thrust_arrows, opt_label, throttle_scale, save_path)`

Converts the polar-coordinate trajectory `(r, θ)` to Cartesian `(x, y) = (r cos θ, r sin θ)` and generates the heliocentric transfer plot.

**Inputs:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result_dict` | dict | — | Optimizer output. Required keys: `'state_history'` (N+1, 6), `'time_history_days'` (N+1,), `'alphas_rad'` (N,), `'tof_days'`, `'propellant_mass_kg'`. |
| `config` | MissionConfig | — | Mission parameters (thrust, Isp, initial mass). Used for figure annotations. |
| `n_thrust_arrows` | int | 0 | Number of thrust-direction arrows to overlay. Set to 0 to suppress arrows. Set to 20 (or via `--thrust-arrows`) for a representative sampling. |
| `opt_label` | str | `'Time-Optimal'` | Label in figure title. Pass `'Mass-Optimal'` for Week 3 Pareto trajectory plots. |
| `throttle_scale` | float | 1.0 | Arrow length multiplier. Pass the optimal throttle value (< 1.0) for mass-optimal trajectories so arrow lengths are directly comparable across plots. The title appends `\| throttle = x.xxx` when `throttle_scale < 1.0`. |
| `save_path` | str | None | File path for saving the figure. Directory is created automatically if it does not exist. |

**Returns:** `matplotlib.figure.Figure`

**Visual elements:**
- Earth's orbit: blue dashed circle at r = 1 AU
- Mars's orbit: red dashed circle at r = 1.52368 AU
- Sun: gold star at the origin
- Departure point: blue filled circle (zorder 6)
- Arrival point: red filled square (zorder 7)
- Transfer trajectory: black curve
- Thrust arrows (if `n_thrust_arrows > 0`): dark-orange quiver arrows overlaid at evenly-spaced arc-length positions

##### `_draw_thrust_arrows(ax, state_history, alphas_rad, n_arrows)` (private)

Places thrust-direction arrows along the trajectory at positions evenly spaced by **arc length** (cumulative chord length along the Cartesian path), not uniformly in time or segment index.

**Algorithm:**
1. Convert polar history to Cartesian: `x_k = r_k cos θ_k`, `y_k = r_k sin θ_k`
2. Compute cumulative chord length: `arc[k] = Σ |Δ(x,y)|` up to node k
3. Sample n_arrows arc-length values uniformly from 1% to 99% of the total arc to avoid crowding the departure and arrival markers
4. For each sample, find the nearest trajectory node
5. Look up the piecewise-constant steering angle for that segment: `seg = min(node_index, N-1)`
6. Compute the Cartesian thrust direction from the steering angle α and polar angle θ:
   ```
   dir_x = sin(α) cos(θ) − cos(α) sin(θ)   [radial_x − transverse_x]
   dir_y = sin(α) sin(θ) + cos(α) cos(θ)   [radial_y + transverse_y]
   ```
7. Set arrow length = `max(0.70 × r_span, 0.40)` AU (constant across all arrows, representing the constant thrust force magnitude)
8. Draw all arrows with `ax.quiver(..., angles='xy', scale_units='xy', scale=1)` at zorder 5

**Why arc-length spacing?** Time or segment spacing produces dense clusters wherever the spacecraft moves slowly (early in the transfer) and sparse regions at the end. Arc-length spacing gives a visually uniform sampling that matches the reader's intuitive expectation for "evenly spaced along the path."

**Why constant arrow length?** For time-optimal trajectories the throttle is always 1 (full thrust), so the thrust force magnitude `F = T` is constant throughout. Using thrust force (not thrust acceleration `T/m`) gives equal-length arrows — visually consistent with the fact that the engine output does not change.

##### `plot_state_history(result_dict, config, save_path)`

Time histories of all 6 state variables (r, u, v, θ, mp, acc_dv) as separate subplots, with physical-unit labels and grid lines.

#### porkchop_plot.py

A **pork chop plot** is a two-dimensional contour chart that maps every (departure date, arrival date) pair to the mission's characteristic energy C3 and arrival ΔV. The distinctive pork-chop shape arises naturally from the structure of Lambert's solutions near a launch window. This module computes those grids and renders a two-panel figure.

##### `compute_porkchop_data(departure_jd_range, arrival_jd_range, n_departure, n_arrival)` → dict

Pre-computes the C3 and arrival ΔV for every point on a uniform 2-D grid by solving Lambert's problem at each (departure JD, arrival JD) pair.

**Inputs:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `departure_jd_range` | list[float, float] | — | [JD_min, JD_max] for departure dates. |
| `arrival_jd_range` | list[float, float] | — | [JD_min, JD_max] for arrival dates. |
| `n_departure` | int | 50 | Number of departure-date grid points. |
| `n_arrival` | int | 50 | Number of arrival-date grid points. |

**Returns:** dict with keys:

| Key | Shape | Description |
|---|---|---|
| `'departure_jd'` | (n_departure,) | Julian Dates of departure axis. |
| `'arrival_jd'` | (n_arrival,) | Julian Dates of arrival axis. |
| `'C3_depart'` | (n_arrival, n_departure) | Departure characteristic energy [km²/s²]. Values exceeding 200 km²/s² are set to NaN. |
| `'dv_arrive'` | (n_arrival, n_departure) | Arrival ΔV [km/s]. Values exceeding 15 km/s are set to NaN. |
| `'tof_days'` | (n_arrival, n_departure) | Transfer time of flight [days]. |
| `'departure_dates_str'` | list[str] | Calendar strings for axis labels (e.g., `'2026-Feb-14'`). |
| `'arrival_dates_str'` | list[str] | Calendar strings for arrival axis labels. |

**Method:**
- Planet positions use the same low-precision Meeus polynomial longitudes as `ephemeris/planetary_states.py`, referenced to J2000.0.
- Each Lambert call is wrapped in `try/except` so a non-converging or short-TOF (< 50 days) pair produces a NaN rather than crashing.
- C3 and ΔV are clipped to NaN beyond the module-level thresholds (`_C3_CLIP = 200.0 km²/s²`, `_DV_CLIP = 15.0 km/s`) to keep contour levels from compressing the interesting low-C3 region.
- The result dict is also usable for any downstream analysis independently of the plotting function.

**Performance note:** For a 75×75 grid, 5625 Lambert solves are performed. Each solve is a few milliseconds of Newton-Raphson iteration, so the total computation is typically 5–30 seconds.

**Usage:**
```python
from visualization.porkchop_plot import compute_porkchop_data, plot_porkchop
from ephemeris.planetary_states import compute_departure_date

jd_dep = compute_departure_date(...)
dep_range = [jd_dep - 182.5, jd_dep + 182.5]   # ±6 months around optimal departure
arr_range = [dep_range[0] + 50.0, dep_range[1] + 500.0]

pc_data = compute_porkchop_data(dep_range, arr_range, n_departure=75, n_arrival=75)
```

##### `plot_porkchop(porkchop_data, optimal_result, save_path)` → plt.Figure

Renders the two-panel pork chop figure.

**Inputs:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `porkchop_data` | dict | — | Output of `compute_porkchop_data`. |
| `optimal_result` | dict or None | None | When provided, must contain `'departure_jd'` and `'arrival_jd'` (floats). Triggers C3 normalization (see below) and marks the optimal point with a white star. |
| `save_path` | str or None | None | File path to save the figure (PNG, PDF, etc.). |

**Left panel — Characteristic Energy (C3):**

When `optimal_result` is *not* provided: raw C3 contours in km²/s², filled from 0 to 100 km²/s², iso-levels at [5, 10, 15, 20, 30, 40, 60, 80] km²/s².

When `optimal_result` *is* provided (the default in `run_week2` and `plot_week2`): the panel shows **normalized C3**, computed as:
```
C3_normalized[i,j] = C3_raw[i,j] / C3_opt
```
where `C3_opt` is evaluated by a single Lambert solve at the exact `(departure_jd, arrival_jd)` pair from `optimal_result`. Iso-levels are [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0] (dimensionless multiples of the optimal mission's C3). The colorbar label reads `'C3 / C3_opt  [-]  (C3_opt = X.X km²/s²)'` so the absolute value of C3_opt is still visible.

The contour at level 1.0 (white, thicker, labeled) marks the set of all (departure, arrival) pairs that require exactly the same departure energy as the time-optimal solution — providing an immediate visual reference for the size of the acceptable launch window.

**Right panel — Arrival ΔV:** Raw arrival ΔV in km/s, iso-levels at [2, 3, 4, 5, 6, 8, 10, 12] km/s. The ΔV is computed as `|v_lambert_arrive − v_mars_circular|` — the speed difference that must be eliminated at Mars to achieve a circular capture orbit.

**Both panels** share:
- Constant time-of-flight diagonal lines (white dashed) at intervals selected automatically
- The optimal mission point marked with a white star (size 15) and a black edge, annotated `'Optimal'`
- Calendar-date tick labels on both axes, formatted as `'YYYY-MMM-DD'`
- Departure dates on the x-axis (rotated 45°), arrival dates on the y-axis

**Usage:**
```python
optimal_point = {'departure_jd': jd_dep, 'arrival_jd': jd_arr}
fig = plot_porkchop(pc_data, optimal_result=optimal_point,
                    save_path='results/week2_porkchop_plot.png')
```

##### `_compute_c3_at_point(jd_dep, jd_arr)` → float or None (private)

Performs a single Lambert solve at `(jd_dep, jd_arr)` and returns the departure C3 in km²/s². Returns `None` if the transfer time is less than 50 days or if the Lambert solver does not converge. Used internally by `plot_porkchop` to find `C3_opt` at the optimal departure/arrival pair.

##### Helper functions (private)

- `_jd_ticks(jd_array, step_days)`: Returns J2000.0-aligned tick positions at multiples of `step_days` (default 90 days), with a fallback to [first, last] if no multiples fall within the range.
- `_apply_date_ticks(ax, dep_jd, arr_jd, step_days)`: Sets formatted calendar-date tick labels on both axes.
- `_draw_tof_lines(ax, dep_jd, arr_jd, tof_levels)`: Draws white dashed diagonals of constant transfer time. Each diagonal satisfies `arrival_jd = departure_jd + T`; the visible segment is clipped to the plot bounds and labeled at 65% along the line.
- `_mark_optimal(ax, optimal_result)`: Places the white star marker and annotation.

#### control_plot.py

- `plot_steering_history(res, config)`: Piecewise-constant staircase plot of the steering angle α over the transfer. This shows when the spacecraft thrusts prograde (α≈0), retrograde (α≈±π), or out-of-plane.
- `plot_mass_history(res, config)`: Spacecraft mass over time, decreasing monotonically as propellant is consumed.

#### pareto_plot.py

- `plot_pareto_frontier(pareto_points, selected_idx)`: Transfer time vs. propellant fraction curve. When `selected_idx` is provided, the selected Pareto point is highlighted with an orange diamond marker and annotated with its TOF and throttle value — making it easy to see which point was chosen for the trajectory visualization.
- `plot_power_sweep(sweep_points)`: Specific power (W/kg) vs. transfer time curve.

---

### 5.16 main.py

**Purpose:** The top-level pipeline driver that ties everything together into a 4-week milestone structure with a command-line interface.

#### Command-Line Interface

```bash
python main.py --week 1                      # Week 1 only: Hohmann baseline
python main.py --week 2                      # Week 2: Time-optimal (N=400, may take ~10-30 min)
python main.py --week 3                      # Week 3: Mass-optimal + Pareto + power sweep
python main.py --week 4                      # Week 4: 3D nonplanar extension
python main.py --week all                    # All weeks in sequence
python main.py --week 2 --quick              # Fast demo: N=20 segments
python main.py --week all --save             # Save results as .pkl files for later plotting
python main.py --week 3 --pareto-idx 7       # Week 3: use Pareto point at index 7 for trajectory plot
python main.py --week 3 --pareto-idx -1      # Week 3: use last (most mass-efficient) Pareto point
python main.py --week 2 --thrust-arrows 30   # Week 2: overlay 30 thrust-direction arrows on trajectory
python main.py --week 2 --thrust-arrows 0    # Week 2: suppress thrust arrows entirely
```

**`--thrust-arrows N`**: Number of thrust-direction arrows to overlay on the Week 2 (and Week 3) trajectory plot. Arrows are placed at arc-length-evenly-spaced positions along the trajectory and point in the direction of the thrust vector. Arrow length is proportional to thrust force (constant for time-optimal full-throttle, scaled by the optimal throttle for mass-optimal). Default is 20. Pass 0 to suppress arrows entirely.

**`--pareto-idx IDX`**: Selects which converged Pareto frontier point to use for the Week 3 mass-optimal trajectory visualization. The index is 0-based into the list sorted by `tof_days` (ascending). Negative indices count from the end: `-1` selects the longest TOF (lowest throttle, most mass-efficient) point, which provides the strongest visual contrast with the Week 2 time-optimal plot. Default is `-1`.

#### Week Runners

- `run_week1(args)`: Computes Hohmann baseline, validates ND state vectors, prints a comparison table.
- `run_week2(args)`: Solves time-optimal problem, generates trajectory and control plots.
- `run_week3(args)`: Solves mass-optimal, generates Pareto frontier and power sweep curves, and generates a trajectory plot (`week3_pareto_trajectory.png`) for the Pareto point selected with `--pareto-idx`.
- `run_week4(args)`: Solves 3D nonplanar problem with user-specified inclination and RAAN.

#### Result Pickling

The `--save` flag pickles the result dictionary to `results/weekN_result.pkl`. A future `--plot-only` run can load this pickle and regenerate plots without rerunning the optimizer — useful when the optimizer takes many minutes and you want to tweak plot formatting.

#### Parallel Execution

The code detects the number of CPU cores with `os.cpu_count()` and passes `n_jobs` to the solvers. On a machine with 8 cores, Jacobian evaluation is 8× faster, reducing a 30-minute run to about 4 minutes.

---

## 6. How to Run the Code

### 6.1 Prerequisites and Installation

**Python version:** 3.9 or later recommended.

**Install dependencies:**
```bash
cd low_thrust_optimization
pip install -r requirements.txt
```

The `requirements.txt` includes:
- `numpy` — numerical arrays, linear algebra
- `scipy` — ODE integration (`solve_ivp`) and optimization (`minimize`)
- `matplotlib` — all figures
- `astropy` — precise planetary constants and time conversions

**Verify installation:**
```bash
python -c "import numpy, scipy, matplotlib, astropy; print('All packages OK')"
```

### 6.2 Running Each Week's Milestone

All commands should be run from the `low_thrust_optimization/` directory.

**Week 1 — Hohmann Baseline (runs in seconds):**
```bash
python main.py --week 1
```
Expected output: Transfer time ≈ 259 days, ΔV ≈ 5.59 km/s, propellant ≈ 476 kg.

**Week 2 — Time-Optimal (N=400, may take 10–30 minutes):**
```bash
python main.py --week 2
```
Or fast demo:
```bash
python main.py --week 2 --quick
```
With thrust arrows and a saved pork chop plot:
```bash
python main.py --week 2 --thrust-arrows 20 --save
```
Expected output: Transfer time ≈ 200 days, propellant ≈ 350–450 kg. Generates `results/week2_trajectory.png`, `results/week2_porkchop_plot.png`, and control/mass-history plots.

**Week 3 — Mass-Optimal + Pareto (may take 30–60 minutes for full Pareto sweep):**
```bash
python main.py --week 3                  # default: trajectory for last Pareto point (--pareto-idx -1)
python main.py --week 3 --pareto-idx 7   # trajectory for the 8th Pareto point (index 7)
```
Expected output: Mass-optimal throttle < 1.0 for TOF=200 days. Pareto curve showing trade-off. Trajectory plot `week3_pareto_trajectory.png` for the selected point.

**Week 4 — 3D Nonplanar:**
```bash
python main.py --week 4
```
Expected output: 3D trajectory plot with inclination change.

**All weeks:**
```bash
python main.py --week all --save
```

### 6.3 Quick Mode for Fast Testing

The `--quick` flag sets N=20 segments (vs. the full N=400). This makes each solver call about (400/20)² = 400 times cheaper in Jacobian computation and typically runs in under 1 minute. The solutions will be less accurate (the piecewise-constant approximation is coarser) but demonstrate all features of the code.

**Individual module self-tests:**
Each module has a `if __name__ == '__main__':` block with self-tests:
```bash
python core/constants.py
python core/nondimensional.py
python core/equations_of_motion.py
python propagator/integrator.py
python optimization/shooting.py
```

---

## 7. Expected Results and How to Interpret Them

### 7.1 Week 1: The Hohmann Baseline

**Console output example:**
```
WEEK 1  —  Hohmann Impulsive Baseline
  -- Hohmann transfer  (Earth -> Mars, coplanar circular orbits)
     Transfer SMA                   1.26184 AU
     Transfer time                  258.85 days
     Departure burn dV1             2.945 km/s
     Arrival burn dV2               2.648 km/s
     Total delta-V                  5.593 km/s
     Propellant consumed            476.2 kg  (9.5%)
     Final mass                     4523.8 kg
     Mass ratio m0/mf               1.1054
```

**How to interpret:**
- The **transfer SMA** of 1.26184 AU is the average of Earth (1.0 AU) and Mars (1.52368 AU) distances — the semi-major axis of the transfer ellipse.
- The **259-day transfer** is the absolute minimum time for a two-impulse Keplerian transfer (no thrust between burns). Low-thrust transfers will take longer.
- The **5.59 km/s total ΔV** is what you need to change from Earth's orbit to the transfer ellipse, then from the transfer ellipse to Mars's orbit.
- **9.5% propellant fraction** seems low because Isp = 3000 s is very high for a chemical rocket perspective. Real impulsive missions use Isp ≈ 450 s for chemical propulsion, which would require about 70% propellant for the same ΔV.

**Non-dimensional state validation:**
```
y0 (Earth): [1.000, 0.000, 1.000, 0.000, 0.000, 0.000]
              r=1AU  u=0    v=1ND  θ=0    mp=0   dv=0
yf (Mars):  [1.52368, 0.000, 0.8101, NaN, NaN, NaN]
```

### 7.2 Week 2: Time-Optimal Transfer

**Console output example:**
```
WEEK 2  —  Time-Optimal Low-Thrust Transfer
  -- Optimizer output
     Converged                      YES
     Wall time                      847.3 s
     N iterations                   24

  -- Trajectory summary
     Transfer time                  200.14 days
     Propellant consumed            348.7 kg  (7.0%)
     Final (delivered) mass         4651.3 kg
     Estimated departure            2026-Feb-14 08:00
     Estimated arrival              2026-Sep-01 12:00
```

**How to interpret:**

- **Transfer time ≈ 200 days** — this is the minimum achievable with this thruster, about 23% less time than the Hohmann transfer (259 days). The low-thrust spacecraft spirals outward more efficiently, using continuous thrust to gradually raise the orbit.
- **Propellant ≈ 348 kg (7.0%)** — slightly less propellant than the Hohmann baseline (9.5%) even though the transfer is faster! This counterintuitive result occurs because the low-thrust SEP engine has Isp = 3000 s, far higher than what a chemical engine could achieve for the same ΔV maneuver. The key insight: low-thrust transfers don't need more ΔV — they need continuous thrust time. With high Isp, the same ΔV requires proportionally less propellant.
- **N=24 iterations** with the warm-start (vs. ~80-100 from cold) — demonstrates the effectiveness of the coarse-to-fine strategy.

**Trajectory plot interpretation:**
- The transfer path spirals outward from Earth's orbit to Mars's orbit, not following the elliptical arc of a Hohmann transfer.
- The spacecraft does not go to the "other side" of the Sun — it spirals around less than 180° typically.
- Near the start and end, the steering angle tends to be nearly prograde (α ≈ 0), with some adjustments for radial vs. tangential targeting.

**Steering history plot interpretation:**
- A staircase plot showing α for each of the N segments.
- Near-zero α (prograde) dominates early and late in the transfer.
- Larger positive or negative α values indicate when radial adjustment is needed.
- The steering is NOT smooth — this is the piecewise-constant approximation. A higher N gives a "smoother" effective steering law.

**Thrust arrow overlay interpretation:**
- 20 dark-orange arrows are overlaid on the trajectory, each pointing in the direction of the thrust vector at that arc-length position.
- All arrows are the same length because the thrust force magnitude is constant (full throttle throughout).
- Arrows pointing nearly tangent to the trajectory (prograde) are more efficient for raising the orbit; arrows pointing outward or inward indicate radial thrusting needed to shape the transfer arc.
- Arc-length spacing ensures the arrows are visually uniform along the path regardless of speed variation.

**State history plots:**
- `r`: monotonically increases from 1.0 AU to 1.52368 AU (in general — may be non-monotonic for some optimal solutions)
- `v`: starts at ~1.0 ND (Earth's speed), ends at ~0.81 ND (Mars's circular speed)
- `u`: starts at 0, fluctuates during transfer, returns to 0 at arrival (circular orbit)
- `mp`: starts at 0, increases monotonically (propellant consumed)
- `acc_dv`: starts at 0, increases monotonically (ΔV accumulated)

**Pork chop plot interpretation (`week2_porkchop_plot.png`):**

The Week 2 pork chop plot is automatically generated alongside the trajectory plot. It shows the entire launch-window landscape surrounding the optimal solution.

*Left panel — Normalized C3 (departure energy):*

The left panel contours show C3 / C3_opt, where C3_opt is the characteristic energy computed via Lambert's problem at the exact (departure JD, arrival JD) of the time-optimal solution.

- **The white contour at level 1.0** is the most important reference: it outlines all (departure date, arrival date) pairs that require the same launch energy as the time-optimal mission. The area enclosed by this contour is your "launch window" — any departure/arrival pair inside it needs no more C3 than the optimal point.
- **Contours at 0.5, 0.75** indicate pairs that are *cheaper* to launch than the time-optimal point (lower C3). These exist because the time-optimal mission minimizes transfer *time*, not launch energy — there may be longer transfers with lower launch energy.
- **Contours at 1.25, 1.5, 2.0, 3.0, 5.0** show how rapidly launch energy grows as you depart earlier/later or target a faster/slower arrival.
- The **colorbar label** reads `C3 / C3_opt  [-]  (C3_opt = X.X km²/s²)` so the absolute C3 of the reference mission is always visible.

*Right panel — Arrival ΔV:*

Shows the speed difference `|v_lambert_arrive − v_Mars_circular|` that must be cancelled to circularize at Mars.

- Low ΔV (dark colors, ≈ 2–3 km/s) occurs near the pork-chop minimum and for longer, more efficient transfers.
- High ΔV (bright/yellow, > 8 km/s) occurs for short, fast transfers that arrive at a steep angle.

*White diagonal lines:*

Constant time-of-flight (TOF) diagonals run from lower-left to upper-right. Each line satisfies `arrival_JD = departure_JD + T`. Reading from the TOF lines, you can immediately see which part of the optimal contour corresponds to transfers shorter or longer than the time-optimal solution.

*White star:*

The white star with black edge marks the (departure date, arrival date) pair from the time-optimal solution. It lies at the reference point C3/C3_opt = 1.0 by construction. Its position on the TOF diagonals confirms the transfer time.

**How to use the pork chop plot for mission planning:**
1. Locate the white star (optimal solution). Read off its TOF from the diagonal lines.
2. The shape of the C3 = 1.0 contour tells you how much flexibility you have: a wide, roughly circular contour indicates a forgiving launch window; a narrow, elongated contour means the window is tight.
3. If a departure slip of ±2 weeks is needed (e.g., due to launch vehicle availability), check the C3 level at the shifted departure date and the same arrival date — this tells you whether the launch vehicle needs more margin.
4. Compare left and right panels: a point with low normalized C3 but high arrival ΔV may be a good departure but require a heavier propulsion system at Mars.

### 7.3 Week 3: Mass-Optimal and Pareto Analysis

**Mass-optimal output example:**
```
Mass-optimal (fixed TOF = 200 days):
  Optimal throttle: 0.973
  Propellant: 341.2 kg  (6.8%)
  Final mass: 4658.8 kg
```

**Interpretation:** At the time-optimal transfer time (200 days), the mass-optimal throttle is very close to 1.0 (full throttle), meaning there is almost no room to reduce propellant at this short transfer time. The transfer is already nearly at minimum time — lowering the throttle would violate the arrival constraints.

**Pareto frontier interpretation:**
The graph of propellant fraction vs. transfer time forms a **Pareto curve**:
- The **left end** (short TOF ≈ 200 days): high throttle ≈ 1.0, propellant ≈ 7%
- The **right end** (long TOF ≈ 400+ days): low throttle, propellant ≈ 4–5%
- **The Hohmann baseline** appears as a reference point: TOF ≈ 259 days, propellant ≈ 9.5%

Note that all low-thrust points on the Pareto curve use less propellant than the Hohmann baseline for comparable transfer times. This is the SEP advantage: higher Isp makes each m/s of ΔV cheaper in propellant.

Any point above the Pareto curve is suboptimal (using more propellant than necessary for that time). Any point below the curve is physically infeasible (violates the physics). The curve itself represents the best achievable performance.

**Pareto trajectory plot (`week3_pareto_trajectory.png`):**

Week 3 generates a trajectory visualization for one user-selected Pareto point (chosen with `--pareto-idx`), directly analogous to the Week 2 time-optimal trajectory plot. The two figures can be placed side-by-side to compare the time-optimal and mass-optimal transfer geometries:

| Feature | Week 2 (time-optimal) | Week 3 (mass-optimal, selected Pareto point) |
|---|---|---|
| Figure file | `week2_trajectory.png` | `week3_pareto_trajectory.png` |
| Title | "Earth-Mars Time-Optimal Transfer" | "Earth-Mars Mass-Optimal Transfer \| throttle = x.xxx" |
| Throttle | 1.0 (full power throughout) | < 1.0 (lower for longer TOF) |
| Transfer time | ~200 days (minimum feasible) | Longer (set by selected Pareto point) |
| Thrust arrows | Full length | Scaled by throttle — shorter arrows indicate weaker thrust |

**How to read the thrust arrows:** Arrow length is proportional to thrust magnitude: `arrow_length = base_scale × throttle`. A mass-optimal solution at throttle = 0.6 will have arrows that are 60% as long as the Week 2 arrows. The arrow direction shows the steering angle α at that location along the trajectory; comparing the two plots reveals how the engine is pointed differently in time-optimal vs. mass-optimal modes.

**What to look for:**
- Mass-optimal trajectories often follow a wider, more gradual arc than time-optimal ones, because the spacecraft is allowed more time to coast on the gravity field rather than thrusting continuously.
- The steering angle history (direction of arrows) may differ significantly — time-optimal trajectories tend to thrust nearly prograde throughout, while mass-optimal trajectories can employ more radial thrusting to exploit gravity turns.
- The Pareto frontier plot highlights the selected point with an orange diamond so you can immediately read off the TOF and throttle associated with the trajectory being visualized.

**Power sweep interpretation:**
The graph of specific power (W/kg) vs. transfer time shows:
- Higher power (stronger thruster) → shorter transfer time
- The relationship is roughly: `TOF ∝ 1/thrust`
- Very high power → diminishing returns (transfer time can't go below ~100 days regardless of thrust, due to orbital mechanics geometry)
- Very low power → very long transfers become impractical (years)

The "correct" operating point depends on the mission's cost function: how do you value 1 kg of power system mass against 1 day of transfer time?

### 7.4 Week 4: 3D Nonplanar Extension

The 3D extension handles the fact that Earth's and Mars's orbital planes are not perfectly aligned (Mars has ~1.85° inclination relative to Earth's ecliptic). More importantly, it demonstrates the capability to design missions to inclined target orbits (e.g., a polar Mars orbit at i=90°).

**Expected behavior:**
- For low target inclinations (i < 5°), the 3D solution is nearly identical to the 2D solution — the out-of-plane steering angles β are nearly zero.
- For higher inclinations (i = 30°–90°), significant out-of-plane thrusting is required, increasing the transfer time and propellant compared to the coplanar case.
- The 3D trajectory plot shows a path that spirals out of the ecliptic plane to reach the inclined target orbit.

**Constraint satisfaction:**
The five terminal constraints should all be satisfied to |cᵢ| < 1e-6:
```
c1 (radius):        |r_f/r_Mars - 1| < 1e-6
c2 (zero rad vel):  |dot(r_f,v_f)/(r2*vc2)| < 1e-6
c3 (circ speed):    |(|v_f|-vc2)/vc2| < 1e-6
c4 (inclination):   |h_fz/|h_f| - cos(i2)| < 1e-6
c5 (RAAN):          |(h_fx*cos(Ω) + h_fy*sin(Ω))/|h_f|| < 1e-6
```

---

## 8. Key Design Decisions and Engineering Trade-offs

### N Segments: Resolution vs. Cost

The number of shooting segments N is the primary trade-off parameter:

| N | Jacobian cost per iteration | Approximation quality |
|---|---|---|
| 20 (quick mode) | 21 propagations | Coarse, fast demo |
| 25 (warm-start coarse) | 26 propagations | Good enough for initialization |
| 200 | 201 propagations | Good |
| 400 (default) | 401 propagations | Excellent |

The optimal N is where further refinement no longer changes the objective or constraint residuals meaningfully. For this Earth-Mars problem, N=200–400 gives well-converged solutions.

### Integration Tolerance vs. Optimization Tolerance

The ODE integration tolerance (1e-8 to 1e-10) is set much tighter than the SLSQP constraint tolerance (1e-6). This hierarchy is intentional:
- **ODE tolerance** controls how accurately the trajectory is propagated — should be tight enough that integration errors don't contaminate the finite-difference Jacobian.
- **SLSQP tolerance** controls when the optimizer declares convergence.

If the ODE tolerance were looser than the SLSQP tolerance, the solver would oscillate because small perturbations in the decision vector would be swamped by integration noise.

The finite-difference step `ε = max(1e-5, sqrt(ODE_tol))` is chosen by the classical formula for the optimal FD step: the step that balances truncation error (too large) against cancellation error (too small). For ODE tolerance 1e-8, `ε ≈ 1e-4`.

### Sequential vs. Parallel Pareto Sweep

**Sequential with warm-starting:** Better when cores are limited or the problems are closely related. Exploits the fact that adjacent TOF values yield similar optimal steering laws.

**Parallel without warm-starting:** Better when many cores are available and problems are independent. Scales linearly with cores.

The code supports both modes via the `n_jobs` parameter.

### Polar vs. Cartesian Coordinates

The 2D solver uses polar coordinates because:
- Natural for circular orbit problems (Earth and Mars are both on nearly circular orbits)
- Circular velocity appears directly as `v = 1.0` (ND) — clean terminal condition
- Centrifugal and Coriolis terms are explicit, which helps physical interpretation

The 3D solver uses Cartesian coordinates because:
- Generalizes naturally to arbitrary inclinations
- No coordinate singularity at the poles (unlike spherical coordinates)
- Angular momentum and RAAN constraints are simpler to write

---

## 9. Glossary

| Term | Definition |
|---|---|
| **Arc-length spacing** | Placing sample points at positions evenly spaced by cumulative path length along a curve, rather than uniformly in time or parameter. Used for thrust arrow placement to give visually uniform distribution regardless of speed variations along the trajectory. |
| **Astronomical Unit (AU)** | Earth-Sun mean distance, ≈ 149.6 million km. Used as the distance unit throughout this code. |
| **C3 (Characteristic Energy)** | `C3 = |v_∞|²` — the square of the hyperbolic excess speed at departure. Equals `|v_lambert_depart − v_Earth_circular|²` in km²/s². C3 = 0 means the spacecraft just barely escapes Earth's sphere of influence; higher C3 means the launcher imparts additional speed beyond the minimum escape energy. Launch vehicles are rated by the mass they can deliver to a given C3. |
| **C3 normalization** | Dividing the raw C3 grid by the C3 of the reference mission (the time-optimal solution) to produce a dimensionless ratio. A normalized C3 of 1.0 corresponds exactly to the reference mission's launch energy; values < 1 are cheaper launches; values > 1 require more energy. |
| **Centrifugal acceleration** | In rotating/polar coordinates, an apparent outward acceleration `v²/r` arising from the coordinate frame. Not a real force. |
| **Coriolis acceleration** | In rotating/polar coordinates, a coupling term `-uv/r` between radial and tangential velocities. Not a real force. |
| **Decision vector** | The finite set of numbers that the NLP optimizer adjusts: transfer time and steering angles. |
| **Delta-V (ΔV)** | Change in velocity — the "cost" of a trajectory maneuver in velocity units. A higher ΔV requires more propellant (Tsiolkovsky equation). |
| **DOP853** | Dormand-Prince 8th-order Runge-Kutta ODE integrator. High-order method with adaptive step control. |
| **Ephemeris** | A table or model of planetary positions as a function of time. |
| **Equality constraint** | A condition that must be exactly satisfied (e.g., arrive at Mars's exact radius). Handled as `c(x) = 0` in the NLP. |
| **Inclination** | The angle between an orbit's plane and the reference plane (ecliptic). Earth's orbit has i ≈ 0°; Mars has i ≈ 1.85°. |
| **Isp (Specific Impulse)** | Thrust per unit weight-flow-rate of propellant. Higher = more efficient engine. Measured in seconds. |
| **Julian Date** | Continuous count of days since January 1, 4713 BC noon. J2000.0 = January 1, 2000 noon = JD 2451545.0. |
| **Lambert's Problem** | Given two position vectors and a transfer time, find the connecting orbit. The classical orbit determination problem. |
| **Low-thrust** | Propulsion mode where thrust is small but continuous over long periods (months). SEP is the primary low-thrust technology. |
| **Mass-optimal** | Objective of minimizing propellant consumption (maximizing final mass) for a fixed transfer time. |
| **NLP (Nonlinear Program)** | Optimization problem with nonlinear objective and/or constraints. Solved here with SLSQP. |
| **Non-dimensional** | Scaled so that key reference quantities (AU, t_cf) equal 1. Improves numerical conditioning in solvers. |
| **Pareto frontier** | The curve of optimal trade-offs between two objectives (here: time vs. propellant). No point on the curve dominates another. |
| **Pork chop plot** | A two-dimensional contour chart with departure date on one axis and arrival date on the other, showing iso-contours of C3 (or arrival ΔV). The distinctive pork-chop shape emerges from the structure of Lambert solutions near a launch window. Used in mission design to identify and evaluate the launch window. |
| **Polar coordinates** | Position described by (r, θ): radial distance and angle, rather than (x, y) Cartesian. |
| **Piecewise-constant control** | The control variable (steering angle) is held constant within each segment and may change between segments. |
| **RAAN (Right Ascension of Ascending Node)** | The angle (measured from the vernal equinox) at which the orbit crosses the ecliptic plane going north. Defines the orbital plane's orientation. |
| **SEP (Solar Electric Propulsion)** | Ion or Hall-effect thruster powered by solar panels. High Isp (1500–10000 s) but low thrust (mN to N). |
| **Single shooting** | Trajectory optimization method where the EOM are integrated forward from departure, and boundary conditions at arrival are enforced as constraints. |
| **SLSQP** | Sequential Least Squares Programming. A gradient-based NLP solver in `scipy.optimize.minimize`. |
| **Specific angular momentum (h)** | h = r × v. Constant for unthrusted Keplerian orbits. Its direction defines the orbit plane; its magnitude determines the orbit size. |
| **Steering angle (α)** | The direction of the thrust vector, measured from the transverse (prograde) direction. α=0 → prograde thrust. |
| **Thrust arrow overlay** | Dark-orange quiver arrows plotted on the trajectory figure, positioned at arc-length-evenly-spaced points and directed by the piecewise-constant steering angle at that segment. All arrows are the same length (constant thrust force for full-throttle time-optimal missions); for mass-optimal trajectories the length is scaled by the optimal throttle. |
| **Synodic period** | Time between successive launch windows for Earth-Mars missions: ≈ 780 days (≈ 2.14 years). |
| **t_cf (time conversion factor)** | `t_cf = sqrt(1/μ_sun)` ≈ 58.13 days. The ND time unit. |
| **Terminal constraints** | Conditions the trajectory must satisfy at arrival: correct radius, zero radial velocity, correct circular speed. |
| **Time-optimal** | Objective of minimizing total transfer time. Spacecraft thrusts at full throttle throughout. |
| **Throttle** | Fraction of maximum thrust used: 0 = coasting, 1 = full thrust. |
| **Transfer orbit** | The intermediate orbit followed by the spacecraft during transit from departure orbit to arrival orbit. |
| **Tsiolkovsky equation** | `ΔV = Isp × g₀ × ln(m₀/m_f)`. Relates propellant consumption to speed change. |
| **Vis-viva equation** | `v² = μ(2/r - 1/a)`. Gives orbital speed at any point in an elliptic orbit. |
| **Warm start** | Using the solution from a nearby (coarser or previous) problem as the initial guess for the current problem. Dramatically reduces iteration count. |

---

*This report covers the complete `low_thrust_optimization` codebase as implemented for AEROSP 548, Spring 2026. All equations are presented in both dimensional and non-dimensional form consistent with the code implementation. The four-week milestone structure progresses from analytical baselines through 2D numerical optimization to full 3D nonplanar trajectory design.*

---

## 10. Low-Thrust Spiral Descent in Mars Orbit

### 10.1 Motivation and Physical Setup

After a spacecraft completes an interplanetary transfer and is captured into Mars orbit, it typically arrives in a large elliptical or high-circular parking orbit. Mission objectives — surface relay, science mapping, or atmospheric entry — often require a much lower circular orbit. Descending chemically (via a single large retrograde burn) is propellant-expensive. A low-thrust SEP engine, by contrast, can accomplish the same radius reduction with far less propellant by applying a small continuous retrograde force over many revolutions.

The maneuver studied here is a **constant-inclination circular-to-circular spiral descent**: the spacecraft lowers its orbit radius from an initial value `r₁` to a target value `r₂ < r₁` while keeping the orbital plane fixed (constant inclination `i` and right ascension of ascending node Ω). Because no plane change is required, the optimal thrust direction is purely **tangential and retrograde** — no radial or out-of-plane component.

The demonstration scenario uses:

| Parameter | Value |
|---|---|
| Initial orbit radius `r₁` | 5000 km (altitude 1610.5 km) |
| Target orbit radius `r₂` | 4000 km (altitude 610.5 km) |
| Orbit inclination `i` | 30° (constant) |
| SEP thrust `T` | 2 N |
| Initial spacecraft mass `m₀` | 500 kg |
| Specific impulse `Isp` | 3000 s |
| Mars gravitational parameter `μ_Mars` | 4.282837 × 10⁴ km³/s² |

---

### 10.2 State Representation: Modified Equinoctial Elements

The trajectory is propagated using **Modified Equinoctial Elements (MEE)**, a set of six orbital elements introduced by Walker, Ireland, and Owens (1985) that remain well-defined for all inclinations and for circular orbits (unlike classical elements, which become singular at `e = 0` or `i = 0`).

The MEE state vector is:

```
q = [p,  f,  g,  h,  k,  L]
```

where:

| Element | Definition | Physical meaning |
|---|---|---|
| `p` | a(1 − e²) | Semi-latus rectum [km] |
| `f` | e·cos(ω + Ω) | Eccentricity x-component |
| `g` | e·sin(ω + Ω) | Eccentricity y-component |
| `h` | tan(i/2)·cos(Ω) | Inclination x-component |
| `k` | tan(i/2)·sin(Ω) | Inclination y-component |
| `L` | Ω + ω + ν | True longitude [rad] |

Here `a` is the semi-major axis, `e` the eccentricity, `i` the inclination, `Ω` the right ascension of the ascending node (RAAN), `ω` the argument of periapsis, and `ν` the true anomaly.

For the **initial circular orbit** at radius `r₁` with inclination `i` and RAAN `Ω`:

```
p₀ = r₁          (circular: a = r₁, e = 0 → p = r₁)
f₀ = 0, g₀ = 0   (zero eccentricity)
h₀ = tan(i/2)·cos(Ω)
k₀ = tan(i/2)·sin(Ω)
L₀ = Ω            (start at the ascending node: ω = ν = 0)
```

The instantaneous orbital radius at true longitude `L` is:

```
r = p / w,   where  w = 1 + f·cos(L) + g·sin(L)
```

For the near-circular spirals studied here, `f ≈ g ≈ 0` throughout, so `w ≈ 1` and `r ≈ p` at every point on the orbit.

The full state vector propagated in code is extended to eight elements:

```
y = [p,  f,  g,  h,  k,  L,  m_p_frac,  ΔV_acc]
```

where `m_p_frac` is the accumulated propellant mass fraction (starts at 0, increases as fuel is burned) and `ΔV_acc` is the accumulated velocity increment (integral of thrust acceleration magnitude over time).

---

### 10.3 Gauss Variational Equations for Tangential Thrust

The **Gauss Variational Equations (GVE)** describe how perturbing accelerations change the orbital elements over time. In the MEE formulation (Walker 1985, Table 1), the perturbation is decomposed into three components in the local orbital frame:

- `F_R`: radial (outward along the position vector)
- `F_T`: tangential (along-track, in the direction of motion)
- `F_N`: normal (out-of-plane, completing the right-handed frame)

For the **constant-inclination spiral descent**, only `F_T ≠ 0`:

```
F_R = 0      (no radial thrust)
F_T = -a_max (constant full retrograde thrust; negative = opposing velocity)
F_N = 0      (no out-of-plane thrust; inclination is preserved)
```

Substituting into the MEE GVE with `μ* = 1` (non-dimensional) gives:

```
ṗ = 2p/w · √p · F_T

ḟ = √p · [(w+1)cosL + f] / w · F_T

ġ = √p · [(w+1)sinL + g] / w · F_T

ḣ = 0     (no F_N → inclination components h, k are constant)

k̇ = 0

L̇ = w² / p^(3/2)     (Keplerian mean motion; no F_T correction)
```

The `ḣ = k̇ = 0` equations are the mathematical statement that **the orbital plane is frozen** throughout the maneuver. Because `h` and `k` encode the inclination and RAAN, both remain exactly constant as long as `F_N = 0`.

**Physical interpretation of the ṗ equation:**

```
ṗ = 2p/w · √p · F_T ≈ 2p^(3/2) · F_T   (for w ≈ 1, near-circular)
```

Since `F_T < 0` (retrograde), `ṗ < 0`: the semi-latus rectum — and thus the semi-major axis — decreases monotonically. This is the mathematical origin of the inward spiral.

**Physical interpretation of the ḟ, ġ equations:**

The tangential perturbation excites eccentricity oscillations (`f` and `g` oscillate at orbital frequency), but their secular (orbit-averaged) rate is zero for a circular orbit. The orbit therefore remains approximately circular throughout the maneuver; the instantaneous eccentricity stays below `e < 0.01` in the demonstration case.

**Propellant mass flow:**

The spacecraft mass decreases as propellant is expelled. The Tsiolkovsky rocket equation in differential form gives:

```
dm/dt = -T / (g₀ · Isp)
```

In terms of mass fraction `m_p_frac = 1 − m/m₀`:

```
ṁ_p_frac = T / (m₀ · g₀ · Isp)   [1/s]
```

As mass decreases, the thrust acceleration `a_T = T/m` increases, so the later revolutions descend slightly faster than the earlier ones. The code models this correctly by computing the effective acceleration at each integration step:

```
F_T_eff = F_T_nominal / (1 − m_p_frac)
```

---

### 10.4 Time-Optimal Control Law

#### Why Constant Retrograde Thrust is Optimal

For a minimum-time circular-to-circular orbit lowering with fixed thrust magnitude, **Pontryagin's Maximum Principle** (Pontryagin et al., 1962) requires maximizing the Hamiltonian at every instant:

```
H = λ^T · f(q, u) − 1
```

where `λ` is the costate (primer vector) and `f` is the right-hand side of the GVE. For a fixed thrust magnitude `|u| = a_max`, this maximum is achieved when `u` is aligned with the primer vector projected onto the thrust-direction space.

For a near-circular transfer with no inclination change, the primer vector analysis (Lawden, 1963; Edelbaum, 1965) shows:

1. **Radial component F_R**: contributes to `ṗ` only at second order; the primer vector component along `F_R` is zero on average. Optimal `F_R = 0`.
2. **Normal component F_N**: changes `h` and `k`, moving the orbit plane. Since the inclination must remain constant, the primer vector component along `F_N` is zero. Optimal `F_N = 0`.
3. **Tangential component F_T**: the primer vector is aligned opposite to the velocity vector (retrograde) throughout the maneuver. Optimal `F_T = −a_max` (constant full retrograde).

This result — constant full retrograde thrust — is the **optimal singular arc** for circular-to-circular low-thrust descent. It is exact for the orbit-averaged dynamics and highly accurate for the full (non-averaged) dynamics when the thrust-to-gravity ratio is small (the low-thrust assumption), as is the case here:

```
ε ≡ a_T / (μ/r²) = (4×10⁻⁶ km/s²) / (42828/5000² km/s²)
                  = 4×10⁻⁶ / 1.713×10⁻³
                  ≈ 2.3×10⁻³   ≪ 1   ✓
```

The small parameter `ε ≈ 0.0023` confirms the low-thrust regime; the spacecraft advances roughly 360/ε ≈ 155,000 degrees of true longitude per unit change in `p`, corresponding to many revolutions per unit radius change. This justifies the orbit-averaging approximation underlying the Edelbaum formula.

---

### 10.5 Edelbaum Analytical Solution for Circular Spirals

#### Instantaneous Circular Speed

For a circular orbit of radius `a = r` around Mars:

```
v_c(r) = √(μ_Mars / r)
```

The circular speed increases as `r` decreases — counter-intuitive, but a fundamental consequence of orbital mechanics. A spacecraft in a lower orbit must move faster to maintain centripetal balance.

#### Time Evolution of Circular Speed

From the GVE for `ṗ` (orbit-averaged over one revolution, using `⟨w⟩ = 1` and `⟨cosL⟩ = ⟨sinL⟩ = 0`):

```
⟨ṗ⟩ ≈ 2p^(3/2) · F_T   (non-dimensional)
```

In dimensional form, for `p = a = r` (circular orbit):

```
ṙ = 2r · F_T / v_c(r) = -2r · a_T / v_c(r)
```

Differentiating `v_c = √(μ/r)` with respect to time:

```
v̇_c = d/dt √(μ/r) = -√μ / (2r^(3/2)) · ṙ
     = -√μ / (2r^(3/2)) · (-2r·a_T/v_c)
     = √μ · a_T / (r^(3/2) · v_c)
     = μ · a_T / (r · v_c²) · v_c / √μ · √μ / v_c
     = a_T
```

The circular speed evolves linearly in time:

```
v_c(t) = v_c(r₁) + a_T · t
```

where `a_T > 0` (the speed increases as the orbit lowers). This is the key result: **the circular speed grows at exactly the thrust acceleration rate**, independent of the current orbit radius.

#### Transfer Time and Total ΔV

The spacecraft reaches `r₂` when `v_c = v_c(r₂)`:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   T_f = [ v_c(r₂) − v_c(r₁) ] / a_T                          │
│                                                                 │
│   ΔV   = a_T × T_f = √(μ_Mars/r₂) − √(μ_Mars/r₁)             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

This is **Edelbaum's formula** for the velocity increment of a constant-inclination low-thrust circular-to-circular transfer. It is exact in the orbit-averaged sense and requires only the initial and final circular speeds.

For the demonstration scenario:
```
v_c(r₁) = √(42828.37/5000) = 2.9271 km/s
v_c(r₂) = √(42828.37/4000) = 3.2720 km/s
ΔV_Edelbaum = 3.2720 − 2.9271 = 0.3449 km/s
T_f = 0.3449 / (4×10⁻⁶) = 86,225 s = 0.998 days
```

#### Number of Revolutions

The instantaneous mean motion is `n(t) = v_c(t)³ / μ` (from `n = √(μ/a³)` and `v_c³ = μ^(3/2)/a^(3/2)` → `a³ = μ³/v_c⁶` → `n = v_c³/μ`). The total number of revolutions is:

```
N_rev = (1/2π) ∫₀^T_f n(t) dt
       = (1/2π) ∫₀^T_f v_c(t)³/μ dt
       = (1/2πμ) ∫₀^T_f (v_c1 + a_T t)³ dt
       = [v_c(r₂)⁴ − v_c(r₁)⁴] / (8π · μ_Mars · a_T)
```

For the demonstration:
```
N_rev = (3.2720⁴ − 2.9271⁴) / (8π × 42828.37 × 4×10⁻⁶)
       = (114.40 − 73.37) / 4.304
       ≈ 9.5 revolutions
```

#### Propellant Consumption

The Tsiolkovsky rocket equation gives the propellant mass fraction:

```
Δm_p / m₀ = 1 − exp(−ΔV / v_e)
```

where the effective exhaust velocity is `v_e = Isp × g₀`. For Isp = 3000 s:

```
v_e = 3000 × 9.80665×10⁻³ = 29.42 km/s
Δm_p / m₀ = 1 − exp(−0.3449/29.42) = 1 − exp(−0.01173) = 0.01166
Δm_p = 500 × 0.01166 = 5.83 kg
```

The propellant consumption is only **1.17% of the initial spacecraft mass** — a testament to the extraordinary efficiency of SEP (`Isp = 3000 s` vs. ~450 s for chemical engines).

---

### 10.6 Mars-Centric Non-Dimensional Scaling

The code uses a Mars-centric non-dimensionalization anchored at the **initial orbit radius** `r_ref = r₁`:

| Quantity | Reference | Non-Dimensional Unit |
|---|---|---|
| Length `L*` | `r₁` [km] | `r₁` |
| Time `T*` | `√(r₁³/μ_Mars)` [s] | ≈ 1222.6 s for r₁ = 5000 km |
| Velocity `V*` | `√(μ_Mars/r₁)` [km/s] | ≈ 2.927 km/s for r₁ = 5000 km |
| Acceleration `A*` | `μ_Mars/r₁²` [km/s²] | ≈ 1.713×10⁻³ km/s² |

With this scaling, the initial circular orbit is at `p₀ = 1.0` (ND), and the target orbit is at `p_target = r₂/r₁ = 4000/5000 = 0.80` (ND). The nominal thrust acceleration is:

```
a_T_ND = (T/m₀ × 10⁻³ km/m) / A* = (4×10⁻⁶ km/s²) / (1.713×10⁻³ km/s²) ≈ 2.34×10⁻³ [ND]
```

The small value of `a_T_ND ≪ 1` confirms the low-thrust regime and validates the orbit-averaging approximation.

**Propagation terminates** when the non-dimensional semi-latus rectum first satisfies `p ≤ r₂/r₁ = 0.80`, at which point the spacecraft has reached the target circular orbit radius.

---

### 10.7 Interpretation of the Demonstration Trajectory

#### Overview of Results

The demonstration trajectory (`visualization/mars_spiral_descent_3d.py`, standalone demo) propagates a 2 N SEP spacecraft from a 5000 km circular Mars orbit to a 4000 km circular orbit over 9 revolutions. The numerical integration (DOP853, tolerance 10⁻⁹) yields:

| Quantity | Analytical prediction | Simulation result |
|---|---|---|
| Transfer time `T_f` | 0.998 days | **0.994 days** |
| Total ΔV | 0.3449 km/s | **0.3455 km/s** |
| Propellant mass `Δm_p` | 5.83 kg (1.17%) | **5.84 kg (1.17%)** |
| Approximate revolutions | 9.5 | **~9 revolutions** |
| Final orbit radius | 4000.0 km | **3999.6 km** |

The excellent agreement (errors < 0.2%) between the Edelbaum analytical predictions and the full numerical integration validates both the physics model and the GVE implementation.

#### 3D Trajectory Figure (`results/mars_spiral_descent_3d.png`)

The 3D figure produced by `plot_mars_spiral_descent_3d()` shows the following features:

**The inward spiral.** The black trajectory coils inward around Mars over approximately 9 revolutions. Because the orbit inclination is held constant at 30°, all loops lie in the same inclined plane — the spiral resembles a coil spring viewed at an angle. The innermost loop (arrival) is clearly smaller than the outermost (departure). Black arrowheads placed at 12 evenly-spaced points along the trajectory confirm that the spacecraft travels in the prograde direction (counter-clockwise when viewed from the north orbital pole) throughout the maneuver.

**Initial and target orbit circles.** The blue dashed circle at `r₁ = 5000 km` and the red dashed circle at `r₂ = 4000 km` both lie in the same 30°-inclined plane, confirming that the orbital plane is preserved. The radius ratio `r₁/r₂ = 1.25` is clearly visible — the outer circle is 25% larger in radius, which corresponds to a 56% larger orbital area.

**Radius decrease per revolution.** The total radius change is 1000 km over ~9.5 revolutions, giving an average of about **105 km per revolution**. This is consistent with the orbit-averaged rate:

```
⟨ṙ⟩ = -2r·a_T/v_c(r)  at r = r₁:
      = -2 × 5000 × 4×10⁻⁶ / 2.9271
      = -1.367×10⁻² km/s
      = -0.0137 km/s × (T_orb at r₁) per revolution
```

The orbital period at `r₁ = 5000 km`:

```
T_orb(r₁) = 2π × √(r₁³/μ_Mars) = 2π × 1222.6 s = 7681 s ≈ 2.13 hours
```

So the average descent per revolution: `0.0137 km/s × 7681 s ≈ 105 km/rev` — exactly as observed.

**Intermediate orbit circles.** Four dotted grey circles at evenly-spaced times show the orbital radius at 20%, 40%, 60%, and 80% of the maneuver duration. Their spacing decreases slightly toward the end (the later circles are closer together) because the orbital period shrinks as `r` decreases and the orbit becomes faster.

**Mars sphere and reference grid.** The translucent Martian red sphere (radius 3389.5 km) occupies the center of the figure, with the grey equatorial reference grid in the z = 0 (equatorial) plane. The spacecraft at `r₁ = 5000 km` is at an altitude of 1610 km — more than 0.47 Mars radii above the surface. At `r₂ = 4000 km`, the altitude is 611 km, comfortably above both the nominal atmosphere interface (~80 km) and the aerocapture threshold.

**Departure and arrival markers.** The blue dot (departure) and red dot (arrival) are both located at the ascending node of the 30°-inclined orbit (where the orbit crosses the equatorial plane going north), which is the initial condition of the propagation (`L₀ = Ω = 0`).

#### Time History Figure (`results/mars_spiral_descent_history.png`)

The 2-panel time history from `plot_spiral_descent_history()` shows:

**Top panel — Orbital radius `r(t)`.** The radius decreases monotonically from 5000 km to 3999.6 km over 0.994 days. The decrease is not perfectly linear: it proceeds as `r(t) ≈ μ / v_c(t)² = μ / (v_c1 + a_T t)²`, a **hyperbolic function of time**. The curve is slightly concave upward — the descent rate in km per unit time slows slightly as the orbit shrinks, because the reduced circumference means fewer kilometers of altitude change per unit arc length.

**Bottom panel — Altitude above surface.** The altitude tracks `r(t) − R_Mars`, descending from 1610.5 km to 609.6 km. The shaded orange band below 80 km marks the Mars atmospheric interface (the aerocapture threshold). The final altitude of 610 km is comfortably above this boundary; the spacecraft is in no danger of atmospheric interaction. A trajectory designer could continue the descent to lower altitudes if needed, subject to propellant availability.

#### Engineering Significance

The 1.17% propellant expenditure for a 1000 km altitude decrease represents an extraordinary figure-of-merit for electric propulsion. For comparison, a single retrograde chemical burn using a bipropellant engine (Isp ≈ 450 s) would require:

```
Δm_p_chem / m₀ = 1 − exp(−ΔV / (Isp_chem × g₀))
                = 1 − exp(−0.3449 / 4.414)
                ≈ 7.5%
```

The low-thrust SEP system uses **6.4× less propellant** for the same orbit change. The cost is time: 0.994 days of continuous engine operation at 2 N, compared to a single chemical burn lasting approximately:

```
Δt_chem = Δm_p_chem × v_e_chem / T_chem ≈ (37.5 kg × 4.414 km/s) / (500 N × 10⁻³ km/N·s²·s)
```

In practice, chemical descent burns last on the order of minutes. The SEP mission designer accepts a ~1 day maneuver time in exchange for a factor-of-six reduction in propellant, which directly translates to either a smaller launch vehicle or more margin for other mission activities.

---

## 11. Low-Thrust Inclination Change in Mars Orbit

### 11.1 Motivation and Physical Setup

Orbital inclination — the tilt of the orbital plane relative to a reference plane — is among the most expensive orbital elements to change in astrodynamics. For an impulsive engine, a pure inclination change at a circular orbit of radius `r` and speed `v_c` requires a Δv of:

```
Δv_imp = 2 v_c sin(Δi / 2)
```

At a circular Mars parking orbit of radius `r = 3789.5 km` (altitude ~400 km), the circular speed is:

```
v_c = √(μ_Mars / r) = √(42828 / 3789.5) ≈ 3.361 km/s
```

A 20° inclination change therefore requires `Δv_imp = 2 × 3.361 × sin(10°) ≈ 1.167 km/s` — comparable to the Δv required for a major interplanetary maneuver. For a spacecraft operating with electric propulsion (e.g., Isp = 3000 s), the mass penalty is modest, but the thrust level is low, so the maneuver must be executed over many orbits. Understanding how the spacecraft slowly tilts its orbital plane over tens of revolutions is the subject of this section.

The physical picture is straightforward: an out-of-plane (normal) thrust force rotates the orbital angular momentum vector. Applying positive normal thrust (perpendicular to the orbital plane, in the direction of the angular momentum vector) above the equator and negative below produces a net torque that tips the orbital plane. The spacecraft spirals in inclination space, tracing out a family of progressively inclined circular orbits around Mars.

### 11.2 State Representation: Modified Equinoctial Elements

The propagation uses the **Modified Equinoctial Elements (MEE)**, a non-singular set of six orbital elements introduced to avoid the numerical singularities that appear in classical Keplerian elements at zero eccentricity or zero inclination. The state vector is:

```
y = [p, f, g, h, k, L]
```

where:

| Symbol | Definition | Physical meaning |
|--------|-----------|-----------------|
| `p`    | semi-latus rectum (km) | orbit "size"; equals `a(1−e²)` |
| `f`    | `e cos(ω + Ω)` | x-component of eccentricity vector |
| `g`    | `e sin(ω + Ω)` | y-component of eccentricity vector |
| `h`    | `tan(i/2) cos(Ω)` | x-component of inclination vector |
| `k`    | `tan(i/2) sin(Ω)` | y-component of inclination vector |
| `L`    | `Ω + ω + ν` | true longitude (fast angle) |

The inclination and right ascension of the ascending node (RAAN) are recovered from `h` and `k` via:

```
i   = 2 arctan(√(h² + k²))
Ω   = arctan2(k, h)
```

For a nearly circular orbit, `f ≈ 0`, `g ≈ 0`, and `p ≈ r`. The inclination state is entirely captured by the two slow elements `h` and `k`. The true longitude `L` is the fast variable, completing one cycle per orbital revolution.

A spacecraft starting at `(i₁, Ω₁) = (0°, 0°)` with a target `(i₂, Ω₂) = (20°, 0°)` has initial MEE:

```
p₀ = r = 3789.5 km,  f₀ = 0,  g₀ = 0
h₀ = tan(0°/2) cos(0°) = 0
k₀ = tan(0°/2) sin(0°) = 0
L₀ = 0  (ascending node, true longitude = Ω = 0)
```

and target values:

```
h_f = tan(20°/2) cos(0°) = tan(10°) ≈ 0.17633
k_f = tan(20°/2) sin(0°) = 0
```

The mass `m` and accumulated propellant `Δm` are tracked as additional state variables (total state dimension = 8 including mass and Δm).

### 11.3 Gauss Variational Equations for Out-of-Plane Thrust

The **Gauss Variational Equations (GVE)** in MEE form (Walker, Ireland & Owens 1985) describe how each orbital element changes under an arbitrary perturbing force expressed in the **radial–transverse–normal (RTN)** frame:

```
F = F_R r̂ + F_T t̂ + F_N n̂
```

The full equations are (using the notation of Walker 1985, equations A5–A13):

```
ṗ = (2p / w) √(p/μ) · F_T

ḟ = √(p/μ) [ F_R sin L  +  F_T ((w+1)cos L + f)/w  −  F_N g(h sin L − k cos L)/w ]

ġ = √(p/μ) [−F_R cos L  +  F_T ((w+1)sin L + g)/w  +  F_N f(h sin L − k cos L)/w ]

ḣ = √(p/μ)  (s²/(2w)) F_N cos L

k̇ = √(p/μ)  (s²/(2w)) F_N sin L

L̇ = √(μ p) (w/p)²  +  (1/w) √(p/μ) (h sin L − k cos L) F_N
```

where:

```
w = 1 + f cos L + g sin L      (inverse radial factor)
s² = 1 + h² + k²              (inclination factor)
```

For a **pure inclination change** the strategy is to apply thrust exclusively in the out-of-plane direction: `F_R = 0`, `F_T = 0`, `F_N ≠ 0`. Under this restriction the equations simplify dramatically. Setting `F_R = F_T = 0`:

```
ṗ = 0                          (semi-latus rectum is constant → circular orbit preserved)

ḟ = −√(p/μ)  F_N g (h sin L − k cos L) / w

ġ = +√(p/μ)  F_N f (h sin L − k cos L) / w

ḣ = √(p/μ)  (s²/2w) F_N cos L

k̇ = √(p/μ)  (s²/2w) F_N sin L

L̇ = √(μ p) (w/p)²  +  (1/w) √(p/μ) (h sin L − k cos L) F_N
```

For a nearly circular orbit (`f ≈ 0`, `g ≈ 0`) the `ḟ` and `ġ` terms vanish, leaving only:

```
ḣ ≈ √(p/μ)  (s²/2w) F_N cos L
k̇ ≈ √(p/μ)  (s²/2w) F_N sin L
L̇ ≈ √(μ/p³)
```

The inclination change is driven entirely by the `ḣ` and `k̇` equations, which are proportional to `F_N cos L` and `F_N sin L` respectively. To drive `h` from 0 to `h_f = tan(i₂/2)` with `k = 0` throughout, the control must primarily excite `ḣ` (the cosine-weighted component).

### 11.4 Bang-Bang Normal Thrust Control Law

Because the thrust magnitude is fixed (the SEP thruster cannot be throttled in this model), the only degree of freedom is the thrust direction: `F_N = ±F_max`. The optimal control law for this problem follows from Pontryagin's minimum principle applied to the switching function:

```
σ(L) = ∂H/∂F_N  ∝  (dh_f/dF_N) · λ_h  +  (dk_f/dF_N) · λ_k
     =  (s²/2w) √(p/μ) [ λ_h cos L  +  λ_k sin L ]
```

where `λ_h`, `λ_k` are the costates conjugate to `h` and `k`. For a single-plane maneuver (`k_f = k₀ = 0`, so `λ_k ≈ 0` and `Ω = 0`), the switching function reduces to:

```
σ(L) ∝ cos L
```

This produces the **bang-bang control law**:

```
         ⎧ +F_max    if cos L > 0   (spacecraft in "north" half of orbit)
F_N(L) = ⎨
         ⎩ −F_max    if cos L < 0   (spacecraft in "south" half of orbit)
```

The sign reversal occurs at the nodal crossings `L = π/2` and `L = 3π/2` (where cos L = 0). Physically: the thruster fires "upward" (in the +n̂ direction) while the spacecraft is ascending through the northern hemisphere and "downward" while it is descending through the southern hemisphere. Both halves of each orbit contribute constructively to tilting the orbital plane.

For a general maneuver where `h` and `k` must both change, the switching function generalizes to:

```
         ⎧ +F_max    if  (dh_f · cos L + dk_f · sin L) > 0
F_N(L) = ⎨
         ⎩ −F_max    otherwise
```

where `dh_f = h_f − h(t)` and `dk_f = k_f − k(t)` are the current defects in `h` and `k`.

### 11.5 Time-of-Flight Estimate and Impulsive Baseline

**Orbit-averaged inclination rate.** Over one full orbit, the orbit-averaged rate of change of the inclination vector magnitude `|h| = tan(i/2)` due to the bang-bang control is:

```
⟨dh/dt⟩ = (1/2π) ∫₀²π  (s²/2w) √(p/μ) · F_N(L) · cos L  dL
```

With `F_N(L) = F_max · sign(cos L)` and the nearly circular approximation (`w ≈ 1`, `s² ≈ 1 + h² + k²`):

```
⟨dh/dt⟩ ≈ (s²/2) √(p/μ) F_max · (1/2π) ∫₀²π |cos L| dL
         = (s²/2) √(p/μ) F_max · (2/π)
```

The factor `2/π` is the average of `|cos L|` over a full orbit — this is the **bang-bang efficiency factor**. The maximum possible orbit-averaged rate (achieved only if the entire thrust is applied at the two nodes, which is not possible with a continuous thruster) would have factor 1; the bang-bang strategy achieves 63.7% of that maximum.

**Effective acceleration.** Define the thrust acceleration `a_eff = F / m`. For the demonstration case:

```
F = 3.5 N,  m = 500 kg  →  a_eff = 7.0 × 10⁻³ m/s² = 7.0 × 10⁻⁶ km/s²
```

**Impulsive Δv equivalent.** The Δv for a pure inclination change at constant circular radius is:

```
Δv_imp = 2 v_c sin(Δi / 2)
```

where `Δi = i₂ − i₁`. For the demonstration:

```
v_c = √(μ_Mars / r) = √(42828 / 3789.5) = 3.361 km/s
Δv_imp = 2 × 3.361 × sin(10°) = 2 × 3.361 × 0.17365 = 1.167 km/s
```

**Time-of-flight estimate.** Combining the bang-bang efficiency with the effective acceleration:

```
TOF ≈ (π/2) · Δv_imp / a_eff
    = (π/2) × 1.167 km/s / (7.0 × 10⁻⁶ km/s²)
    = 2.62 × 10⁵ s
    ≈ 3.03 days
```

The factor `π/2` accounts for the bang-bang efficiency: the continuous thrust at angle L produces only `2/π` of the maximum possible orbit-averaged Δv per unit time, so the maneuver takes `π/2` longer than a hypothetically perfectly phased impulsive sequence.

**Number of revolutions.** At `r = 3789.5 km`, the orbital period is:

```
T_orb = 2π √(r³/μ_Mars) = 2π × √(3789.5³ / 42828) = 2π × 596.9 s = 3751 s ≈ 1.042 hr
```

The total number of revolutions:

```
N_rev = TOF / T_orb = 2.62 × 10⁵ / 3751 ≈ 70 revolutions
```

**Propellant consumption.** The mass flow rate for an Isp = 3000 s engine:

```
ṁ = F / (g₀ · Isp) = 3.5 / (9.80665 × 3000) = 1.190 × 10⁻⁴ kg/s
```

Over 3.03 days (2.62 × 10⁵ s):

```
Δm_p = ṁ · TOF = 1.190 × 10⁻⁴ × 2.62 × 10⁵ ≈ 31.2 kg
```

Propellant fraction: `Δm_p / m₀ = 31.2 / 500 = 6.24%`. The final spacecraft mass is approximately 468.8 kg.

### 11.6 Mars-Centric Non-Dimensional Scaling

To improve numerical conditioning of the ODE integrator, all quantities are non-dimensionalized relative to the reference orbit:

```
L* = r_ref   (reference length, chosen as the initial orbit radius in km)
T* = √(r_ref³ / μ_Mars)   (reference time in seconds)
V* = √(μ_Mars / r_ref)    (reference velocity = circular speed at r_ref)
A* = μ_Mars / r_ref²      (reference acceleration)
```

For the demonstration (`r_ref = 3789.5 km`, `μ_Mars = 42828 km³/s²`):

```
L* = 3789.5 km
T* = √(3789.5³ / 42828) = 596.9 s
V* = 3.361 km/s
A* = 42828 / 3789.5² = 2.983 × 10⁻³ km/s²
```

In non-dimensional units, the initial orbit has `p_nd = 1`, `μ_nd = 1`, and the thrust acceleration is:

```
a_nd = a_eff / A* = 7.0 × 10⁻⁶ / 2.983 × 10⁻³ = 2.347 × 10⁻³
```

This small number correctly indicates that the thrust is a weak perturbation relative to gravity — the spacecraft completes many revolutions per unit thrust change.

### 11.7 Interpretation of the Demonstration Trajectory

The demonstration trajectory uses the following parameters:

| Parameter | Value |
|-----------|-------|
| Orbit radius `r` | 3789.5 km (altitude ~400 km) |
| Initial inclination `i₁` | 0° (equatorial) |
| Final inclination `i₂` | 20° |
| Thrust `F` | 3.5 N |
| Spacecraft mass `m₀` | 500 kg |
| Specific impulse `Isp` | 3000 s |
| Integration segments `N` | 600 |

#### 3D Trajectory Figure

The 3D visualization (`results/mars_inclination_3d.png`) shows a sequence of instantaneous orbit circles drawn at evenly spaced revolution snapshots. Each ring represents the complete circular orbit that the spacecraft would trace if thrust were suddenly turned off at that moment. The rings evolve from a flat equatorial circle (blue, bottom) to a tilted 20°-inclined circle (red, top), stacking concentrically around the translucent Mars sphere.

**Ring spacing and tilt progression.** The rings are nearly evenly spaced in inclination because the bang-bang control law produces an approximately constant orbit-averaged inclination rate (the cosine-weighted integral is approximately constant when `h` and `k` are small compared to 1, which holds for inclinations up to ~40°). Each displayed ring therefore represents roughly the same Δi increment, resulting in a visually uniform "fan" of orbital planes.

**Nodal drift.** For a purely equatorial-to-inclined maneuver with initial `Ω = 0°`, the control law drives `h` exclusively (since `dk = 0`). The switching function `σ = dh·cos L + dk·sin L = dh·cos L` means that thrust is applied symmetrically about the x-axis of the orbit plane; there is no systematic RAAN drift. The rings all share a common ascending node at `Ω ≈ 0°`, appearing as a fan of planes pivoting about the Earth-Mars x-axis in the figure.

**Inclination history.** The 2-panel history plot (`results/mars_inclination_history.png`) confirms the theoretical predictions. The top panel shows inclination increasing monotonically from 0° to ~20° over approximately 3 days, with a nearly linear trend. The slight S-shape (faster in the middle, slower at the very start and end) reflects the `sin(i/2)` nonlinearity: at small inclinations the `h` rate is approximately linear in inclination, but the actual GVE rates have mild angular dependence. The bottom panel shows RAAN remaining near 0° throughout, consistent with the symmetric bang-bang law.

**Final inclination accuracy.** The bang-bang TOF estimate predicts maneuver completion at `TOF = (π/2) · Δv_imp / a_eff`. Without a margin factor, the ODE propagation reaches the final epoch at `i_f ≈ 20.1°` — within 0.5% of the 20° target. The small overshoot arises because the bang-bang switching function is not precisely optimal for the nonlinear MEE GVE (it is the heuristic approximation of the true Pontryagin co-state direction).

**Propellant consumption.** The mass history decreases from 500 kg to approximately 468–469 kg, a consumption of ~31–32 kg (6.2–6.4% of initial mass). This is consistent with the analytical estimate from Section 11.5. Compared to a chemical propulsion system (Isp ≈ 450 s), which would consume:

```
Δm_p_chem / m₀ = 1 − exp(−Δv_imp / (Isp_chem · g₀))
               = 1 − exp(−1.167 / (450 × 9.80665 × 10⁻³))
               = 1 − exp(−0.2645)
               ≈ 23.2%
```

the SEP system uses **3.7× less propellant** for the same inclination change. The trade-off, as always, is time: the chemical system executes two burns (at the ascending and descending nodes) lasting on the order of minutes; the SEP system requires approximately 3 days of continuous firing.

**Engineering significance.** The inclination change is particularly expensive for any propulsion system precisely because it rotates the angular momentum vector rather than scaling it. The efficiency of the bang-bang strategy comes from applying thrust perpendicular to the orbital plane in a coordinated fashion across many orbits, accumulating small rotations into the required 20° tilt. The 2/π bang-bang factor represents the fundamental limit of a bang-bang (full on/off) control strategy; a continuously throttled engine with the optimal thrust profile (which applies thrust near the nodes only) could in principle approach 100% efficiency, but at the cost of larger instantaneous thrust requirements.

---
