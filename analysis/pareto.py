"""Pareto frontier generation for Earth-Mars low-thrust trajectory trade studies.

Implements two sweep analyses that trace the design trade space described in
Morante et al. 2021 (Figure 5):

1. generate_pareto_frontier
   Sweeps fixed transfer times and solves the mass-optimal problem at each
   point.  Traces the Pareto curve of propellant fraction vs. transfer time.
   Longer transfers allow the spacecraft to reach Mars with less thrust,
   reducing propellant at the cost of mission duration.

2. generate_power_sweep
   Sweeps SEP thruster power levels (encoded as thrust magnitude) and solves
   the time-optimal problem at each point.  Traces the curve of specific
   power vs. minimum transfer time.  More powerful thrusters shorten the
   transfer but require a heavier power system.

Speed knobs
-----------
integ_tol : float (default 1e-8)
    ODE integration tolerance passed to propagate_trajectory.  1e-8 gives
    ~2-3x fewer RK45 steps than the default 1e-10, at no meaningful cost in
    optimality since the solver tolerance is 1e-6.

warm_start : bool (default True, sequential mode only)
    In sequential mode (n_jobs=1), use the previous point's optimal angles
    and throttle as the initial guess for the next point.  Reduces SLSQP
    iterations from ~90 (cold) to ~20-40 (warm) for adjacent TOF values.

n_jobs : int (default 1)
    Number of parallel worker processes.  When n_jobs > 1, the sweep points
    are distributed across workers using ProcessPoolExecutor; warm-starting
    is automatically disabled.  Use n_jobs = os.cpu_count() for maximum
    parallelism.

Reference
---------
Morante, D., Sanjurjo Rivo, M., and Soler, M. (2021). A Survey on
Low-Thrust Trajectory Optimization Approaches. Aerospace, 8(3), 88.
https://doi.org/10.3390/aerospace8030088

Units
-----
All internal computation is non-dimensional; results are returned in
dimensional units (days, kg, W/kg) for direct use by visualisation code.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from typing import List

import numpy as np

from config.mission_config import MissionConfig
from core.constants import g0
from optimization.mass_optimal import solve_mass_optimal
from optimization.time_optimal import solve_time_optimal

# Thruster efficiency assumed for specific-power calculation
_ETA_THRUSTER: float = 0.65


# ---------------------------------------------------------------------------
# Module-level worker functions (must be picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _pareto_worker(task):
    """Compute one mass-optimal Pareto point.

    Parameters (packed as a single tuple for ProcessPoolExecutor)
    -------------------------------------------------------------
    k          : int   — point index (for ordering results)
    tof        : float — fixed transfer time [days]
    base_config: MissionConfig — base spacecraft configuration
    integ_tol  : float — ODE integration tolerance
    maxiter    : int   — SLSQP iteration cap
    """
    k, tof, base_config, integ_tol, maxiter = task
    cfg = replace(
        base_config,
        opt_mode         = 'mass_optimal',
        time_guess_days  = tof,
        time_lb_days     = tof,
        time_ub_days     = tof,
        throttle_guess   = 0.5,
        throttle_lb      = 0.0,
        throttle_ub      = 1.0,
    )
    try:
        res = solve_mass_optimal(cfg, integ_tol=integ_tol, maxiter=maxiter)
        return k, tof, res, None
    except Exception as exc:
        return k, tof, None, str(exc)


def _sweep_worker(task):
    """Compute one time-optimal power sweep point.

    Parameters (packed as a single tuple for ProcessPoolExecutor)
    -------------------------------------------------------------
    k          : int   — point index (for ordering results)
    T          : float — thrust magnitude [N]
    base_config: MissionConfig — base spacecraft configuration
    tof_lb     : float — per-thrust TOF lower bound [days]
    tof_ub     : float — per-thrust TOF upper bound [days]
    tof_guess  : float — per-thrust TOF initial guess [days]
    integ_tol  : float — ODE integration tolerance
    maxiter    : int   — SLSQP iteration cap
    """
    k, T, base_config, tof_lb, tof_ub, tof_guess, integ_tol, maxiter = task
    cfg = replace(
        base_config,
        thrust_N         = T,
        opt_mode         = 'time_optimal',
        time_guess_days  = tof_guess,
        time_lb_days     = tof_lb,
        time_ub_days     = tof_ub,
        throttle_guess   = 1.0,
        throttle_lb      = 1.0,
        throttle_ub      = 1.0,
    )
    try:
        res = solve_time_optimal(cfg, integ_tol=integ_tol, maxiter=maxiter)
        return k, T, res, None
    except Exception as exc:
        return k, T, None, str(exc)


# ---------------------------------------------------------------------------
# TOF-bound helper for power sweep
# ---------------------------------------------------------------------------

def _tof_bounds_for_thrust(T_N: float, m0_kg: float) -> tuple:
    """Estimate a sensible [tof_lb, tof_ub, tof_guess] for a given thrust.

    Uses a rough jet-power heuristic: characteristic ΔV for Earth-Mars is
    approximately 3–8 km/s.  The minimum-time bound is clamped to 90 days
    because the orbital geometry makes sub-90-day Earth-Mars transfers
    impractical regardless of thrust.

    Returns
    -------
    (tof_lb_days, tof_ub_days, tof_guess_days) : tuple of float
    """
    acc = T_N / m0_kg                              # m/s²
    tof_lb = max(90.0,   3_000.0 / acc / 86400.0)  # 3 km/s characteristic
    tof_ub = min(450.0,  8_000.0 / acc / 86400.0)  # 8 km/s characteristic
    # Ensure at least a 50-day window so the optimizer has room to search
    if tof_ub <= tof_lb:
        tof_ub = tof_lb + 50.0
    tof_guess = 0.5 * (tof_lb + tof_ub)
    return tof_lb, tof_ub, tof_guess


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_pareto_frontier(
    base_config: MissionConfig,
    tof_range_days: list,
    n_points: int = 20,
    integ_tol: float = 1e-8,
    warm_start: bool = True,
    n_jobs: int = 1,
    sweep_maxiter: int = 300,
) -> List[dict]:
    """Sweep transfer times and solve the mass-optimal problem at each point.

    For a fixed transfer time, the mass-optimal NLP finds the minimum
    uniform throttle that satisfies the Mars arrival constraints.  Repeating
    this across a range of transfer times traces the Pareto frontier of
    propellant consumption vs. transfer time.

    Parameters
    ----------
    base_config : MissionConfig
        Reference mission configuration.  Spacecraft parameters are
        inherited unchanged.  ``opt_mode``, ``time_*``, and ``throttle_*``
        are overridden for each sweep point.
    tof_range_days : list of two floats
        ``[tof_min_days, tof_max_days]`` — inclusive endpoints [days].
    n_points : int, optional
        Number of linearly spaced transfer times to evaluate (default 20).
    integ_tol : float, optional
        ODE integration tolerance (default 1e-8).  Looser than the
        standalone default (1e-10) for speed; still 100× tighter than the
        SLSQP constraint tolerance (1e-6).
    warm_start : bool, optional
        When True and n_jobs==1, seed each point's initial guess with the
        previous converged solution (default True).  Sweep proceeds from
        longest to shortest TOF (easy → hard).  Ignored when n_jobs > 1.
    n_jobs : int, optional
        Number of parallel worker processes (default 1 = sequential).
        When > 1, disables warm-starting and uses ProcessPoolExecutor.
    sweep_maxiter : int, optional
        SLSQP iteration cap for each sweep solve (default 300).  Keeps
        stuck solves from monopolising runtime — problems that haven't
        converged in 300 iterations are almost certainly in a bad basin.

    Returns
    -------
    pareto_points : list of dict
        Successful results sorted by ``tof_days``.  Each entry contains:

        'tof_days'            : float
        'propellant_fraction' : float
        'propellant_mass_kg'  : float
        'final_mass_kg'       : float
        'throttle'            : float
        'solver_message'      : str

    Notes
    -----
    The minimum feasible TOF is bounded below by the thrust-to-mass ratio.
    Very short transfers require throttle > 1.0 and will fail.
    """
    tof_min, tof_max = float(tof_range_days[0]), float(tof_range_days[1])
    if tof_min >= tof_max:
        raise ValueError(
            f"tof_range_days must be [min, max] with min < max, "
            f"got [{tof_min}, {tof_max}]"
        )
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")

    tof_values = np.linspace(tof_min, tof_max, n_points)

    mode_str = (f"parallel n_jobs={n_jobs}" if n_jobs > 1
                else ("sequential+warm-start" if warm_start else "sequential"))
    print(f"Pareto sweep: {n_points} TOF values in [{tof_min:.1f}, {tof_max:.1f}] days"
          f"  [{mode_str}, integ_tol={integ_tol:.0e}]")
    print(f"  Spacecraft: {base_config.initial_mass_kg:.0f} kg, "
          f"{base_config.thrust_N:.2f} N, "
          f"Isp={base_config.isp_s:.0f} s, "
          f"N={base_config.n_segments} segments\n")

    results = []

    if n_jobs > 1:
        # ------------------------------------------------------------------
        # Parallel mode: all points submitted simultaneously
        # ------------------------------------------------------------------
        from concurrent.futures import ProcessPoolExecutor, as_completed

        tasks = [
            (k, float(tof), base_config, integ_tol, sweep_maxiter)
            for k, tof in enumerate(tof_values)
        ]
        raw: dict = {}
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(_pareto_worker, t): t[0] for t in tasks}
            for future in as_completed(futures):
                k, tof, res, err = future.result()
                raw[k] = (tof, res, err)

        for k in sorted(raw):
            tof, res, err = raw[k]
            if err is not None:
                print(f"  [{k+1:3d}/{n_points}]  TOF={tof:7.1f} d  ERROR: {err}")
            elif not res['success']:
                print(f"  [{k+1:3d}/{n_points}]  TOF={tof:7.1f} d  "
                      f"throttle={res['throttle']:.3f}  "
                      f"FAIL ({res['solver_message']})")
            else:
                print(f"  [{k+1:3d}/{n_points}]  TOF={tof:7.1f} d  "
                      f"throttle={res['throttle']:.3f}  "
                      f"prop_frac={res['propellant_mass_kg']/base_config.initial_mass_kg:.4f}  "
                      f"OK")
                results.append({
                    'tof_days':            tof,
                    'propellant_fraction': res['propellant_mass_kg'] / base_config.initial_mass_kg,
                    'propellant_mass_kg':  res['propellant_mass_kg'],
                    'final_mass_kg':       res['final_mass_kg'],
                    'throttle':            res['throttle'],
                    'solver_message':      res['solver_message'],
                    'state_history':       res['state_history'],
                    'alphas_rad':          res['alphas_rad'],
                    'time_history_days':   res['time_history_days'],
                })

    else:
        # ------------------------------------------------------------------
        # Sequential mode: iterate high TOF → low TOF for warm-starting
        # (high TOF is easier — lower throttle, more room to maneuver)
        # ------------------------------------------------------------------
        order = list(range(n_points - 1, -1, -1))  # high → low TOF
        prev_x0 = None   # warm-start vector from previous converged solve

        for k in order:
            tof = float(tof_values[k])
            cfg = replace(
                base_config,
                opt_mode         = 'mass_optimal',
                time_guess_days  = tof,
                time_lb_days     = tof,
                time_ub_days     = tof,
                throttle_guess   = 0.5,
                throttle_lb      = 0.0,
                throttle_ub      = 1.0,
            )

            x0 = prev_x0 if (warm_start and prev_x0 is not None) else None

            try:
                res = solve_mass_optimal(cfg, x0_override=x0, integ_tol=integ_tol,
                                         maxiter=sweep_maxiter)
            except Exception as exc:
                print(f"  [{k+1:3d}/{n_points}]  TOF={tof:7.1f} d  ERROR: {exc}")
                prev_x0 = None   # reset warm start after failure
                continue

            status = "OK" if res['success'] else f"FAIL ({res['solver_message']})"
            print(
                f"  [{k+1:3d}/{n_points}]  TOF={tof:7.1f} d  "
                f"throttle={res['throttle']:.3f}  "
                f"prop_frac={res['propellant_mass_kg']/base_config.initial_mass_kg:.4f}  "
                f"{status}"
            )

            if res['success']:
                # Build x for next warm start: [throttle, alpha_1, ..., alpha_N]
                prev_x0 = np.concatenate([[res['throttle']], res['alphas_rad']])
                results.append({
                    'tof_days':            tof,
                    'propellant_fraction': res['propellant_mass_kg'] / base_config.initial_mass_kg,
                    'propellant_mass_kg':  res['propellant_mass_kg'],
                    'final_mass_kg':       res['final_mass_kg'],
                    'throttle':            res['throttle'],
                    'solver_message':      res['solver_message'],
                    'state_history':       res['state_history'],
                    'alphas_rad':          res['alphas_rad'],
                    'time_history_days':   res['time_history_days'],
                })
            else:
                prev_x0 = None   # reset warm start after failed solve

    results.sort(key=lambda r: r['tof_days'])
    print(f"\n  {len(results)}/{n_points} points converged.")
    return results


def generate_power_sweep(
    base_config: MissionConfig,
    thrust_range_N: list,
    n_points: int = 15,
    integ_tol: float = 1e-8,
    n_jobs: int = 1,
    adapt_tof_bounds: bool = True,
    sweep_maxiter: int = 300,
) -> List[dict]:
    """Sweep SEP thrust levels and solve the time-optimal problem at each point.

    Higher thrust reduces the minimum transfer time but requires more
    electrical power, implying a heavier power system.  The specific power
    (W/kg of spacecraft) captures this trade-off.

    Parameters
    ----------
    base_config : MissionConfig
        Reference mission configuration.  ``initial_mass_kg``, ``isp_s``,
        ``n_segments`` are inherited.  ``thrust_N`` and time/throttle
        settings are overridden for each sweep point.
    thrust_range_N : list of two floats
        ``[T_min, T_max]`` — inclusive thrust range [N].
    n_points : int, optional
        Number of linearly spaced thrust values to evaluate (default 15).
    integ_tol : float, optional
        ODE integration tolerance (default 1e-8).
    n_jobs : int, optional
        Number of parallel worker processes (default 1 = sequential).
    adapt_tof_bounds : bool, optional
        When True (default), compute per-thrust TOF bounds from a rough
        ΔV heuristic instead of inheriting from base_config.  This
        dramatically improves convergence for low- and high-thrust extremes
        where the base_config window [175, 225] days is unsuitable.

    Returns
    -------
    sweep_points : list of dict
        Successful results sorted by ``thrust_N`` (ascending).  Each entry:

        'thrust_N'            : float
        'specific_power_W_kg' : float
        'tof_days'            : float
        'propellant_mass_kg'  : float
        'solver_message'      : str

    Notes
    -----
    Specific power estimate (jet power formula):

        P_electric = T * Isp * g0 / (2 * eta)   [W]
        P_sp       = P_electric / m0             [W/kg]

    where eta = 0.65 is the assumed overall thruster efficiency.
    """
    T_min, T_max = float(thrust_range_N[0]), float(thrust_range_N[1])
    if T_min >= T_max:
        raise ValueError(
            f"thrust_range_N must be [min, max] with min < max, "
            f"got [{T_min}, {T_max}]"
        )
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")

    T_values = np.linspace(T_min, T_max, n_points)

    mode_str = f"parallel n_jobs={n_jobs}" if n_jobs > 1 else "sequential"
    bounds_str = "adaptive" if adapt_tof_bounds else "from config"
    print(f"Power sweep: {n_points} thrust values in [{T_min:.2f}, {T_max:.2f}] N"
          f"  [{mode_str}, integ_tol={integ_tol:.0e}, TOF bounds: {bounds_str}]")
    print(f"  Spacecraft: {base_config.initial_mass_kg:.0f} kg, "
          f"Isp={base_config.isp_s:.0f} s, "
          f"eta={_ETA_THRUSTER:.2f}, "
          f"N={base_config.n_segments} segments\n")

    # Pre-compute per-thrust TOF bounds
    if adapt_tof_bounds:
        tof_bounds = [
            _tof_bounds_for_thrust(T, base_config.initial_mass_kg)
            for T in T_values
        ]
    else:
        default = (base_config.time_lb_days,
                   base_config.time_ub_days,
                   base_config.time_guess_days)
        tof_bounds = [default] * n_points

    results = []

    if n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        tasks = [
            (k, float(T_values[k]), base_config,
             tof_bounds[k][0], tof_bounds[k][1], tof_bounds[k][2],
             integ_tol, sweep_maxiter)
            for k in range(n_points)
        ]
        raw: dict = {}
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(_sweep_worker, t): t[0] for t in tasks}
            for future in as_completed(futures):
                k, T, res, err = future.result()
                raw[k] = (T, res, err)

        for k in sorted(raw):
            T, res, err = raw[k]
            specific_power = (T * base_config.isp_s * g0) / (
                2.0 * _ETA_THRUSTER * base_config.initial_mass_kg
            )
            if err is not None:
                print(f"  [{k+1:3d}/{n_points}]  T={T:6.3f} N  "
                      f"P_sp={specific_power:7.2f} W/kg  ERROR: {err}")
            elif not res['success']:
                print(f"  [{k+1:3d}/{n_points}]  T={T:6.3f} N  "
                      f"P_sp={specific_power:7.2f} W/kg  "
                      f"TOF={res['tof_days']:7.1f} d  FAIL ({res['solver_message']})")
            else:
                print(f"  [{k+1:3d}/{n_points}]  T={T:6.3f} N  "
                      f"P_sp={specific_power:7.2f} W/kg  "
                      f"TOF={res['tof_days']:7.1f} d  OK")
                results.append({
                    'thrust_N':            T,
                    'specific_power_W_kg': specific_power,
                    'tof_days':            res['tof_days'],
                    'propellant_mass_kg':  res['propellant_mass_kg'],
                    'solver_message':      res['solver_message'],
                })

    else:
        # Sequential: high thrust → low thrust for warm-starting
        # (high thrust = shorter TOF, more constrained, converges reliably first)
        order = list(range(n_points - 1, -1, -1))  # high → low T
        prev_x0 = None

        for k in order:
            T = float(T_values[k])
            tof_lb, tof_ub, tof_guess = tof_bounds[k]
            specific_power = (T * base_config.isp_s * g0) / (
                2.0 * _ETA_THRUSTER * base_config.initial_mass_kg
            )

            cfg = replace(
                base_config,
                thrust_N         = T,
                opt_mode         = 'time_optimal',
                time_guess_days  = tof_guess,
                time_lb_days     = tof_lb,
                time_ub_days     = tof_ub,
                throttle_guess   = 1.0,
                throttle_lb      = 1.0,
                throttle_ub      = 1.0,
            )

            try:
                res = solve_time_optimal(cfg, x0_override=prev_x0,
                                         integ_tol=integ_tol,
                                         maxiter=sweep_maxiter)
            except Exception as exc:
                print(f"  [{k+1:3d}/{n_points}]  T={T:6.3f} N  "
                      f"P_sp={specific_power:7.2f} W/kg  ERROR: {exc}")
                prev_x0 = None
                continue

            status = "OK" if res['success'] else f"FAIL ({res['solver_message']})"
            print(
                f"  [{k+1:3d}/{n_points}]  T={T:6.3f} N  "
                f"P_sp={specific_power:7.2f} W/kg  "
                f"TOF={res['tof_days']:7.1f} d  "
                f"{status}"
            )

            if res['success']:
                # Warm start for next (lower-thrust) point:
                # [tof_nd, alpha_1, ..., alpha_N] — need to re-express tof_nd
                # for the new bounds; carry alphas only and let solver pick tof.
                from core.nondimensional import NonDimensional
                prev_tof_nd = NonDimensional.time_to_nd(tof_guess)
                prev_x0 = np.concatenate([[prev_tof_nd], res['alphas_rad']])

                results.append({
                    'thrust_N':            T,
                    'specific_power_W_kg': specific_power,
                    'tof_days':            res['tof_days'],
                    'propellant_mass_kg':  res['propellant_mass_kg'],
                    'solver_message':      res['solver_message'],
                })
            else:
                prev_x0 = None

    results.sort(key=lambda r: r['thrust_N'])
    print(f"\n  {len(results)}/{n_points} points converged.")
    return results


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    sys.path.insert(0, __import__('os').path.dirname(
        __import__('os').path.dirname(__import__('os').path.abspath(__file__))
    ))

    print('=== pareto.py self-test ===\n')

    # Reduced configuration: N=5 for speed, wide bounds for robustness
    base = MissionConfig(
        initial_mass_kg      = 5000.0,
        thrust_N             = 3.5,
        isp_s                = 3000.0,
        opt_mode             = 'time_optimal',   # overridden inside sweep
        n_segments           = 5,
        time_guess_days      = 200.0,
        time_lb_days         = 100.0,
        time_ub_days         = 350.0,
        throttle_guess       = 1.0,
        throttle_lb          = 1.0,
        throttle_ub          = 1.0,
        alpha_guess_deg      = 0.0,
        alpha_lb_deg         = -180.0,
        alpha_ub_deg         =  180.0,
        departure_date_guess = '2020-01-01',
    )

    # ------------------------------------------------------------------
    # 1. Input validation
    # ------------------------------------------------------------------
    print('Test 1: input validation')
    try:
        generate_pareto_frontier(base, [300.0, 200.0], n_points=2)
        assert False, 'Should have raised ValueError'
    except ValueError as e:
        print(f"  Caught bad tof_range: {e}")

    try:
        generate_power_sweep(base, [5.0, 1.0], n_points=2)
        assert False, 'Should have raised ValueError'
    except ValueError as e:
        print(f"  Caught bad thrust_range: {e}")
    print('  PASSED\n')

    # ------------------------------------------------------------------
    # 2. Pareto frontier: 2-point sweep
    # ------------------------------------------------------------------
    print('Test 2: generate_pareto_frontier (n_points=2, N=5)\n')
    pareto = generate_pareto_frontier(base, [180.0, 250.0], n_points=2,
                                       integ_tol=1e-8, warm_start=True)

    print(f"\n  Returned {len(pareto)} converged point(s)\n")

    required_keys = {
        'tof_days', 'propellant_fraction', 'propellant_mass_kg',
        'final_mass_kg', 'throttle', 'solver_message',
    }
    for p in pareto:
        assert required_keys.issubset(p.keys()), f"Missing keys: {required_keys - p.keys()}"
        assert 100.0 <= p['tof_days']            <= 400.0
        assert 0.0   <= p['propellant_fraction'] <= 1.0
        assert 0.0   <= p['throttle']            <= 1.0

    if len(pareto) == 2:
        print(f"  TOF {pareto[0]['tof_days']:.1f} d: "
              f"prop_frac={pareto[0]['propellant_fraction']:.4f}, "
              f"throttle={pareto[0]['throttle']:.3f}")
        print(f"  TOF {pareto[1]['tof_days']:.1f} d: "
              f"prop_frac={pareto[1]['propellant_fraction']:.4f}, "
              f"throttle={pareto[1]['throttle']:.3f}")
        if pareto[0]['throttle'] > pareto[1]['throttle']:
            print("  Monotonicity OK: shorter TOF -> higher throttle")
    print('  PASSED\n')

    # ------------------------------------------------------------------
    # 3. Power sweep: 2-point sweep
    # ------------------------------------------------------------------
    print('Test 3: generate_power_sweep (n_points=2, N=5)\n')
    sweep = generate_power_sweep(base, [1.0, 5.0], n_points=2,
                                  integ_tol=1e-8)

    print(f"\n  Returned {len(sweep)} converged point(s)\n")

    required_sweep_keys = {
        'thrust_N', 'specific_power_W_kg', 'tof_days',
        'propellant_mass_kg', 'solver_message',
    }
    for s in sweep:
        assert required_sweep_keys.issubset(s.keys()), \
            f"Missing keys: {required_sweep_keys - s.keys()}"
        assert s['thrust_N']            > 0
        assert s['specific_power_W_kg'] > 0
        assert s['tof_days']            > 0

    # Verify specific power formula: P = T*Isp*g0 / (2*eta*m0)
    for s in sweep:
        expected_P = (s['thrust_N'] * base.isp_s * g0) / (
            2 * _ETA_THRUSTER * base.initial_mass_kg
        )
        assert abs(s['specific_power_W_kg'] - expected_P) < 1e-6, \
            f"Specific power formula mismatch at T={s['thrust_N']:.3f} N"

    if len(sweep) == 2:
        print(f"  T={sweep[0]['thrust_N']:.2f} N: "
              f"P_sp={sweep[0]['specific_power_W_kg']:.2f} W/kg, "
              f"TOF={sweep[0]['tof_days']:.1f} d")
        print(f"  T={sweep[1]['thrust_N']:.2f} N: "
              f"P_sp={sweep[1]['specific_power_W_kg']:.2f} W/kg, "
              f"TOF={sweep[1]['tof_days']:.1f} d")
        if (len(sweep) == 2 and sweep[0]['tof_days'] > 0
                and sweep[1]['tof_days'] > 0
                and sweep[1]['tof_days'] < sweep[0]['tof_days']):
            print("  Monotonicity OK: higher thrust -> shorter TOF")
    print('  PASSED')
