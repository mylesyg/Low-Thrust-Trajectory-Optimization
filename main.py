#!/usr/bin/env python3
"""Earth-Mars low-thrust trajectory optimization — top-level pipeline driver.

4-week milestone plan
---------------------
  Week 1   Hohmann impulsive baseline + ND state validation
  Week 2   Time-optimal low-thrust (2D coplanar)
  Week 3   Mass-optimal solve + Pareto frontier + power sweep
  Week 4   Nonplanar 3D extension with inclination change

Usage
-----
    python main.py --week 1               # Week 1 only
    python main.py --week 2               # Week 2 only (N=200, may take minutes)
    python main.py --week 2 --quick       # Week 2 fast demo (N=20)
    python main.py --week all             # Weeks 1-4 in sequence
    python main.py --week all --quick     # Full pipeline, fast demo segments

All figures are saved to results/.
"""

import argparse
import os
import pickle
import sys
import textwrap
import time as _time
from dataclasses import replace

import numpy as np

# ---------------------------------------------------------------------------
# Set matplotlib backend BEFORE any pyplot import so figures are written to
# files without requiring a GUI display.  This must come before importing the
# visualization sub-package (which imports pyplot at module level).
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from config.mission_config import MissionConfig
from core.constants import r_earth, r_mars
from baseline.impulsive_transfer import compute_baseline
from ephemeris.planetary_states import (
    get_initial_state_earth,
    get_final_state_mars,
    compute_departure_date,
    jd_to_calendar_string,
)
from optimization.time_optimal import solve_time_optimal
from optimization.mass_optimal import solve_mass_optimal
from spiral.spiral_config import SpiralConfig
from spiral.spiral_optimizer import solve_time_optimal_spiral
from analysis.pareto import generate_pareto_frontier, generate_power_sweep
from nonplanar.nonplanar_optimizer import solve_nonplanar
from visualization.trajectory_plot import plot_2d_trajectory, plot_state_history
from visualization.control_plot import plot_steering_history, plot_mass_history
from visualization.pareto_plot import plot_pareto_frontier
from visualization.pareto_plot import plot_power_sweep as _vis_plot_power_sweep
from visualization.porkchop_plot import compute_porkchop_data, plot_porkchop

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE        = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(_HERE, 'results')


# ===========================================================================
# Console formatting helpers
# ===========================================================================

def _banner(title: str) -> None:
    bar = '=' * 64
    print(f'\n{bar}')
    print(f'  {title}')
    print(bar)


def _section(title: str) -> None:
    print(f'\n  -- {title}')


def _kv(label: str, value: str) -> None:
    print(f'     {label:<30s} {value}')


# ===========================================================================
# Private helpers
# ===========================================================================

def _iso_to_jd(iso_date: str) -> float:
    """Convert an ISO-format date string to a Julian Date via astropy."""
    from astropy.time import Time
    return Time(iso_date, format='iso', scale='utc').jd


def _quick_cfg(cfg: MissionConfig, n: int = 20) -> MissionConfig:
    """Return a copy of *cfg* with n_segments=n for fast testing."""
    return replace(cfg, n_segments=n)


def _save(fig: plt.Figure, filename: str) -> None:
    """Save *fig* to RESULTS_DIR/<filename> and close it."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'     Saved -> results/{filename}')


def _save_result(week: str, data: dict) -> None:
    """Pickle *data* to results/weekN_result.pkl."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f'week{week}_result.pkl')
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f'     Saved results -> results/week{week}_result.pkl')


def _load_result(week: str) -> dict:
    """Load pickled result from results/weekN_result.pkl."""
    path = os.path.join(RESULTS_DIR, f'week{week}_result.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved results found at {path}.\n"
            f"  Run first:  python main.py --week {week} --save"
        )
    with open(path, 'rb') as f:
        return pickle.load(f)


def _plot_3d_trajectory(
    res: dict,
    i2_deg: float,
    raan2_deg: float,
    save_path: str = None,
) -> plt.Figure:
    """Render the 3D Cartesian nonplanar transfer trajectory.

    The state history from ``solve_nonplanar`` is in Cartesian coordinates
    [x, y, z, vx, vy, vz, mp, acc_dv].  Planet orbits are projected onto the
    z = 0 (ecliptic) plane for reference.

    Parameters
    ----------
    res : dict
        Output of solve_nonplanar().  Must contain 'state_history' (N+1, 8).
    i2_deg : float
        Target orbit inclination [deg] — used in title annotation.
    raan2_deg : float
        Target orbit RAAN [deg] — used in title annotation.
    save_path : str, optional
        Full file path to save the figure.

    Returns
    -------
    fig or None
        Returns None if mpl_toolkits.mplot3d is not available.
    """
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        print('     mpl_toolkits.mplot3d not available — 3D plot skipped.')
        return None

    sh = res['state_history']   # (N+1, 8): [x, y, z, ...]
    xt, yt, zt = sh[:, 0], sh[:, 1], sh[:, 2]

    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection='3d')

    # Transfer path
    ax.plot(xt, yt, zt, color='black', linewidth=1.5, label='Transfer')
    ax.scatter(xt[0],  yt[0],  zt[0],  color='royalblue', s=60,
               marker='o', zorder=5, label='Departure')
    ax.scatter(xt[-1], yt[-1], zt[-1], color='firebrick',  s=60,
               marker='s', zorder=5, label='Arrival')
    ax.scatter(0, 0, 0, color='gold', s=200, marker='*', zorder=6, label='Sun')

    # Planet orbits projected onto z = 0 (ecliptic plane)
    th   = np.linspace(0.0, 2.0 * np.pi, 360)
    z0   = np.zeros(360)
    ax.plot(r_earth * np.cos(th), r_earth * np.sin(th), z0,
            'b--', linewidth=0.8, alpha=0.5, label='Earth orbit')
    ax.plot(r_mars  * np.cos(th), r_mars  * np.sin(th), z0,
            'r--', linewidth=0.8, alpha=0.5, label='Mars orbit (ecliptic)')

    tof = res.get('tof_days', float('nan'))
    ax.set_xlabel('X [AU]', fontsize=10)
    ax.set_ylabel('Y [AU]', fontsize=10)
    ax.set_zlabel('Z [AU]', fontsize=10)
    ax.set_title(
        f'3D Nonplanar Transfer  (TOF = {tof:.1f} d)\n'
        f'Target: i = {i2_deg:.1f} deg,  RAAN = {raan2_deg:.1f} deg',
        fontsize=12,
    )
    ax.legend(fontsize=8, loc='upper left')

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# ===========================================================================
# Week runners
# ===========================================================================

# ===========================================================================
# Spiral runners
# ===========================================================================

def run_escape_spiral(args) -> dict:
    """Earth escape: low-thrust spiral from LEO to the Earth SOI."""
    _banner('EARTH ESCAPE SPIRAL  —  Time-Optimal Planetocentric Spiral')

    if args.quick:
        cfg = SpiralConfig(
            planet             = 'earth',
            r0_km              = 6371.0 + 400.0,
            rf_km              = 925_000.0,
            initial_mass_kg    = 500.0,
            propellant_mass_kg = 200.0,
            thrust_N           = 0.5,
            isp_s              = 3000.0,
            n_segments         = 20,
        )
    else:
        cfg = SpiralConfig.default_earth_escape()

    _section('Configuration')
    _kv('Planet',          cfg.planet)
    _kv('r0 (LEO)',        f'{cfg.r0_km:.1f} km')
    _kv('rf (SOI)',        f'{cfg.rf_km:.0f} km')
    _kv('Initial mass',    f'{cfg.initial_mass_kg:.0f} kg')
    _kv('Propellant mass', f'{cfg.propellant_mass_kg:.0f} kg')
    _kv('Thrust',          f'{cfg.thrust_N:.2f} N')
    _kv('Isp',             f'{cfg.isp_s:.0f} s')
    _kv('N segments',      str(cfg.n_segments))

    nd = cfg.to_nd_params()
    tof_est_days = nd['tof_nd'] * nd['t_star_s'] / 86_400.0
    _kv('TOF estimate (Edelbaum)', f'{tof_est_days:.1f} days')

    _n_jobs = os.cpu_count() or 1
    print(f'\n  Solving escape spiral NLP (N={cfg.n_segments}, n_jobs={_n_jobs}) ...')
    res = solve_time_optimal_spiral(cfg, n_jobs=_n_jobs, verbose=True)

    _section('Optimizer output')
    _kv('Converged',      'YES' if res['success'] else 'NO (see solver message)')
    _kv('Solver message', res['solver_message'])
    _kv('Wall time',      f'{res["wall_time_s"]:.1f} s')
    _kv('N iterations',   str(res['n_iterations']))

    _section('Trajectory summary')
    tof_days = res['tof_seconds'] / 86_400.0
    _kv('Transfer time',       f'{tof_days:.2f} days  ({res["tof_seconds"]/3600:.1f} h)')
    _kv('Propellant consumed', f'{res["propellant_used_kg"]:.2f} kg  '
        f'({res["propellant_used_kg"]/cfg.initial_mass_kg*100:.1f}%)')
    _kv('Final mass',          f'{res["final_mass_kg"]:.2f} kg')
    _kv('Delta-V',             f'{res["delta_v_ms"]/1e3:.3f} km/s')

    if args.save:
        _save_result('escape_spiral', {'res': res, 'cfg': cfg})
    return res


def run_capture_spiral(args) -> dict:
    """Mars capture: low-thrust spiral from the Mars SOI to LMO."""
    _banner('MARS CAPTURE SPIRAL  —  Time-Optimal Planetocentric Spiral')

    if args.quick:
        cfg = SpiralConfig(
            planet             = 'mars',
            r0_km              = 577_000.0,
            rf_km              = 3389.5 + 400.0,
            initial_mass_kg    = 500.0,
            propellant_mass_kg = 200.0,
            thrust_N           = 0.5,
            isp_s              = 3000.0,
            n_segments         = 20,
        )
    else:
        cfg = SpiralConfig.default_mars_capture()

    _section('Configuration')
    _kv('Planet',          cfg.planet)
    _kv('r0 (SOI)',        f'{cfg.r0_km:.0f} km')
    _kv('rf (LMO)',        f'{cfg.rf_km:.1f} km')
    _kv('Initial mass',    f'{cfg.initial_mass_kg:.0f} kg')
    _kv('Propellant mass', f'{cfg.propellant_mass_kg:.0f} kg')
    _kv('Thrust',          f'{cfg.thrust_N:.2f} N')
    _kv('Isp',             f'{cfg.isp_s:.0f} s')
    _kv('N segments',      str(cfg.n_segments))

    nd = cfg.to_nd_params()
    tof_est_days = nd['tof_nd'] * nd['t_star_s'] / 86_400.0
    _kv('TOF estimate (Edelbaum)', f'{tof_est_days:.1f} days')

    _n_jobs = os.cpu_count() or 1
    print(f'\n  Solving capture spiral NLP (N={cfg.n_segments}, n_jobs={_n_jobs}) ...')
    res = solve_time_optimal_spiral(cfg, n_jobs=_n_jobs, verbose=True)

    _section('Optimizer output')
    _kv('Converged',      'YES' if res['success'] else 'NO (see solver message)')
    _kv('Solver message', res['solver_message'])
    _kv('Wall time',      f'{res["wall_time_s"]:.1f} s')
    _kv('N iterations',   str(res['n_iterations']))

    _section('Trajectory summary')
    tof_days = res['tof_seconds'] / 86_400.0
    _kv('Transfer time',       f'{tof_days:.2f} days  ({res["tof_seconds"]/3600:.1f} h)')
    _kv('Propellant consumed', f'{res["propellant_used_kg"]:.2f} kg  '
        f'({res["propellant_used_kg"]/cfg.initial_mass_kg*100:.1f}%)')
    _kv('Final mass',          f'{res["final_mass_kg"]:.2f} kg')
    _kv('Delta-V',             f'{res["delta_v_ms"]/1e3:.3f} km/s')

    if args.save:
        _save_result('capture_spiral', {'res': res, 'cfg': cfg})
    return res


def run_week1(args) -> dict:
    """Week 1: Hohmann impulsive baseline + non-dimensional state validation."""
    _banner('WEEK 1  —  Hohmann Impulsive Baseline')

    cfg = MissionConfig.default_time_optimal()   # spacecraft params only

    _section('Spacecraft parameters')
    _kv('Initial mass',  f'{cfg.initial_mass_kg:.0f} kg')
    _kv('Thrust',        f'{cfg.thrust_N:.2f} N')
    _kv('Specific impulse (Isp)', f'{cfg.isp_s:.0f} s')

    _section('Hohmann transfer  (Earth -> Mars, coplanar circular orbits)')
    bl = compute_baseline(cfg)
    _kv('Transfer SMA',         f'{bl["a_transfer_au"]:.5f} AU')
    _kv('Transfer time',        f'{bl["tof_days"]:.2f} days')
    _kv('Departure burn dV1',   f'{bl["delta_v1_ms"]/1e3:.3f} km/s')
    _kv('Arrival burn dV2',     f'{bl["delta_v2_ms"]/1e3:.3f} km/s')
    _kv('Total delta-V',        f'{bl["total_delta_v_ms"]/1e3:.3f} km/s')
    _kv('Propellant consumed',
        f'{bl["propellant_mass_kg"]:.1f} kg  ({bl["propellant_fraction"]*100:.1f}%)')
    _kv('Final mass',           f'{bl["final_mass_kg"]:.1f} kg')
    _kv('Mass ratio m0/mf',     f'{bl["mass_ratio"]:.4f}')

    _section('Non-dimensional initial / final states')
    y0 = get_initial_state_earth()
    yf = get_final_state_mars()
    _kv('y0  [r, u, v, theta, mp, acc_dv]',
        f'[{y0[0]:.3f}, {y0[1]:.3f}, {y0[2]:.3f}, '
        f'{y0[3]:.3f}, {y0[4]:.3f}, {y0[5]:.3f}]')
    _kv('yf  [r, u, v, theta, mp, acc_dv]',
        f'[{yf[0]:.5f}, {yf[1]:.3f}, {yf[2]:.5f}, NaN, NaN, NaN]')

    # Validation assertions
    assert abs(y0[0] - 1.0) < 1e-10,                     'Earth r_ND must be 1.0'
    assert abs(y0[2] - 1.0) < 1e-10,                     'Earth v_circ_ND must be 1.0'
    assert abs(yf[0] - r_mars) < 1e-10,                  'Mars r_ND must match constant'
    assert abs(yf[2] - np.sqrt(1.0 / r_mars)) < 1e-10,   'Mars v_circ_ND: vis-viva check'
    assert np.isnan(yf[3]) and np.isnan(yf[4]) and np.isnan(yf[5]), \
        'theta_f, mp_f, acc_dv_f must be NaN (free)'
    print('\n     State validation: PASSED')

    if args.save:
        _save_result('1', {'bl': bl, 'cfg': cfg})
    return bl


def run_week2(args) -> dict:
    """Week 2: time-optimal low-thrust trajectory (2D coplanar)."""
    _banner('WEEK 2  —  Time-Optimal Low-Thrust Transfer')

    cfg = MissionConfig.default_time_optimal()
    if args.quick:
        cfg = _quick_cfg(cfg)

    _section('Configuration')
    _kv('n_segments',  str(cfg.n_segments))
    _kv('TOF window',  f'[{cfg.time_lb_days:.0f}, {cfg.time_ub_days:.0f}] days')
    _kv('Thrust',      f'{cfg.thrust_N:.2f} N  (full throttle)')
    _kv('Isp',         f'{cfg.isp_s:.0f} s')

    _n_jobs = os.cpu_count() or 1
    print(f'\n  Solving time-optimal NLP (N = {cfg.n_segments} segments,'
          f' n_jobs = {_n_jobs}) ...')
    res = solve_time_optimal(cfg, n_jobs=_n_jobs)

    _section('Optimizer output')
    _kv('Converged',       'YES' if res['success'] else 'NO (see solver message)')
    _kv('Solver message',  res['solver_message'])
    _kv('Wall time',       f'{res["wall_time_s"]:.1f} s')
    _kv('N iterations',    str(res['n_iterations']))

    _section('Trajectory summary')
    _kv('Transfer time',      f'{res["tof_days"]:.2f} days')
    _kv('Propellant consumed',
        f'{res["propellant_mass_kg"]:.1f} kg  '
        f'({res["propellant_mass_kg"]/cfg.initial_mass_kg*100:.1f}%)')
    _kv('Final (delivered) mass', f'{res["final_mass_kg"]:.1f} kg')

    # Departure / arrival calendar dates
    jd_guess        = _iso_to_jd(cfg.departure_date_guess)
    jd_dep, jd_arr, _ = compute_departure_date(
        jd_guess,
        res['tof_days'],
        transfer_angle_deg=200.0,   # typical low-thrust swept angle
    )
    _kv('Estimated departure', jd_to_calendar_string(jd_dep))
    _kv('Estimated arrival',   jd_to_calendar_string(jd_arr))

    _section('Figures')
    n_arrows = getattr(args, 'thrust_arrows', 20)
    _save(plot_2d_trajectory(res, cfg, n_thrust_arrows=n_arrows), 'week2_trajectory.png')
    _save(plot_steering_history(res, cfg), 'week2_steering.png')
    _save(plot_state_history(res, cfg),   'week2_states.png')
    _save(plot_mass_history(res, cfg),    'week2_mass.png')

    _section('Pork Chop Plot')
    dep_range = [jd_dep - 50.0, jd_dep + 200.0]
    arr_range = [dep_range[0] + 50.0,   dep_range[1] + 500.0]
    n_pc = 75
    _kv('Departure window',
        f'{jd_to_calendar_string(dep_range[0])} – '
        f'{jd_to_calendar_string(dep_range[1])}')
    _kv('Arrival window',
        f'{jd_to_calendar_string(arr_range[0])} – '
        f'{jd_to_calendar_string(arr_range[1])}')
    print(f'\n  Computing {n_pc}×{n_pc} = {n_pc * n_pc} Lambert solves ...')

    pc_data = compute_porkchop_data(dep_range, arr_range,
                                    n_departure=n_pc, n_arrival=n_pc)

    if np.isfinite(pc_data['C3_depart']).any():
        i_min, j_min = np.unravel_index(
            np.nanargmin(pc_data['C3_depart']), pc_data['C3_depart'].shape,
        )
        _kv('Min C3 in grid',
            f'{pc_data["C3_depart"][i_min, j_min]:.2f} km\u00b2/s\u00b2')
        _kv('  -> Departure', jd_to_calendar_string(pc_data['departure_jd'][j_min]))
        _kv('  -> Arrival',   jd_to_calendar_string(pc_data['arrival_jd'][i_min]))
    else:
        _kv('Min C3 in grid', 'N/A (no converged Lambert solutions in range)')

    optimal_point = {'departure_jd': jd_dep, 'arrival_jd': jd_arr}
    _kv('Optimal (low-thrust) departure', jd_to_calendar_string(jd_dep))
    _kv('Optimal (low-thrust) arrival',   jd_to_calendar_string(jd_arr))

    _save(plot_porkchop(pc_data, optimal_result=optimal_point), 'porkchop_plot.png')

    if args.save:
        _save_result('2', {'res': res, 'cfg': cfg})
    return res


def run_week3(args) -> dict:
    """Week 3: mass-optimal solve, Pareto frontier, and power sweep."""
    _banner('WEEK 3  —  Mass-Optimal + Pareto Frontier + Power Sweep')

    # Auto-detect available CPU cores for parallel sweep execution.
    # Each sweep point is an independent NLP solve, so parallelism is
    # embarrassingly parallel.  Cap at the number of sweep points so we
    # don't spawn idle workers.
    import os as _os
    _cpu = _os.cpu_count() or 1

    # Integration tolerance for sweep solves: 1e-8 is ~2-3× faster than
    # the standalone default (1e-10) with negligible accuracy loss
    # (constraint tolerance is 1e-6, so 1e-8 has 100× margin).
    _SWEEP_TOL = 1e-8

    # ------------------------------------------------------------------ #
    # 3a. Single mass-optimal solve at fixed TOF = 200 days               #
    # ------------------------------------------------------------------ #
    cfg_mo = MissionConfig.default_mass_optimal()
    if args.quick:
        cfg_mo = _quick_cfg(cfg_mo)

    _section(f'Mass-optimal solve  (fixed TOF = {cfg_mo.time_guess_days:.0f} days,'
             f'  N = {cfg_mo.n_segments} segments)')
    print(f'\n  Running mass-optimal NLP (integ_tol={_SWEEP_TOL:.0e},'
          f' n_jobs={_cpu}) ...')
    res_mo = solve_mass_optimal(cfg_mo, integ_tol=_SWEEP_TOL, n_jobs=_cpu)

    _kv('Converged',      'YES' if res_mo['success'] else 'NO (see solver message)')
    _kv('Solver message', res_mo['solver_message'])
    _kv('Wall time',      f'{res_mo["wall_time_s"]:.1f} s')
    _kv('Optimal throttle',   f'{res_mo["throttle"]:.4f}')
    _kv('Propellant consumed',
        f'{res_mo["propellant_mass_kg"]:.1f} kg  '
        f'({res_mo["propellant_mass_kg"]/cfg_mo.initial_mass_kg*100:.1f}%)')
    _kv('Final mass', f'{res_mo["final_mass_kg"]:.1f} kg')

    # ------------------------------------------------------------------ #
    # 3b. Pareto frontier: TOF sweep, mass-optimal at each point           #
    # ------------------------------------------------------------------ #
    n_pareto = 5 if args.quick else 15
    cfg_base = MissionConfig.default_time_optimal()
    if args.quick:
        cfg_base = _quick_cfg(cfg_base)

    n_jobs_pareto = min(_cpu, n_pareto)
    _section(f'Pareto frontier sweep  (n = {n_pareto} points,'
             f'  TOF in [180, 400] days,  n_jobs = {n_jobs_pareto})')
    pareto = generate_pareto_frontier(
        cfg_base, [180.0, 400.0], n_points=n_pareto,
        integ_tol=_SWEEP_TOL,
        warm_start=(n_jobs_pareto == 1),
        n_jobs=n_jobs_pareto,
    )

    if pareto:
        _kv('Converged points', f'{len(pareto)} / {n_pareto}')
        _kv('TOF range',
            f'{pareto[0]["tof_days"]:.0f} – {pareto[-1]["tof_days"]:.0f} days')
        _kv('Propellant fraction range',
            f'{pareto[0]["propellant_fraction"]*100:.1f}% – '
            f'{pareto[-1]["propellant_fraction"]*100:.1f}%')
    else:
        print('     No points converged.')

    # ------------------------------------------------------------------ #
    # 3c. Power sweep: thrust level sweep, time-optimal at each point      #
    # ------------------------------------------------------------------ #
    n_power = 5 if args.quick else 10
    n_jobs_sweep = min(_cpu, n_power)
    _section(f'Power sweep  (n = {n_power} thrust levels,'
             f'  T in [1.0, 10.0] N,  n_jobs = {n_jobs_sweep})')
    sweep = generate_power_sweep(
        cfg_base, [1.0, 10.0], n_points=n_power,
        integ_tol=_SWEEP_TOL,
        n_jobs=n_jobs_sweep,
        adapt_tof_bounds=True,
    )

    if sweep:
        _kv('Converged points', f'{len(sweep)} / {n_power}')
        _kv('Specific power range',
            f'{sweep[0]["specific_power_W_kg"]:.1f} – '
            f'{sweep[-1]["specific_power_W_kg"]:.1f} W/kg')
        _kv('TOF range',
            f'{sweep[0]["tof_days"]:.0f} – {sweep[-1]["tof_days"]:.0f} days')
    else:
        print('     No points converged.')

    # Hohmann reference for power sweep plot
    bl = compute_baseline(cfg_base)

    _section('Figures')
    pareto_idx = getattr(args, 'pareto_idx', -1)
    if pareto:
        sidx = int(pareto_idx) % len(pareto)
        _save(plot_pareto_frontier(pareto, selected_idx=sidx), 'week3_pareto.png')
    else:
        print('     Pareto figure skipped (no converged points).')

    if sweep:
        _save(_vis_plot_power_sweep(sweep, bl), 'week3_power_sweep.png')
    else:
        print('     Power sweep figure skipped (no converged points).')

    if pareto:
        sidx = int(pareto_idx) % len(pareto)
        pt   = pareto[sidx]
        if 'state_history' in pt and 'alphas_rad' in pt:
            _section(
                f'Pareto trajectory  (point {sidx}: '
                f'TOF = {pt["tof_days"]:.1f} d, '
                f'throttle = {pt["throttle"]:.3f})'
            )
            n_arrows = getattr(args, 'thrust_arrows', 20)
            _save(
                plot_2d_trajectory(
                    pt, cfg_base,
                    n_thrust_arrows=n_arrows,
                    opt_label='Mass-Optimal',
                    throttle_scale=pt['throttle'],
                ),
                'week3_pareto_trajectory.png',
            )
        else:
            print('     Pareto trajectory figure skipped '
                  '(state_history not stored — re-run without --plot-only).')

    if args.save:
        _save_result('3', {
            'res_mo': res_mo, 'cfg_mo': cfg_mo,
            'pareto': pareto, 'sweep': sweep,
            'bl': bl, 'cfg_base': cfg_base,
        })
    return {'mass_optimal': res_mo, 'pareto': pareto, 'sweep': sweep,
            'hohmann': bl}


def run_week4(args) -> dict:
    """Week 4: nonplanar 3D transfer with inclination change."""
    _banner('WEEK 4  —  Nonplanar 3D Extension')

    # Use fewer segments for the 3D problem: the decision vector is 2N+1 and
    # gradient evaluation cost scales with N.  N=50 is a reasonable balance
    # between accuracy and runtime; use N=20 in --quick mode.
    N_3D = 20 if args.quick else 50

    cfg = MissionConfig(
        initial_mass_kg      = 5000.0,
        thrust_N             = 3.5,
        isp_s                = 3000.0,
        opt_mode             = 'time_optimal',
        n_segments           = N_3D,
        time_guess_days      = 200.0,
        time_lb_days         = 175.0,
        time_ub_days         = 225.0,
        throttle_guess       = 1.0,
        throttle_lb          = 1.0,
        throttle_ub          = 1.0,
        alpha_guess_deg      = 20.0,
        alpha_lb_deg         = -180.0,
        alpha_ub_deg         =  180.0,
        departure_date_guess = '2020-01-01',
    )

    # Sample orbital element sets for the nonplanar case
    r1_elements = [1.0,     0.0, 0.0 ]   # Earth: equatorial circular
    r2_elements = [1.52368, 5.0, 49.6]   # Mars: 5 deg inclined, RAAN = 49.6 deg

    i2_deg   = r2_elements[1]
    raan2_deg = r2_elements[2]

    _section('Transfer specification')
    _kv('Departure orbit',
        f'a = {r1_elements[0]:.2f} AU,  '
        f'i = {r1_elements[1]:.1f} deg,  '
        f'RAAN = {r1_elements[2]:.1f} deg  (Earth equatorial)')
    _kv('Target orbit',
        f'a = {r2_elements[0]:.5f} AU,  '
        f'i = {i2_deg:.1f} deg,  '
        f'RAAN = {raan2_deg:.1f} deg  (Mars inclined)')
    _kv('Inclination change', f'{i2_deg - r1_elements[1]:.1f} deg')
    _kv('N segments (3D)',    str(N_3D))

    _n_jobs_4 = os.cpu_count() or 1
    _INTEG_TOL_4 = 1e-8
    print(f'\n  Solving nonplanar NLP (decision vector length = {2*N_3D+1},'
          f' n_jobs = {_n_jobs_4}) ...')
    res = solve_nonplanar(cfg, r1_elements, r2_elements,
                          integ_tol=_INTEG_TOL_4, n_jobs=_n_jobs_4)

    _section('Optimizer output')
    _kv('Converged',      'YES' if res['success'] else 'NO (see solver message)')
    _kv('Solver message', res['solver_message'])
    _kv('Wall time',      f'{res["wall_time_s"]:.1f} s')
    _kv('N iterations',   str(res['n_iterations']))

    _section('Trajectory summary')
    _kv('Transfer time',
        f'{res["tof_days"]:.2f} days')
    _kv('Propellant consumed',
        f'{res["propellant_mass_kg"]:.1f} kg  '
        f'({res["propellant_mass_kg"]/cfg.initial_mass_kg*100:.1f}%)')
    _kv('Final (delivered) mass', f'{res["final_mass_kg"]:.1f} kg')
    _kv('Steering: max |alpha|',
        f'{np.degrees(np.abs(res["alphas_rad"]).max()):.2f} deg')
    _kv('Steering: max |beta|',
        f'{np.degrees(np.abs(res["betas_rad"]).max()):.2f} deg')

    # Terminal constraint residuals
    _section('Terminal constraint residuals at solution')
    from nonplanar.nonplanar_optimizer import _shooting_3d, _initial_state_from_elements
    nd   = cfg.to_nd_params()
    y0   = _initial_state_from_elements(r1_elements[0], r1_elements[1], r1_elements[2])
    x_opt = np.concatenate([
        [res['tof_days'] / 58.13],   # approx ND conversion for display only
        res['alphas_rad'],
        res['betas_rad'],
    ])
    # Re-use the final eval stored in state_history instead of re-propagating
    sh   = res['state_history']
    yf   = sh[-1]
    r_f  = yf[:3];  v_f = yf[3:6]
    r_fm = np.linalg.norm(r_f); v_fm = np.linalg.norm(v_f)
    h_f  = np.cross(r_f, v_f);  h_fm = max(np.linalg.norm(h_f), 1e-10)
    r2   = r2_elements[0]
    vc2  = np.sqrt(1.0 / r2)
    i2r  = np.radians(i2_deg); raan2r = np.radians(raan2_deg)
    _kv('c1 = (|r_f|/r2 - 1)',     f'{(r_fm - r2)/r2:+.4e}  (target 0)')
    _kv('c2 = r·v / (r2·vc2)',     f'{float(np.dot(r_f,v_f))/(r2*vc2):+.4e}  (target 0)')
    _kv('c3 = (|v_f|/vc2 - 1)',    f'{(v_fm - vc2)/vc2:+.4e}  (target 0)')
    _kv('c4 = hz/|h| - cos(i2)',   f'{h_f[2]/h_fm - np.cos(i2r):+.4e}  (target 0)')
    _kv('c5 = RAAN constraint',
        f'{(h_f[0]*np.cos(raan2r)+h_f[1]*np.sin(raan2r))/h_fm:+.4e}  (target 0)')

    _section('Figures')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_3d = os.path.join(RESULTS_DIR, 'week4_3d_trajectory.png')
    fig_3d  = _plot_3d_trajectory(res, i2_deg, raan2_deg, save_path=save_3d)
    if fig_3d is not None:
        plt.close(fig_3d)
        print(f'     Saved -> results/week4_3d_trajectory.png')

    if args.save:
        _save_result('4', {
            'res': res, 'cfg': cfg,
            'r1_elements': r1_elements,
            'r2_elements': r2_elements,
        })
    return res


# ===========================================================================
# Plot-only runners  (load saved .pkl → regenerate figures, skip solver)
# ===========================================================================

def plot_week1(data: dict, args) -> None:
    """Re-display Week 1 Hohmann summary from saved results (no figures)."""
    _banner('WEEK 1  —  Hohmann Impulsive Baseline  [saved results]')
    bl  = data['bl']
    cfg = data['cfg']
    _section('Spacecraft parameters')
    _kv('Initial mass',           f'{cfg.initial_mass_kg:.0f} kg')
    _kv('Thrust',                 f'{cfg.thrust_N:.2f} N')
    _kv('Specific impulse (Isp)', f'{cfg.isp_s:.0f} s')
    _section('Hohmann transfer  (Earth -> Mars, coplanar circular orbits)')
    _kv('Transfer SMA',       f'{bl["a_transfer_au"]:.5f} AU')
    _kv('Transfer time',      f'{bl["tof_days"]:.2f} days')
    _kv('Departure burn dV1', f'{bl["delta_v1_ms"]/1e3:.3f} km/s')
    _kv('Arrival burn dV2',   f'{bl["delta_v2_ms"]/1e3:.3f} km/s')
    _kv('Total delta-V',      f'{bl["total_delta_v_ms"]/1e3:.3f} km/s')
    _kv('Propellant consumed',
        f'{bl["propellant_mass_kg"]:.1f} kg  ({bl["propellant_fraction"]*100:.1f}%)')
    _kv('Final mass',         f'{bl["final_mass_kg"]:.1f} kg')
    _kv('Mass ratio m0/mf',   f'{bl["mass_ratio"]:.4f}')
    print('\n     (Week 1 has no figures to regenerate.)')


def plot_week2(data: dict, args) -> None:
    """Regenerate Week 2 figures from saved time-optimal results."""
    _banner('WEEK 2  —  Time-Optimal Low-Thrust Transfer  [saved results]')
    res = data['res']
    cfg = data['cfg']
    _section('Trajectory summary')
    _kv('Converged',      'YES' if res['success'] else 'NO')
    _kv('Transfer time',  f'{res["tof_days"]:.2f} days')
    _kv('Propellant consumed',
        f'{res["propellant_mass_kg"]:.1f} kg  '
        f'({res["propellant_mass_kg"]/cfg.initial_mass_kg*100:.1f}%)')
    _kv('Final (delivered) mass', f'{res["final_mass_kg"]:.1f} kg')

    # Re-derive departure / arrival dates from the saved result (deterministic)
    jd_guess = _iso_to_jd(cfg.departure_date_guess)
    jd_dep, jd_arr, _ = compute_departure_date(
        jd_guess,
        res['tof_days'],
        transfer_angle_deg=200.0,
    )
    _kv('Estimated departure', jd_to_calendar_string(jd_dep))
    _kv('Estimated arrival',   jd_to_calendar_string(jd_arr))

    _section('Figures')
    n_arrows = getattr(args, 'thrust_arrows', 20)
    _save(plot_2d_trajectory(res, cfg, n_thrust_arrows=n_arrows), 'week2_trajectory.png')
    _save(plot_steering_history(res, cfg), 'week2_steering.png')
    _save(plot_state_history(res, cfg),    'week2_states.png')
    _save(plot_mass_history(res, cfg),     'week2_mass.png')

    _section('Pork Chop Plot')
    dep_range = [jd_dep - 365.0 / 2.0, jd_dep + 365.0 / 2.0]
    arr_range = [dep_range[0] + 50.0,   dep_range[1] + 500.0]
    n_pc = 75
    _kv('Departure window',
        f'{jd_to_calendar_string(dep_range[0])} – '
        f'{jd_to_calendar_string(dep_range[1])}')
    _kv('Arrival window',
        f'{jd_to_calendar_string(arr_range[0])} – '
        f'{jd_to_calendar_string(arr_range[1])}')
    print(f'\n  Computing {n_pc}\u00d7{n_pc} = {n_pc * n_pc} Lambert solves ...')

    pc_data = compute_porkchop_data(dep_range, arr_range,
                                    n_departure=n_pc, n_arrival=n_pc)

    if np.isfinite(pc_data['C3_depart']).any():
        i_min, j_min = np.unravel_index(
            np.nanargmin(pc_data['C3_depart']), pc_data['C3_depart'].shape,
        )
        _kv('Min C3 in grid',
            f'{pc_data["C3_depart"][i_min, j_min]:.2f} km\u00b2/s\u00b2')
        _kv('  -> Departure', jd_to_calendar_string(pc_data['departure_jd'][j_min]))
        _kv('  -> Arrival',   jd_to_calendar_string(pc_data['arrival_jd'][i_min]))
    else:
        _kv('Min C3 in grid', 'N/A (no converged Lambert solutions in range)')

    optimal_point = {'departure_jd': jd_dep, 'arrival_jd': jd_arr}
    _kv('Optimal (low-thrust) departure', jd_to_calendar_string(jd_dep))
    _kv('Optimal (low-thrust) arrival',   jd_to_calendar_string(jd_arr))

    _save(plot_porkchop(pc_data, optimal_result=optimal_point), 'porkchop_plot.png')


def plot_week3(data: dict, args) -> None:
    """Regenerate Week 3 figures from saved mass-optimal / Pareto / sweep results."""
    _banner('WEEK 3  —  Mass-Optimal + Pareto Frontier + Power Sweep  [saved results]')
    res_mo   = data['res_mo']
    cfg_mo   = data['cfg_mo']
    pareto   = data['pareto']
    sweep    = data['sweep']
    bl       = data['bl']

    _section('Mass-optimal summary')
    _kv('Converged',      'YES' if res_mo['success'] else 'NO')
    _kv('Optimal throttle',   f'{res_mo["throttle"]:.4f}')
    _kv('Propellant consumed',
        f'{res_mo["propellant_mass_kg"]:.1f} kg  '
        f'({res_mo["propellant_mass_kg"]/cfg_mo.initial_mass_kg*100:.1f}%)')
    _kv('Final mass',     f'{res_mo["final_mass_kg"]:.1f} kg')

    _section('Pareto sweep summary')
    if pareto:
        _kv('Converged points', str(len(pareto)))
        _kv('TOF range',
            f'{pareto[0]["tof_days"]:.0f} – {pareto[-1]["tof_days"]:.0f} days')
    else:
        print('     No converged Pareto points in saved data.')

    _section('Power sweep summary')
    if sweep:
        _kv('Converged points', str(len(sweep)))
        _kv('TOF range',
            f'{sweep[0]["tof_days"]:.0f} – {sweep[-1]["tof_days"]:.0f} days')
    else:
        print('     No converged power sweep points in saved data.')

    _section('Figures')
    pareto_idx = getattr(args, 'pareto_idx', -1)
    if pareto:
        sidx = int(pareto_idx) % len(pareto)
        _save(plot_pareto_frontier(pareto, selected_idx=sidx), 'week3_pareto.png')
    else:
        print('     Pareto figure skipped (no converged points in saved data).')
    if sweep:
        _save(_vis_plot_power_sweep(sweep, bl), 'week3_power_sweep.png')
    else:
        print('     Power sweep figure skipped (no converged points in saved data).')

    if pareto:
        sidx = int(pareto_idx) % len(pareto)
        pt   = pareto[sidx]
        if 'state_history' in pt and 'alphas_rad' in pt:
            _kv('Pareto trajectory point',
                f'idx {sidx},  TOF = {pt["tof_days"]:.1f} d,'
                f'  throttle = {pt["throttle"]:.3f}')
            cfg_base = data.get('cfg_base', data.get('cfg_mo'))
            n_arrows = getattr(args, 'thrust_arrows', 20)
            _save(
                plot_2d_trajectory(
                    pt, cfg_base,
                    n_thrust_arrows=n_arrows,
                    opt_label='Mass-Optimal',
                    throttle_scale=pt['throttle'],
                ),
                'week3_pareto_trajectory.png',
            )
        else:
            print('     Pareto trajectory figure skipped '
                  '(state_history not in saved data — re-run with --save).')


def plot_week4(data: dict, args) -> None:
    """Regenerate Week 4 3D trajectory figure from saved nonplanar results."""
    _banner('WEEK 4  —  Nonplanar 3D Extension  [saved results]')
    res         = data['res']
    r2_elements = data['r2_elements']
    i2_deg      = r2_elements[1]
    raan2_deg   = r2_elements[2]

    _section('Trajectory summary')
    _kv('Converged',      'YES' if res['success'] else 'NO')
    _kv('Transfer time',  f'{res["tof_days"]:.2f} days')
    _kv('Propellant consumed', f'{res["propellant_mass_kg"]:.1f} kg')
    _kv('Final mass',     f'{res["final_mass_kg"]:.1f} kg')

    _section('Figures')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_3d = os.path.join(RESULTS_DIR, 'week4_3d_trajectory.png')
    fig_3d  = _plot_3d_trajectory(res, i2_deg, raan2_deg, save_path=save_3d)
    if fig_3d is not None:
        plt.close(fig_3d)
        print('     Saved -> results/week4_3d_trajectory.png')
    else:
        print('     3D figure skipped (mpl_toolkits.mplot3d not available).')


# ===========================================================================
# CLI entry point
# ===========================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='main.py',
        description='Earth-Mars low-thrust trajectory optimization pipeline.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python main.py --week 1              Hohmann baseline only
              python main.py --week 2 --quick      Fast time-optimal demo (N=20)
              python main.py --week 3              Pareto + power sweep (N=200, slow)
              python main.py --week all --quick    Full pipeline, fast demo
              python main.py --week 2 --save       Solve week 2 and save results to disk
              python main.py --week 2 --plot-only  Reload saved results and regenerate figures
              python main.py --week all --save     Solve all weeks and save each to disk
        """),
    )
    parser.add_argument(
        '--week',
        choices=['1', '2', '3', '4', 'all'],
        required=False,
        default=None,
        metavar='WEEK',
        help="Week to run: 1 | 2 | 3 | 4 | all",
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help=(
            'Use N=20 segments for fast testing.  '
            'Default N values: 200 for 2D solvers, 50 for 3D (Week 4).'
        ),
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help=(
            'After each week completes, pickle the solver results to '
            'results/weekN_result.pkl so figures can be regenerated later '
            'with --plot-only.'
        ),
    )
    parser.add_argument(
        '--plot-only',
        action='store_true',
        dest='plot_only',
        help=(
            'Skip the solver entirely; load results/weekN_result.pkl and '
            'regenerate all figures.  Requires a prior run with --save.  '
            'Cannot be combined with --save.'
        ),
    )
    parser.add_argument(
        '--escape_spiral',
        action='store_true',
        dest='escape_spiral',
        help=(
            'Run the Earth escape spiral solver: time-optimal low-thrust '
            'spiral from a 400-km LEO to the Earth sphere of influence.  '
            'Use --quick for N=20 (fast demo), default is N=100.'
        ),
    )
    parser.add_argument(
        '--capture_spiral',
        action='store_true',
        dest='capture_spiral',
        help=(
            'Run the Mars capture spiral solver: time-optimal low-thrust '
            'spiral from the Mars sphere of influence to a 400-km LMO.  '
            'Use --quick for N=20 (fast demo), default is N=100.'
        ),
    )
    parser.add_argument(
        '--thrust-arrows',
        type=int,
        default=20,
        dest='thrust_arrows',
        metavar='N',
        help=(
            'Number of thrust-direction arrows to overlay on the Week 2 '
            'trajectory plot.  Arrows are evenly spaced in arc length; '
            'their length is proportional to the instantaneous thrust '
            'acceleration (normalised so the longest arrow has a fixed '
            'display size).  Default: 20.  Set to 0 to disable.'
        ),
    )
    parser.add_argument(
        '--pareto-idx',
        type=int,
        default=-1,
        dest='pareto_idx',
        metavar='IDX',
        help=(
            'Index (0-based) of the Pareto frontier point whose mass-optimal '
            'trajectory is plotted in Week 3.  Negative indices count from '
            'the end of the list sorted by TOF (e.g. -1 = longest TOF = most '
            'mass-efficient point).  Default: -1.'
        ),
    )
    return parser


def main() -> None:
    parser  = _build_parser()
    args    = parser.parse_args()

    if args.plot_only and args.save:
        parser.error('--plot-only and --save cannot be used together.')

    # ------------------------------------------------------------------
    # Spiral modes are independent of --week; run them and exit early
    # so that --week is not required when using spiral flags.
    # ------------------------------------------------------------------
    if args.escape_spiral or args.capture_spiral:
        mode_str = '  [--quick: N=20 segments]' if args.quick else ''
        print(f'\nEarth-Mars Low-Thrust Trajectory Optimization — Spiral Solver{mode_str}')
        print(f'Output directory: {RESULTS_DIR}')
        t_total = _time.perf_counter()
        if args.escape_spiral:
            run_escape_spiral(args)
        if args.capture_spiral:
            run_capture_spiral(args)
        total = _time.perf_counter() - t_total
        print()
        print('=' * 64)
        print(f'  Done.  Total wall time: {total:.1f} s  ({total/60:.1f} min)')
        print('=' * 64)
        return

    if args.week is None:
        parser.error('--week is required unless --escape_spiral or --capture_spiral is given.')

    weeks   = ['1', '2', '3', '4'] if args.week == 'all' else [args.week]

    if args.plot_only:
        mode_str = '  [--plot-only: loading saved results]'
    elif args.quick:
        mode_str = '  [--quick: N=20 segments]'
    else:
        mode_str = '  [production: N from MissionConfig]'
    if args.save:
        mode_str += '  [--save: results will be pickled]'

    print(f'\nEarth-Mars Low-Thrust Trajectory Optimization{mode_str}')
    print(f'Output directory: {RESULTS_DIR}')

    t_total = _time.perf_counter()
    results = {}

    runners = {
        '1': run_week1,
        '2': run_week2,
        '3': run_week3,
        '4': run_week4,
    }
    plot_runners = {
        '1': plot_week1,
        '2': plot_week2,
        '3': plot_week3,
        '4': plot_week4,
    }

    for week in weeks:
        t_week = _time.perf_counter()
        try:
            if args.plot_only:
                data = _load_result(week)
                plot_runners[week](data, args)
            else:
                results[week] = runners[week](args)
        except Exception as exc:          # pragma: no cover
            print(f'\n  ERROR in Week {week}: {exc}')
            import traceback
            traceback.print_exc()
        elapsed = _time.perf_counter() - t_week
        print(f'\n  Week {week} completed in {elapsed:.1f} s')

    total = _time.perf_counter() - t_total
    print()
    print('=' * 64)
    print(f'  Done.  Total wall time: {total:.1f} s  ({total/60:.1f} min)')
    print('=' * 64)


if __name__ == '__main__':
    main()
