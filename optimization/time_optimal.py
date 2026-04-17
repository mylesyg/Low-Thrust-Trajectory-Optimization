"""Time-optimal Earth-Mars low-thrust trajectory NLP solver.

Analogous to ilt_opt_snopt.m with iopt=1 in the MATLAB reference.

Problem statement
-----------------
Minimise   J = acc_dv (accumulated delta-V = yfinal[5])
subject to r_f  = r_mars        (arrive at Mars orbital radius)
           u_f  = 0             (circular orbit, zero radial velocity)
           v_f  = v_mars_circ   (circular transverse speed at Mars)
           tof_lb <= tof_nd <= tof_ub
           alpha_lb <= alpha_k <= alpha_ub   for k = 1 .. N

For fixed full throttle (throttle=1), minimising acc_dv is equivalent to
minimising transfer time because the thrust acceleration is always positive
and monotonically increasing (mass decreases, so T/m grows).

Decision vector
---------------
x = [tof_nd, alpha_1, alpha_2, ..., alpha_N]   length = N + 1

NLP solver
----------
scipy.optimize.minimize with method='SLSQP' (Sequential Least Squares
Programming).  Gradients are approximated by finite differences (default);
no analytic Jacobian is provided.

Performance notes
-----------------
Two mechanisms keep N=200 runs tractable:

1. Shared Jacobian (_run_slsqp): a single forward-difference sweep over
   the N+1 decision variables builds all four Jacobian rows (objective +
   3 constraints) simultaneously.  Cost per SLSQP iteration drops from
   4*(N+1) ≈ 804 propagations to N+1 ≈ 201 (~4× reduction).

2. Multi-resolution warm-start: for N >= _N_COARSE_THRESHOLD the solver
   first converges a coarse (N=_N_COARSE=25) problem and linearly
   interpolates the result onto the fine grid.  The coarse solve is ~64×
   cheaper in Jacobian cost and reduces fine-grid iteration count from
   ~80-100 (cold) to ~10-20 (warm).
"""

import os
import time
import numpy as np
from scipy.optimize import minimize
from dataclasses import replace as _dc_replace

from config.mission_config import MissionConfig
from core.nondimensional import NonDimensional
from optimization.shooting import shooting_function

# ---------------------------------------------------------------------------
# Constants for multi-resolution warm-start
# ---------------------------------------------------------------------------
_N_COARSE_THRESHOLD: int = 50   # use coarse→fine warm-start for N >= this
_N_COARSE: int           = 25   # coarse grid size


def _interpolate_coarse_solution(x_coarse: np.ndarray, N_fine: int) -> np.ndarray:
    """Linearly interpolate a coarse steering solution to a finer segment grid.

    Parameters
    ----------
    x_coarse : ndarray, shape (N_coarse + 1,)
        Coarse solution [tof_nd, alpha_1, ..., alpha_N_coarse].
    N_fine : int
        Number of segments on the fine grid.

    Returns
    -------
    x_fine : ndarray, shape (N_fine + 1,)
        Interpolated vector [tof_nd, alpha_1_fine, ..., alpha_N_fine].
    """
    tof_nd        = x_coarse[0]
    alphas_coarse = x_coarse[1:]
    N_coarse      = len(alphas_coarse)
    if N_coarse == N_fine:
        return x_coarse.copy()
    t_c         = np.linspace(0.0, 1.0, N_coarse)
    t_f         = np.linspace(0.0, 1.0, N_fine)
    alphas_fine = np.interp(t_f, t_c, alphas_coarse)
    return np.concatenate([[tof_nd], alphas_fine])


def _jacobian_worker(args):
    """Evaluate the time-optimal shooting function at one perturbed point.

    Module-level (not a closure) so it can be pickled by ProcessPoolExecutor.
    """
    k, xp, nd, integ_tol = args
    fp = shooting_function(xp, nd, mode='time_optimal', tol=integ_tol)
    return k, np.array([fp['objective']] + fp['constraints'])


def _run_slsqp(
    N: int,
    nd: dict,
    x0: np.ndarray,
    alpha_lb: float,
    alpha_ub: float,
    integ_tol: float,
    maxiter: int,
    n_jobs: int = 1,
):
    """Run one SLSQP stage for the time-optimal NLP with a shared Jacobian.

    A single forward-difference sweep over the N+1 decision variables builds
    all four Jacobian rows (objective + 3 constraints) from N+1 trajectory
    propagations.  Without sharing, SLSQP would compute each row separately,
    costing 4*(N+1) propagations per iteration.

    When n_jobs > 1, the N+1 perturbed propagations are distributed across
    worker processes via ProcessPoolExecutor, giving near-linear speedup on
    multi-core hardware.

    The FD step is chosen as sqrt(integ_tol) — the classical optimal step for
    a function with noise level ~ integ_tol.
    """
    _eps: float = max(1e-5, np.sqrt(integ_tol))
    _ec: dict   = {'x': None, 'res': None}   # evaluation cache
    _jc: dict   = {'x': None, 'J':   None}   # Jacobian cache

    _pool = None
    if n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        _pool = ProcessPoolExecutor(max_workers=n_jobs)

    def _ev(x: np.ndarray) -> dict:
        x = np.asarray(x, dtype=float)
        if _ec['x'] is None or not np.array_equal(x, _ec['x']):
            _ec['x']   = x.copy()
            _ec['res'] = shooting_function(x, nd, mode='time_optimal', tol=integ_tol)
        return _ec['res']

    def _jac(x: np.ndarray) -> np.ndarray:
        """Return (4, N+1) Jacobian rows: [obj_grad, c0_grad, c1_grad, c2_grad]."""
        x = np.asarray(x, dtype=float)
        if _jc['x'] is not None and np.array_equal(x, _jc['x']):
            return _jc['J']
        f0 = _ev(x)
        v0 = np.array([f0['objective']] + f0['constraints'])
        J  = np.empty((4, len(x)), dtype=float)
        if _pool is not None:
            tasks = []
            for k in range(len(x)):
                xp = x.copy(); xp[k] += _eps
                tasks.append((k, xp, nd, integ_tol))
            for k, vp in _pool.map(_jacobian_worker, tasks):
                J[:, k] = (vp - v0) / _eps
        else:
            for k in range(len(x)):
                xp      = x.copy()
                xp[k]  += _eps
                fp      = shooting_function(xp, nd, mode='time_optimal', tol=integ_tol)
                vp      = np.array([fp['objective']] + fp['constraints'])
                J[:, k] = (vp - v0) / _eps
        _jc['x'] = x.copy()
        _jc['J'] = J
        return J

    bounds = [(nd['tof_lb_nd'], nd['tof_ub_nd'])] + [(alpha_lb, alpha_ub)] * N
    cons   = [
        {
            'type': 'eq',
            'fun': lambda x, i=i: _ev(x)['constraints'][i],
            'jac': lambda x, i=i: _jac(x)[i + 1],
        }
        for i in range(3)
    ]
    try:
        return minimize(
            fun         = lambda x: _ev(x)['objective'],
            jac         = lambda x: _jac(x)[0],
            x0          = x0,
            method      = 'SLSQP',
            bounds      = bounds,
            constraints = cons,
            tol         = 1e-6,
            options     = {'maxiter': maxiter, 'ftol': 1e-6, 'disp': False},
        )
    finally:
        if _pool is not None:
            _pool.shutdown(wait=False)


def solve_time_optimal(
    config: MissionConfig,
    x0_override: np.ndarray = None,
    integ_tol: float = 1e-8,
    maxiter: int = 1000,
    n_jobs: int = 1,
) -> dict:
    """Solve the time-optimal Earth-Mars low-thrust NLP with SLSQP.

    Parameters
    ----------
    config : MissionConfig
        Fully populated mission configuration.  Must have
        ``opt_mode='time_optimal'`` and ``throttle_lb == throttle_ub == 1.0``
        (full thrust throughout).

    Returns
    -------
    result : dict
        'success'          : bool
            True if SLSQP reported convergence and all terminal constraints
            are satisfied within ``tol``.
        'tof_days'         : float
            Optimal transfer time [days].
        'alphas_rad'       : np.ndarray, shape (N,)
            Optimal per-segment steering angles [rad].
        'propellant_mass_kg' : float
            Propellant consumed = yfinal[4] * initial_mass_kg [kg].
        'final_mass_kg'    : float
            Remaining spacecraft mass [kg].
        'state_history'    : np.ndarray, shape (N+1, 6)
            State vector at every shooting node (ND units).
        'time_history_days': np.ndarray, shape (N+1,)
            Node times converted to days.
        'solver_message'   : str
            Human-readable SLSQP exit message.
        'n_iterations'     : int
            Number of SLSQP major iterations.
        'wall_time_s'      : float
            Elapsed wall-clock time [s].

    Notes
    -----
    The objective is the accumulated delta-V ``yfinal[5]``.  Because throttle
    is fixed at 1 throughout, this is monotonically proportional to the
    transfer time, making the two objectives equivalent.

    Constraint satisfaction is checked at ``tol=1e-6``; values below this
    threshold are considered converged.
    """
    if config.opt_mode != 'time_optimal':
        raise ValueError(
            f"solve_time_optimal requires opt_mode='time_optimal', "
            f"got {config.opt_mode!r}.  Use solve_mass_optimal for mass-optimal."
        )

    nd             = config.to_nd_params()
    N              = config.n_segments
    alpha_lb, alpha_ub = config.get_alpha_bounds_rad()
    alpha_guess    = config.alpha_guess_rad()

    # ------------------------------------------------------------------
    # Build initial guess
    # For large N without an external warm start, solve a fast coarse
    # problem first and interpolate the result onto the fine grid.  The
    # coarse solve at N=_N_COARSE is ~(N/_N_COARSE)^2 cheaper in Jacobian
    # cost and places the initial guess in the correct basin, cutting
    # fine-grid SLSQP iterations from ~80-100 (cold) down to ~10-20.
    # ------------------------------------------------------------------
    t0 = time.perf_counter()

    if x0_override is not None and len(x0_override) == N + 1:
        x0 = np.asarray(x0_override, dtype=float)
    elif N >= _N_COARSE_THRESHOLD:
        cfg_coarse = _dc_replace(config, n_segments=_N_COARSE)
        nd_coarse  = cfg_coarse.to_nd_params()
        x0_c       = np.empty(_N_COARSE + 1)
        x0_c[0]    = nd_coarse['tof_nd']
        x0_c[1:]   = alpha_guess
        res_coarse = _run_slsqp(_N_COARSE, nd_coarse, x0_c,
                                alpha_lb, alpha_ub, integ_tol, maxiter=500)
        x0 = _interpolate_coarse_solution(res_coarse.x, N)
    else:
        x0      = np.empty(N + 1)
        x0[0]   = nd['tof_nd']
        x0[1:]  = alpha_guess

    # ------------------------------------------------------------------
    # Fine-grid solve with shared Jacobian (parallel columns when n_jobs > 1)
    # ------------------------------------------------------------------
    opt_result = _run_slsqp(N, nd, x0, alpha_lb, alpha_ub, integ_tol, maxiter,
                             n_jobs=n_jobs)

    wall_time = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Post-process optimal solution
    # ------------------------------------------------------------------
    x_opt       = opt_result.x
    tof_nd_opt  = float(x_opt[0])
    alphas_opt  = x_opt[1:]

    # Re-evaluate at x_opt to get the final trajectory
    # (the cache may not hold x_opt if the last SLSQP evaluation was a
    # gradient perturbation at a slightly different point)
    final_eval  = shooting_function(x_opt, nd, mode='time_optimal', tol=integ_tol)
    yfinal      = final_eval['final_state']

    tof_days           = NonDimensional.time_to_dim(tof_nd_opt)
    propellant_mass_kg = float(yfinal[4]) * config.initial_mass_kg
    final_mass_kg      = config.initial_mass_kg * (1.0 - float(yfinal[4]))
    time_history_days  = NonDimensional.time_to_dim(final_eval['time_history'])

    # Constraint residuals at optimal solution
    c_residuals = final_eval['constraints']
    tol         = 1e-6
    constraints_ok = all(abs(c) < tol for c in c_residuals)

    return {
        'success':             bool(opt_result.success) and constraints_ok,
        'tof_days':            tof_days,
        'alphas_rad':          alphas_opt,
        'propellant_mass_kg':  propellant_mass_kg,
        'final_mass_kg':       final_mass_kg,
        'state_history':       final_eval['state_history'],
        'time_history_days':   time_history_days,
        'solver_message':      opt_result.message,
        'n_iterations':        opt_result.get('nit', -1),
        'wall_time_s':         wall_time,
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import numpy as np
    from config.mission_config import MissionConfig

    print('=== time_optimal.py self-test ===\n')

    # ------------------------------------------------------------------
    # Use a very small N for speed.  The solution will be coarse but the
    # interface, shapes, and types must all be correct.
    # ------------------------------------------------------------------
    cfg = MissionConfig(
        initial_mass_kg      = 5000.0,
        thrust_N             = 3.5,
        isp_s                = 3000.0,
        opt_mode             = 'time_optimal',
        n_segments           = 5,            # tiny for test speed
        time_guess_days      = 200.0,
        time_lb_days         = 100.0,
        time_ub_days         = 350.0,
        throttle_guess       = 1.0,
        throttle_lb          = 1.0,
        throttle_ub          = 1.0,
        alpha_guess_deg      = 0.0,          # pure prograde initial guess
        alpha_lb_deg         = -180.0,
        alpha_ub_deg         =  180.0,
        departure_date_guess = '2020-01-01',
    )

    print(f"  n_segments  : {cfg.n_segments}")
    print(f"  thrust_N    : {cfg.thrust_N} N")
    print(f"  mass        : {cfg.initial_mass_kg} kg")
    print(f"  TOF window  : [{cfg.time_lb_days}, {cfg.time_ub_days}] days\n")
    print("  Running SLSQP (N=5 coarse solve) ...")

    res = solve_time_optimal(cfg)

    print(f"\n  Solver message  : {res['solver_message']}")
    print(f"  Wall time       : {res['wall_time_s']:.2f} s")
    print(f"  N iterations    : {res['n_iterations']}")
    print()

    # ------------------------------------------------------------------
    # Structural checks (independent of convergence)
    # ------------------------------------------------------------------
    print('Test 1: result dict has all required keys')
    required_keys = {
        'success', 'tof_days', 'alphas_rad', 'propellant_mass_kg',
        'final_mass_kg', 'state_history', 'time_history_days',
        'solver_message', 'n_iterations', 'wall_time_s',
    }
    assert required_keys.issubset(res.keys()), \
        f"Missing keys: {required_keys - res.keys()}"
    print('  PASSED\n')

    print('Test 2: array shapes')
    N = cfg.n_segments
    assert res['alphas_rad'].shape       == (N,),      f"alphas shape wrong: {res['alphas_rad'].shape}"
    assert res['state_history'].shape    == (N+1, 6),  f"state_history shape wrong"
    assert res['time_history_days'].shape == (N+1,),   f"time_history_days shape wrong"
    print(f"  alphas_rad shape      : {res['alphas_rad'].shape}      (expect ({N},))")
    print(f"  state_history shape   : {res['state_history'].shape}  (expect ({N+1}, 6))")
    print(f"  time_history_days[-1] : {res['time_history_days'][-1]:.2f} days")
    print('  PASSED\n')

    print('Test 3: physical sanity')
    assert res['tof_days'] > 0,                              "TOF must be positive"
    assert 100.0 <= res['tof_days'] <= 350.0,               f"TOF out of bounds: {res['tof_days']:.1f}"
    assert res['propellant_mass_kg'] >= 0,                   "Propellant consumed must be >= 0"
    assert res['final_mass_kg'] <= cfg.initial_mass_kg,      "Cannot gain mass"
    assert abs(res['propellant_mass_kg'] + res['final_mass_kg']
               - cfg.initial_mass_kg) < 1e-6,               "Mass budget violated"
    assert res['time_history_days'][0] == 0.0,               "First node must be t=0"
    print(f"  tof_days              : {res['tof_days']:.2f}")
    print(f"  propellant_mass_kg    : {res['propellant_mass_kg']:.2f} kg")
    print(f"  final_mass_kg         : {res['final_mass_kg']:.2f} kg")
    print(f"  mass conservation err : {abs(res['propellant_mass_kg']+res['final_mass_kg']-cfg.initial_mass_kg):.2e} kg")
    print('  PASSED\n')

    print('Test 4: wrong opt_mode raises ValueError')
    cfg_bad = MissionConfig(opt_mode='mass_optimal',
                            time_lb_days=200.0, time_guess_days=200.0, time_ub_days=200.0)
    try:
        solve_time_optimal(cfg_bad)
        assert False, 'Should have raised ValueError'
    except ValueError as e:
        print(f"  Caught expected error: {e}")
    print('  PASSED\n')

    # ------------------------------------------------------------------
    # Print terminal constraint residuals at the solution
    # ------------------------------------------------------------------
    from optimization.shooting import shooting_function
    nd   = cfg.to_nd_params()
    x_opt = np.concatenate([[NonDimensional.time_to_nd(res['tof_days'])],
                             res['alphas_rad']])
    ev   = shooting_function(x_opt, nd, mode='time_optimal')
    c    = ev['constraints']
    print(f"Terminal constraint residuals at solution:")
    print(f"  r_f - r_mars   = {c[0]:.4e}  (target 0)")
    print(f"  u_f            = {c[1]:.4e}  (target 0)")
    print(f"  v_f - v_circ   = {c[2]:.4e}  (target 0)")
    print(f"\n  success = {res['success']}")
    if not res['success']:
        print("  (N=5 too coarse to converge — increase n_segments for real runs)")
