"""Time-optimal nonplanar low-thrust trajectory NLP solver.

Extends ``optimization/time_optimal.py`` to full 3D Cartesian coordinates.
Transfers are specified by classical orbital elements (SMA, inclination, RAAN)
for both the departure and target orbit; the optimizer finds the minimum
transfer time and the piecewise-constant steering angles (alpha, beta) that
satisfy five terminal orbital-element constraints.

Problem statement
-----------------
Minimise   J = tof_nd   (total non-dimensional transfer time)
subject to
    c1: |r_f| / r2  - 1              = 0   (arrive at target orbital radius)
    c2: dot(r_f, v_f) / (r2 * vc2)  = 0   (zero radial velocity, circular)
    c3: (|v_f| - vc2) / vc2          = 0   (circular speed at r2)
    c4: h_fz / |h_f| - cos(i2)       = 0   (correct inclination)
    c5: (h_fx*cos(RAAN2) + h_fy*sin(RAAN2)) / |h_f| = 0   (correct RAAN)

where h_f = r_f x v_f is the specific angular momentum at arrival,
vc2 = sqrt(mu* / r2) is the circular speed at the target radius.

Constraint c5 is derived from the orbit-normal direction:
    h_hat = [sin(i)*sin(RAAN), -sin(i)*cos(RAAN), cos(i)]
    => h_x*cos(RAAN) + h_y*sin(RAAN) = |h|*sin(i)*sin(RAAN-RAAN2) = 0
                                                        iff RAAN_f = RAAN2.

For equatorial target orbits (i2 ~ 0), c5 is automatically satisfied (both
h_fx and h_fy -> 0) and provides no additional constraint on the RAAN, which
is physically undefined for equatorial orbits.  The 5-constraint set reduces
to a 4-constraint set in this limit, leaving RAAN unconstrained.

Decision vector
---------------
x = [tof_nd, alpha_1, ..., alpha_N, beta_1, ..., beta_N]   length = 2*N + 1

    tof_nd     : non-dimensional transfer time (bounded)
    alpha_k    : in-plane azimuth for segment k [rad], in [-pi, pi]
    beta_k     : out-of-plane declination for segment k [rad], in [-pi/2, pi/2]

Initial state
-------------
Spacecraft is placed at the ascending node of the initial orbit:
    r_vec0 = r1 * [cos(RAAN1), sin(RAAN1), 0]
    v_vec0 = vc1 * [-cos(i1)*sin(RAAN1), cos(i1)*cos(RAAN1), sin(i1)]
where r1 = a1 (circular orbit), vc1 = sqrt(mu*/r1).

This coincides exactly with the 2D initial state [r=1, u=0, v=1, theta=0]
when i1 = 0 and RAAN1 = 0 (Earth's equatorial orbit).

NLP solver
----------
scipy.optimize.minimize with method='SLSQP'.  Gradients are approximated by
finite differences.  All 5 constraints are enforced as equality constraints.
"""

import time
import numpy as np
from scipy.optimize import minimize
from dataclasses import replace as _dc_replace

from config.mission_config import MissionConfig
from core.nondimensional import NonDimensional
from nonplanar.eom_3d import propagate_3d_trajectory, cartesian_to_orbital_elements

# ND gravitational parameter (mu* = 1 in AU/t_cf units)
_MU_ND: float = 1.0

# Guard value for angular momentum norm to avoid division by zero
_H_MIN: float = 1.0e-10

# ---------------------------------------------------------------------------
# Constants for multi-resolution warm-start
# ---------------------------------------------------------------------------
_N_COARSE_THRESHOLD_3D: int = 20   # use coarse→fine warm-start for N >= this
_N_COARSE_3D: int           = 10   # coarse grid size


def _initial_state_from_elements(
    a_au: float,
    i_deg: float,
    raan_deg: float,
) -> np.ndarray:
    """Build the length-8 ND initial state for a circular orbit.

    Places the spacecraft at the ascending node (argument of latitude = 0)
    of the circular orbit defined by (a, i, RAAN).

    Parameters
    ----------
    a_au : float
        Semi-major axis (= radius for circular orbit) [AU, ND].
    i_deg : float
        Inclination [deg].
    raan_deg : float
        Right ascension of ascending node [deg].

    Returns
    -------
    y0 : np.ndarray, shape (8,)
        [x, y_c, z, vx, vy, vz, 0, 0]  (mp=0, acc_dv=0 at departure)
    """
    i_r    = np.radians(i_deg)
    raan_r = np.radians(raan_deg)
    r      = a_au
    vc     = np.sqrt(_MU_ND / r)

    # Position: at ascending node
    rx = r * np.cos(raan_r)
    ry = r * np.sin(raan_r)
    rz = 0.0

    # Velocity: prograde circular velocity at ascending node
    # Derived from the perifocal -> inertial rotation matrix (omega=0, nu=0)
    vx = -vc * np.cos(i_r) * np.sin(raan_r)
    vy =  vc * np.cos(i_r) * np.cos(raan_r)
    vz =  vc * np.sin(i_r)

    return np.array([rx, ry, rz, vx, vy, vz, 0.0, 0.0])


def _shooting_3d(
    x: np.ndarray,
    y0: np.ndarray,
    r2: float,
    i2_rad: float,
    raan2_rad: float,
    nd: dict,
    throttle: float,
    N: int,
    integ_tol: float = 1e-8,
) -> dict:
    """3D shooting function: propagate and evaluate terminal constraints.

    Parameters
    ----------
    x : np.ndarray, length 2*N+1
        Decision vector [tof_nd, alpha_1..N, beta_1..N].
    y0 : np.ndarray, length 8
        Initial state vector.
    r2 : float
        Target orbital radius [AU, ND].
    i2_rad : float
        Target inclination [rad].
    raan2_rad : float
        Target RAAN [rad].
    nd : dict
        Non-dimensional propulsion params from ``config.to_nd_params()``.
    throttle : float
        Throttle setting (1.0 for time-optimal).
    N : int
        Number of shooting segments.

    Returns
    -------
    result : dict
        'objective'    : float — tof_nd (the quantity to minimise)
        'constraints'  : list of 5 floats — equality residuals
        'final_state'  : np.ndarray (8,) — state at arrival
        'state_history': np.ndarray (N+1, 8)
        'time_history' : np.ndarray (N+1,)
    """
    tof_nd = float(x[0])
    alphas = x[1:N + 1]
    betas  = x[N + 1:]

    vc2       = np.sqrt(_MU_ND / r2)
    cos_i2    = np.cos(i2_rad)
    cos_raan2 = np.cos(raan2_rad)
    sin_raan2 = np.sin(raan2_rad)

    # ------------------------------------------------------------------
    # Propagate
    # ------------------------------------------------------------------
    try:
        traj = propagate_3d_trajectory(
            y0, tof_nd, alphas, betas,
            nd['acc_thrust'], nd['mp_dot'], throttle,
            tol=integ_tol,
        )
    except RuntimeError:
        # Return large penalty values so the solver steers away
        penalty = 1.0e3
        dummy = {
            'objective':    1.0e6 * tof_nd,
            'constraints':  [penalty] * 5,
            'final_state':  y0.copy(),
            'state_history': np.tile(y0, (N + 1, 1)),
            'time_history':  np.linspace(0.0, tof_nd, N + 1),
        }
        return dummy

    yf    = traj['final_state']
    r_f   = yf[:3]
    v_f   = yf[3:6]
    r_f_m = np.linalg.norm(r_f)
    v_f_m = np.linalg.norm(v_f)
    h_f   = np.cross(r_f, v_f)
    h_f_m = max(np.linalg.norm(h_f), _H_MIN)

    # ------------------------------------------------------------------
    # Terminal constraints (all dimensionless O(1) quantities)
    # ------------------------------------------------------------------
    c1 = (r_f_m - r2) / r2                           # orbital radius
    c2 = float(np.dot(r_f, v_f)) / (r2 * vc2)        # no radial velocity
    c3 = (v_f_m - vc2) / vc2                          # circular speed
    c4 = h_f[2] / h_f_m - cos_i2                      # inclination
    c5 = (h_f[0] * cos_raan2 + h_f[1] * sin_raan2) / h_f_m  # RAAN

    return {
        'objective':    tof_nd,
        'constraints':  [c1, c2, c3, c4, c5],
        'final_state':  yf,
        'state_history': traj['state_history'],
        'time_history':  traj['time_history'],
    }


def _interpolate_coarse_3d(x_coarse: np.ndarray, N_fine: int) -> np.ndarray:
    """Linearly interpolate a coarse 3D solution to a finer segment grid.

    Parameters
    ----------
    x_coarse : ndarray, shape (2*N_coarse + 1,)
        Coarse solution [tof_nd, alpha_1..N_coarse, beta_1..N_coarse].
    N_fine : int
        Number of segments on the fine grid.

    Returns
    -------
    x_fine : ndarray, shape (2*N_fine + 1,)
        Interpolated vector [tof_nd, alpha_1..N_fine, beta_1..N_fine].
    """
    N_coarse = (len(x_coarse) - 1) // 2
    tof_nd   = x_coarse[0]
    alphas_c = x_coarse[1: N_coarse + 1]
    betas_c  = x_coarse[N_coarse + 1:]
    if N_coarse == N_fine:
        return x_coarse.copy()
    t_c      = np.linspace(0.0, 1.0, N_coarse)
    t_f      = np.linspace(0.0, 1.0, N_fine)
    alphas_f = np.interp(t_f, t_c, alphas_c)
    betas_f  = np.interp(t_f, t_c, betas_c)
    return np.concatenate([[tof_nd], alphas_f, betas_f])


def _jacobian_worker_3d(args):
    """Evaluate the 3D shooting function at one perturbed point.

    Module-level (not a closure) so it can be pickled by ProcessPoolExecutor.
    Returns the 5 constraint values; the objective gradient is analytic.
    """
    k, xp, y0, r2, i2_rad, raan2_rad, nd, throttle, N, integ_tol = args
    fp = _shooting_3d(xp, y0, r2, i2_rad, raan2_rad, nd, throttle, N,
                      integ_tol=integ_tol)
    return k, np.array(fp['constraints'])


def _run_slsqp_3d(
    N: int,
    nd: dict,
    x0: np.ndarray,
    y0: np.ndarray,
    r2: float,
    i2_rad: float,
    raan2_rad: float,
    throttle: float,
    tof_nd_lb: float,
    tof_nd_ub: float,
    alpha_lb_r: float,
    alpha_ub_r: float,
    integ_tol: float,
    maxiter: int,
    n_jobs: int = 1,
):
    """Run SLSQP for the 3D time-optimal NLP with a shared Jacobian.

    The objective J = tof_nd = x[0] has analytic gradient e_0 = [1, 0, ..., 0],
    eliminating finite-difference evaluation of the objective.  A single
    forward-difference sweep over the 2*N+1 decision variables builds all five
    constraint Jacobian rows simultaneously, reducing the per-iteration
    propagation count from 6*(2N+1) (scipy default) to 2N+1 (~6× reduction).

    When n_jobs > 1, the 2N+1 perturbed evaluations are distributed across
    worker processes via ProcessPoolExecutor for near-linear speedup.
    """
    _eps: float = max(1e-5, np.sqrt(integ_tol))
    _ec: dict   = {'x': None, 'res': None}
    _jc: dict   = {'x': None, 'J':   None}
    _nx: int    = 2 * N + 1

    _pool = None
    if n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        _pool = ProcessPoolExecutor(max_workers=n_jobs)

    def _ev(x: np.ndarray) -> dict:
        x = np.asarray(x, dtype=float)
        if _ec['x'] is None or not np.array_equal(x, _ec['x']):
            _ec['x']   = x.copy()
            _ec['res'] = _shooting_3d(
                x, y0, r2, i2_rad, raan2_rad, nd, throttle, N,
                integ_tol=integ_tol,
            )
        return _ec['res']

    def _cjac(x: np.ndarray) -> np.ndarray:
        """Return (5, 2N+1) constraint Jacobian rows via shared FD sweep."""
        x = np.asarray(x, dtype=float)
        if _jc['x'] is not None and np.array_equal(x, _jc['x']):
            return _jc['J']
        f0 = _ev(x)
        v0 = np.array(f0['constraints'])
        J  = np.empty((5, _nx), dtype=float)
        if _pool is not None:
            tasks = []
            for k in range(_nx):
                xp = x.copy(); xp[k] += _eps
                tasks.append((k, xp, y0, r2, i2_rad, raan2_rad,
                               nd, throttle, N, integ_tol))
            for k, vp in _pool.map(_jacobian_worker_3d, tasks):
                J[:, k] = (vp - v0) / _eps
        else:
            for k in range(_nx):
                xp      = x.copy()
                xp[k]  += _eps
                fp      = _shooting_3d(
                    xp, y0, r2, i2_rad, raan2_rad, nd, throttle, N,
                    integ_tol=integ_tol,
                )
                vp      = np.array(fp['constraints'])
                J[:, k] = (vp - v0) / _eps
        _jc['x'] = x.copy()
        _jc['J'] = J
        return J

    # Analytic objective gradient: J = tof_nd = x[0]  =>  grad_J = e_0
    _obj_grad = np.zeros(_nx)
    _obj_grad[0] = 1.0

    bounds = (
        [(tof_nd_lb, tof_nd_ub)]
        + [(alpha_lb_r, alpha_ub_r)] * N
        + [(-np.pi / 2.0, np.pi / 2.0)] * N
    )
    cons = [
        {
            'type': 'eq',
            'fun': lambda x, i=i: _ev(x)['constraints'][i],
            'jac': lambda x, i=i: _cjac(x)[i],
        }
        for i in range(5)
    ]
    try:
        return minimize(
            fun         = lambda x: _ev(x)['objective'],
            jac         = lambda x: _obj_grad.copy(),
            x0          = x0,
            method      = 'SLSQP',
            bounds      = bounds,
            constraints = cons,
            tol         = 1.0e-6,
            options     = {'maxiter': maxiter, 'ftol': 1.0e-6, 'disp': False},
        )
    finally:
        if _pool is not None:
            _pool.shutdown(wait=False)


def solve_nonplanar(
    config: MissionConfig,
    r1_elements: list,
    r2_elements: list,
    integ_tol: float = 1e-8,
    n_jobs: int = 1,
) -> dict:
    """Solve the time-optimal 3D nonplanar low-thrust NLP with SLSQP.

    Finds the minimum transfer time and piecewise-constant steering angles
    (alpha, beta per segment) for a heliocentric transfer between two circular
    orbits specified by their classical orbital elements.

    Parameters
    ----------
    config : MissionConfig
        Mission configuration.  Spacecraft parameters (``initial_mass_kg``,
        ``thrust_N``, ``isp_s``) and solver settings (``n_segments``,
        time bounds, ``alpha_*``) are used.  The throttle is fixed at
        ``config.throttle_guess`` (set to 1.0 for time-optimal mode).
    r1_elements : list or array-like, length 3
        Departure orbit: ``[a1_au, i1_deg, RAAN1_deg]``.

        * ``a1_au``: semi-major axis [AU] (= radius for circular orbit)
        * ``i1_deg``: inclination [deg]
        * ``RAAN1_deg``: RAAN [deg]

        Example — Earth equatorial: ``[1.0, 0.0, 0.0]``
    r2_elements : list or array-like, length 3
        Target orbit: ``[a2_au, i2_deg, RAAN2_deg]``.

        Example — Mars equatorial: ``[1.52368, 0.0, 0.0]``
        Example — Mars inclined  : ``[1.52368, 1.85, 49.6]``
    integ_tol : float, optional
        ODE integration tolerance (default 1e-8).  Still 100× tighter than
        the SLSQP constraint tolerance (1e-6).
    n_jobs : int, optional
        Number of parallel worker processes for the Jacobian FD sweep
        (default 1 = sequential).  Set to ``os.cpu_count()`` for maximum
        parallelism on a single solve.

    Returns
    -------
    result : dict
        'success'            : bool
        'tof_days'           : float  — optimal transfer time [days]
        'alphas_rad'         : np.ndarray (N,) — in-plane angles [rad]
        'betas_rad'          : np.ndarray (N,) — out-of-plane angles [rad]
        'propellant_mass_kg' : float  — propellant consumed [kg]
        'final_mass_kg'      : float  — remaining mass [kg]
        'state_history'      : np.ndarray (N+1, 8) — ND state at each node
        'time_history_days'  : np.ndarray (N+1,) — node times [days]
        'solver_message'     : str
        'n_iterations'       : int
        'wall_time_s'        : float

    Notes
    -----
    The initial state is placed at the ascending node of the departure orbit
    (argument of latitude = 0).  For equatorial orbits (i=0), this reduces
    to the standard 2D initial condition (spacecraft at [r, 0, 0] moving
    in the +y direction).

    The five terminal constraints enforce: correct orbital radius, zero
    radial velocity, correct circular speed, correct inclination, and
    correct RAAN.  The spacecraft's position along the target orbit at
    arrival is free (not constrained), analogous to the 2D case where the
    arrival longitude is free.

    For equatorial target orbits (i2 ~ 0 deg), the RAAN constraint c5
    becomes numerically degenerate (near zero regardless of RAAN).  This is
    physically correct: RAAN is undefined for equatorial orbits.

    The coplanar case (i1 = i2 = 0, RAAN1 = RAAN2 = 0) reduces exactly to
    the 2D time-optimal problem; results should match ``solve_time_optimal``
    to within ODE tolerances.

    Raises
    ------
    ValueError
        If ``r1_elements`` or ``r2_elements`` have incorrect length, or if
        orbital radii are non-positive.
    """
    # ------------------------------------------------------------------
    # Parse and validate inputs
    # ------------------------------------------------------------------
    r1_elements = list(r1_elements)
    r2_elements = list(r2_elements)
    if len(r1_elements) != 3 or len(r2_elements) != 3:
        raise ValueError(
            "r1_elements and r2_elements must each have length 3: "
            "[a_au, i_deg, RAAN_deg]"
        )
    a1_au, i1_deg, raan1_deg = float(r1_elements[0]), float(r1_elements[1]), float(r1_elements[2])
    a2_au, i2_deg, raan2_deg = float(r2_elements[0]), float(r2_elements[1]), float(r2_elements[2])

    if a1_au <= 0.0 or a2_au <= 0.0:
        raise ValueError(
            f"Orbital semi-major axes must be positive; "
            f"got a1={a1_au}, a2={a2_au}"
        )

    nd       = config.to_nd_params()
    N        = config.n_segments
    throttle = float(config.throttle_guess)   # 1.0 for time-optimal

    # ------------------------------------------------------------------
    # Initial state (departure orbit, at ascending node)
    # ------------------------------------------------------------------
    y0 = _initial_state_from_elements(a1_au, i1_deg, raan1_deg)

    # Target orbit parameters (ND)
    r2        = a2_au    # circular orbit: r = a
    i2_rad    = np.radians(i2_deg)
    raan2_rad = np.radians(raan2_deg)

    # ------------------------------------------------------------------
    # Decision vector: x = [tof_nd, alpha_1..N, beta_1..N]
    # ------------------------------------------------------------------
    tof_nd_lb  = nd['tof_lb_nd']
    tof_nd_ub  = nd['tof_ub_nd']
    tof_nd_0   = nd['tof_nd']

    alpha_lb_r, alpha_ub_r = config.get_alpha_bounds_rad()

    alpha_guess = np.full(N, config.alpha_guess_rad())

    # Beta initial guess: distribute the required inclination change uniformly.
    # A small floor (1e-3 rad ~ 0.057 deg) is applied so that betas are never
    # exactly zero at x0.  Exactly-zero betas make the trajectory coplanar,
    # which drives the constraint Jacobian rows for c4/c5 to zero in the
    # alpha and tof columns — causing a singular matrix in the SLSQP QP
    # subproblem.  The floor breaks this degeneracy without biasing the result.
    delta_i_rad = i2_rad - np.radians(i1_deg)
    beta_mag    = max(abs(delta_i_rad) / (2.0 * N), 1.0e-3)
    beta_guess  = np.full(N, np.sign(delta_i_rad) * beta_mag
                          if abs(delta_i_rad) > 1.0e-10 else beta_mag)

    # ------------------------------------------------------------------
    # Build initial guess
    # For large N, solve a fast coarse problem first and interpolate the
    # steering angles onto the fine grid.
    # ------------------------------------------------------------------
    t0 = time.perf_counter()

    if N >= _N_COARSE_THRESHOLD_3D:
        cfg_coarse = _dc_replace(config, n_segments=_N_COARSE_3D)
        nd_coarse  = cfg_coarse.to_nd_params()
        N_c        = _N_COARSE_3D
        delta_i_c  = delta_i_rad
        beta_mag_c = max(abs(delta_i_c) / (2.0 * N_c), 1.0e-3)
        beta_g_c   = np.full(N_c, np.sign(delta_i_c) * beta_mag_c
                             if abs(delta_i_c) > 1.0e-10 else beta_mag_c)
        x0_c = np.empty(2 * N_c + 1)
        x0_c[0]          = tof_nd_0
        x0_c[1:N_c + 1]  = np.full(N_c, config.alpha_guess_rad())
        x0_c[N_c + 1:]   = beta_g_c
        res_coarse = _run_slsqp_3d(
            N_c, nd_coarse, x0_c, y0, r2, i2_rad, raan2_rad, throttle,
            nd_coarse['tof_lb_nd'], nd_coarse['tof_ub_nd'],
            alpha_lb_r, alpha_ub_r, integ_tol, maxiter=500,
        )
        x0 = _interpolate_coarse_3d(res_coarse.x, N)
    else:
        x0 = np.empty(2 * N + 1)
        x0[0]        = tof_nd_0
        x0[1:N + 1]  = alpha_guess
        x0[N + 1:]   = beta_guess

    # ------------------------------------------------------------------
    # Fine-grid solve with shared Jacobian (parallel columns when n_jobs > 1)
    # ------------------------------------------------------------------
    opt_result = _run_slsqp_3d(
        N, nd, x0, y0, r2, i2_rad, raan2_rad, throttle,
        tof_nd_lb, tof_nd_ub, alpha_lb_r, alpha_ub_r,
        integ_tol, maxiter=2000, n_jobs=n_jobs,
    )

    wall_time = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Post-process
    # ------------------------------------------------------------------
    x_opt      = opt_result.x
    tof_nd_opt = float(x_opt[0])
    alphas_opt = x_opt[1:N + 1]
    betas_opt  = x_opt[N + 1:]

    # Re-evaluate at the final solution (gradient perturbations may have been last)
    final_eval = _shooting_3d(x_opt, y0, r2, i2_rad, raan2_rad, nd, throttle, N,
                               integ_tol=integ_tol)
    yf         = final_eval['final_state']

    tof_days           = NonDimensional.time_to_dim(tof_nd_opt)
    propellant_mass_kg = float(yf[6]) * config.initial_mass_kg
    final_mass_kg      = config.initial_mass_kg * (1.0 - float(yf[6]))
    time_history_days  = NonDimensional.time_to_dim(final_eval['time_history'])

    tol_check  = 1.0e-4   # slightly relaxed: 3D is harder to converge tightly
    c          = final_eval['constraints']
    cstr_ok    = all(abs(ci) < tol_check for ci in c)

    return {
        'success':             bool(opt_result.success) and cstr_ok,
        'tof_days':            tof_days,
        'alphas_rad':          alphas_opt,
        'betas_rad':           betas_opt,
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

    print('=== nonplanar_optimizer.py self-test ===\n')

    # ------------------------------------------------------------------
    # 1. _initial_state_from_elements: equatorial Earth reduces to 2D case
    # ------------------------------------------------------------------
    print('Test 1: initial state for Earth equatorial orbit = 2D initial state')
    from ephemeris.planetary_states import get_initial_state_earth
    y0_3d = _initial_state_from_elements(1.0, 0.0, 0.0)
    y0_2d = get_initial_state_earth()   # [r, u, v, theta, mp, acc_dv] in polar
    # Cartesian [x,y,z,vx,vy,vz] should be [1,0,0,0,1,0]
    print(f"  3D state: {y0_3d}")
    assert abs(y0_3d[0] - 1.0) < 1e-14, f"x != 1.0: {y0_3d[0]}"
    assert abs(y0_3d[1] - 0.0) < 1e-14, f"y != 0.0: {y0_3d[1]}"
    assert abs(y0_3d[2] - 0.0) < 1e-14, f"z != 0.0: {y0_3d[2]}"
    assert abs(y0_3d[3] - 0.0) < 1e-14, f"vx != 0.0: {y0_3d[3]}"
    assert abs(y0_3d[4] - 1.0) < 1e-14, f"vy != 1.0: {y0_3d[4]}"
    assert abs(y0_3d[5] - 0.0) < 1e-14, f"vz != 0.0: {y0_3d[5]}"
    print('  PASSED\n')

    # ------------------------------------------------------------------
    # 2. _initial_state_from_elements: inclined orbit OE round-trip
    # ------------------------------------------------------------------
    print('Test 2: inclined orbit initial state -> OE round-trip')
    y0_inc = _initial_state_from_elements(1.52368, 10.0, 60.0)
    oe = cartesian_to_orbital_elements(y0_inc[:3], y0_inc[3:6])
    print(f"  a         = {oe['a']:.5f}  (expect 1.52368)")
    print(f"  e         = {oe['e']:.2e}   (expect ~0)")
    print(f"  i_deg     = {oe['i_deg']:.4f}  (expect 10.0)")
    print(f"  raan_deg  = {oe['raan_deg']:.4f}  (expect 60.0)")
    assert abs(oe['a']        - 1.52368) < 1e-8
    assert oe['e']                        < 1e-8
    assert abs(oe['i_deg']    - 10.0)   < 1e-5
    assert abs(oe['raan_deg'] - 60.0)   < 1e-5
    print('  PASSED\n')

    # ------------------------------------------------------------------
    # 3. Input validation
    # ------------------------------------------------------------------
    print('Test 3: input validation')
    cfg_valid = MissionConfig(
        opt_mode        = 'time_optimal',
        n_segments      = 3,
        time_guess_days = 200.0,
        time_lb_days    = 175.0,
        time_ub_days    = 225.0,
        throttle_guess  = 1.0,
        throttle_lb     = 1.0,
        throttle_ub     = 1.0,
    )
    try:
        solve_nonplanar(cfg_valid, [1.0], [1.52368, 0.0, 0.0])
        assert False, 'Should have raised ValueError'
    except ValueError as e:
        print(f"  Caught bad r1_elements length: {e}")

    try:
        solve_nonplanar(cfg_valid, [-1.0, 0.0, 0.0], [1.52368, 0.0, 0.0])
        assert False, 'Should have raised ValueError'
    except ValueError as e:
        print(f"  Caught non-positive a1: {e}")
    print('  PASSED\n')

    # ------------------------------------------------------------------
    # 4. Coplanar solve: N=5 quick test (should resemble 2D time-optimal)
    # ------------------------------------------------------------------
    print('Test 4: coplanar transfer (i=0, RAAN=0) — N=5 quick solve')
    cfg = MissionConfig(
        initial_mass_kg      = 5000.0,
        thrust_N             = 3.5,
        isp_s                = 3000.0,
        opt_mode             = 'time_optimal',
        n_segments           = 5,
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

    r1_elem = [1.0,     0.0, 0.0]   # Earth equatorial
    r2_elem = [1.52368, 0.0, 0.0]   # Mars equatorial

    print("  Running SLSQP (N=5, coplanar) ...")
    res = solve_nonplanar(cfg, r1_elem, r2_elem)

    print(f"\n  Solver message : {res['solver_message']}")
    print(f"  Wall time      : {res['wall_time_s']:.2f} s")
    print(f"  N iterations   : {res['n_iterations']}")
    print(f"  success        : {res['success']}")
    print(f"  tof_days       : {res['tof_days']:.2f} d  (expect ~175-225 d)")
    print(f"  propellant     : {res['propellant_mass_kg']:.2f} kg")
    print(f"  final mass     : {res['final_mass_kg']:.2f} kg")
    print(f"  betas range    : [{res['betas_rad'].min():.4f}, "
          f"{res['betas_rad'].max():.4f}] rad  (expect ~0 for coplanar)")

    required_keys = {
        'success', 'tof_days', 'alphas_rad', 'betas_rad',
        'propellant_mass_kg', 'final_mass_kg',
        'state_history', 'time_history_days',
        'solver_message', 'n_iterations', 'wall_time_s',
    }
    assert required_keys.issubset(res.keys()), \
        f"Missing keys: {required_keys - res.keys()}"

    N5 = cfg.n_segments
    assert res['alphas_rad'].shape == (N5,),      f"alphas shape: {res['alphas_rad'].shape}"
    assert res['betas_rad'].shape  == (N5,),      f"betas shape:  {res['betas_rad'].shape}"
    assert res['state_history'].shape == (N5+1, 8), f"history shape: {res['state_history'].shape}"

    assert 150.0 < res['tof_days'] < 250.0, \
        f"TOF outside expected range: {res['tof_days']:.2f} d"
    assert res['propellant_mass_kg'] >= 0,    "Negative propellant"
    assert res['final_mass_kg'] > 0,          "Zero final mass"
    mass_err = abs(res['propellant_mass_kg'] + res['final_mass_kg'] - cfg.initial_mass_kg)
    assert mass_err < 1e-6,                   f"Mass budget violated: {mass_err:.2e}"

    # For coplanar case, betas should be near zero
    beta_max = np.max(np.abs(res['betas_rad']))
    print(f"  max |beta|     : {np.degrees(beta_max):.3f} deg  "
          f"(expect < 5 deg for coplanar)")
    print('  PASSED\n')

    # ------------------------------------------------------------------
    # 5. Nonplanar solve: N=5, small inclination change (2 deg)
    # ------------------------------------------------------------------
    print('Test 5: nonplanar transfer — 2 deg inclination change, N=5')
    r2_inclined = [1.52368, 2.0, 0.0]   # Mars with 2 deg inclination

    print("  Running SLSQP (N=5, 2 deg inclination change) ...")
    res_np = solve_nonplanar(cfg, r1_elem, r2_inclined)

    print(f"\n  Solver message : {res_np['solver_message']}")
    print(f"  success        : {res_np['success']}")
    print(f"  tof_days       : {res_np['tof_days']:.2f} d")
    print(f"  max |beta|     : {np.degrees(np.max(np.abs(res_np['betas_rad']))):.2f} deg  "
          f"(expect > 0 for nonplanar)")

    # Check keys and shapes (same requirement as above)
    assert required_keys.issubset(res_np.keys())
    assert res_np['alphas_rad'].shape == (N5,)
    assert res_np['betas_rad'].shape  == (N5,)

    # Nonplanar case should require some out-of-plane thrust
    beta_max_np = np.max(np.abs(res_np['betas_rad']))
    if res_np['success']:
        print(f"  Nonplanar: max |beta| = {np.degrees(beta_max_np):.3f} deg  (expect > 0)")
    print('  PASSED')
