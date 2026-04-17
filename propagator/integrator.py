"""Trajectory propagator for 2D heliocentric low-thrust optimization.

Integrates the equations of motion (core.equations_of_motion.eom_2d_ivp)
over piecewise-constant thrust arcs using scipy RK45, mirroring the role of
the MATLAB rkf78.m integrator in the reference implementation.

Piecewise-constant control transcription
-----------------------------------------
The total time of flight is divided into N equal segments of duration

    delta_t = tof_nd / N

Within each segment k the steering angle alpha_k is held constant.
State continuity across segment boundaries is enforced by using the terminal
state of segment k as the initial condition for segment k+1.

Units
-----
All times and states are non-dimensional (see core.nondimensional for the
normalization scheme).  The state vector throughout is

    y = [r, u, v, theta, mp, acc_dv]

where:
    r       : radial distance          [-]
    u       : radial velocity          [-]
    v       : transverse velocity      [-]
    theta   : polar angle              [rad]
    mp      : propellant mass fraction [-]
    acc_dv  : accumulated delta-V      [-]
"""

import numpy as np
from scipy.integrate import solve_ivp

from core.equations_of_motion import eom_2d_ivp


def propagate_segment(
    y0: np.ndarray,
    t_start: float,
    t_end: float,
    alpha: float,
    acc_thrust: float,
    mp_dot: float,
    throttle: float,
    tol: float = 1e-10,
) -> np.ndarray:
    """Integrate the EOM across a single constant-steering-angle segment.

    Uses scipy's RK45 adaptive integrator with tight tolerance settings
    to match the accuracy of the MATLAB RKF7(8) reference (rkf78.m).

    Parameters
    ----------
    y0 : array-like, length 6
        Initial state [r, u, v, theta, mp, acc_dv] at the start of the segment.
    t_start : float
        Non-dimensional start time of the segment [-].
    t_end : float
        Non-dimensional end time of the segment [-].
    alpha : float
        Thrust steering angle [rad], held constant over this segment.
        Measured from the transverse direction toward the outward radial
        (alpha=0 → prograde; alpha=pi/2 → outward radial).
    acc_thrust : float
        Reference non-dimensional thrust acceleration (T/m0 normalized) [-].
        Computed via NonDimensional.compute_thrust_acc(T_N, m0_kg).
    mp_dot : float
        Reference non-dimensional propellant mass-flow rate [-].
        Computed via NonDimensional.compute_mass_flow(T_N, Isp_s, m0_kg).
    throttle : float
        Throttle setting in [0, 1].
    tol : float, optional
        Both relative and absolute integration tolerance (default 1e-10).
        Tighter tolerances increase accuracy at the cost of more function
        evaluations.  The default matches the MATLAB reference.

    Returns
    -------
    y_final : np.ndarray, shape (6,)
        State vector at the end of the segment.

    Raises
    ------
    RuntimeError
        If solve_ivp reports integration failure (sol.success is False).
    """
    params = {
        'alpha':      alpha,
        'acc_thrust': acc_thrust,
        'mp_dot':     mp_dot,
        'throttle':   throttle,
    }

    sol = solve_ivp(
        fun=lambda t, y: eom_2d_ivp(t, y, params),
        t_span=(t_start, t_end),
        y0=np.asarray(y0, dtype=float),
        method='DOP853',
        rtol=tol,
        atol=tol,
        dense_output=False,
    )

    if not sol.success:
        raise RuntimeError(
            f"Integration failed over [{t_start:.6f}, {t_end:.6f}] "
            f"(alpha={alpha:.4f} rad, throttle={throttle}): {sol.message}"
        )

    return sol.y[:, -1]


def propagate_trajectory(
    y0: np.ndarray,
    tof_nd: float,
    alphas: np.ndarray,
    acc_thrust: float,
    mp_dot: float,
    throttle: float,
    tol: float = 1e-10,
) -> dict:
    """Propagate a full multi-segment low-thrust trajectory.

    Divides the total time of flight into N equal segments (N = len(alphas))
    and integrates sequentially, passing the terminal state of each segment
    as the initial condition for the next.

    Parameters
    ----------
    y0 : array-like, length 6
        Initial state [r, u, v, theta, mp, acc_dv].
        For an Earth departure on a circular orbit::

            y0 = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    tof_nd : float
        Total non-dimensional time of flight [-].
        Convert from days via NonDimensional.time_to_nd(tof_days).
    alphas : array-like, length N
        Steering angles [rad] for each of the N segments.
        alphas[k] is applied over the k-th segment
        [k * delta_t, (k+1) * delta_t].
    acc_thrust : float
        Reference non-dimensional thrust acceleration [-].
    mp_dot : float
        Reference non-dimensional propellant mass-flow rate [-].
    throttle : float
        Throttle setting in [0, 1], applied uniformly to all segments.
    tol : float, optional
        Integration tolerance (default 1e-10).

    Returns
    -------
    result : dict
        'final_state'  : np.ndarray, shape (6,)
            State at the end of the last segment.
        'state_history': np.ndarray, shape (N+1, 6)
            State at each node: row 0 = y0, row k = state after segment k-1.
        'time_history' : np.ndarray, shape (N+1,)
            Non-dimensional times of each node:
            [0, delta_t, 2*delta_t, ..., tof_nd].

    Notes
    -----
    N is inferred from len(alphas), so the number of control segments is
    determined entirely by the length of the alphas array.  A typical
    low-thrust optimization uses N in the range 20–200.
    """
    alphas = np.asarray(alphas, dtype=float)
    y0     = np.asarray(y0,     dtype=float)

    N       = len(alphas)
    delta_t = tof_nd / N

    # Pre-allocate output arrays
    time_history  = np.linspace(0.0, tof_nd, N + 1)  # avoids float accumulation
    state_history = np.empty((N + 1, 6), dtype=float)
    state_history[0] = y0

    y_current = y0.copy()

    for k in range(N):
        t_start = time_history[k]
        t_end   = time_history[k + 1]

        y_current = propagate_segment(
            y0        = y_current,
            t_start   = t_start,
            t_end     = t_end,
            alpha     = alphas[k],
            acc_thrust= acc_thrust,
            mp_dot    = mp_dot,
            throttle  = throttle,
            tol       = tol,
        )
        state_history[k + 1] = y_current

    return {
        'final_state':   y_current,
        'state_history': state_history,
        'time_history':  time_history,
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import numpy as np
    from core.nondimensional import NonDimensional
    from core.constants import r_mars

    print("=== integrator.py self-test ===\n")

    # ------------------------------------------------------------------
    # 1. Coasting circular Earth orbit: radial distance stays at 1.0
    # ------------------------------------------------------------------
    print("Test 1: coasting Earth orbit, N=20 segments, 1 full period")

    y0_earth     = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    one_period   = 2 * np.pi                  # 1 Keplerian year in ND time
    N_coast      = 20
    alphas_coast = np.zeros(N_coast)          # steering irrelevant (throttle=0)

    res = propagate_trajectory(
        y0         = y0_earth,
        tof_nd     = one_period,
        alphas     = alphas_coast,
        acc_thrust = 0.0,
        mp_dot     = 0.0,
        throttle   = 0.0,
    )

    r_f = res['final_state'][0]
    u_f = res['final_state'][1]
    v_f = res['final_state'][2]
    print(f"  state_history shape : {res['state_history'].shape}  (expect (21, 6))")
    print(f"  time_history[-1]    : {res['time_history'][-1]:.6f}  (expect {one_period:.6f})")
    print(f"  r after 1 orbit     : {r_f:.8f}  (expect 1.0)")
    print(f"  u after 1 orbit     : {u_f:.2e}   (expect ~0)")
    print(f"  v after 1 orbit     : {v_f:.8f}  (expect 1.0)")
    assert abs(r_f - 1.0) < 1e-7, f"r drift: {r_f - 1.0}"
    assert abs(u_f)       < 1e-7, f"u non-zero: {u_f}"
    assert abs(v_f - 1.0) < 1e-7, f"v drift: {v_f - 1.0}"
    print("  PASSED\n")

    # ------------------------------------------------------------------
    # 2. Node count and time grid sanity
    # ------------------------------------------------------------------
    print("Test 2: state_history and time_history dimensions for N=50")
    N_check = 50
    tof_nd  = NonDimensional.time_to_nd(300.0)   # 300-day transfer
    res2 = propagate_trajectory(
        y0         = y0_earth,
        tof_nd     = tof_nd,
        alphas     = np.zeros(N_check),
        acc_thrust = 0.0,
        mp_dot     = 0.0,
        throttle   = 0.0,
    )
    sh = res2['state_history']
    th = res2['time_history']
    print(f"  state_history shape : {sh.shape}  (expect ({N_check+1}, 6))")
    print(f"  time_history[0]     : {th[0]:.6f}  (expect 0.0)")
    print(f"  time_history[-1]    : {th[-1]:.6f}  (expect {tof_nd:.6f})")
    assert sh.shape == (N_check + 1, 6)
    assert th[0]  == 0.0
    assert abs(th[-1] - tof_nd) < 1e-12
    print("  PASSED\n")

    # ------------------------------------------------------------------
    # 3. Prograde thrust increases orbital energy (v grows, r grows)
    # ------------------------------------------------------------------
    print("Test 3: sustained prograde thrust raises orbit")
    T_N, Isp_s, m0_kg = 0.09, 3000.0, 500.0
    acc_thrust = NonDimensional.compute_thrust_acc(T_N, m0_kg)
    mp_dot_nd  = NonDimensional.compute_mass_flow(T_N, Isp_s, m0_kg)

    N_thrust  = 30
    tof_raise = NonDimensional.time_to_nd(180.0)          # 180-day burn
    alphas_pg = np.zeros(N_thrust)                        # all prograde

    res3 = propagate_trajectory(
        y0         = y0_earth,
        tof_nd     = tof_raise,
        alphas     = alphas_pg,
        acc_thrust = acc_thrust,
        mp_dot     = mp_dot_nd,
        throttle   = 1.0,
    )
    r_end  = res3['final_state'][0]
    mp_end = res3['final_state'][4]
    dv_end = res3['final_state'][5]
    print(f"  r after 180-day prograde burn : {r_end:.5f}  (expect > 1.0)")
    print(f"  propellant consumed (mp)      : {mp_end:.6f}")
    print(f"  accumulated delta-V (acc_dv)  : {dv_end:.6f}")
    assert r_end > 1.0,  "Orbit should expand under prograde thrust"
    assert mp_end > 0.0, "Propellant must be consumed"
    assert dv_end > 0.0, "Delta-V must accumulate"
    print("  PASSED")
