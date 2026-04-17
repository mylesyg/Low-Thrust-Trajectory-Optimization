"""Core constraint evaluation (shooting function) for low-thrust NLP.

Analogous to ilt_shoot.m in the MATLAB reference.  Called at every NLP
function evaluation during the SLSQP solve.

Decision vector layout
----------------------
mode='time_optimal'
    x = [tof_nd, alpha_1, alpha_2, ..., alpha_N]
    tof_nd  : non-dimensional transfer time (free variable)
    alpha_k : steering angle for segment k [rad] (free variable)
    throttle: fixed at 1.0 (full thrust throughout)

mode='mass_optimal'
    x = [throttle, alpha_1, alpha_2, ..., alpha_N]
    throttle: uniform throttle in [0, 1] (free variable)
    alpha_k : steering angle for segment k [rad] (free variable)
    tof_nd  : fixed at config_nd['tof_nd']

Terminal constraints
--------------------
Three scalar equality constraints enforce arrival at the Mars circular orbit:
    c0 = r_f - r_mars     = 0   (correct orbital radius)
    c1 = u_f - 0          = 0   (zero radial velocity)
    c2 = v_f - v_mars_circ= 0   (circular transverse velocity)

The polar angle theta_f is unconstrained (free final angle).
"""

import numpy as np

from core.equations_of_motion import terminal_conditions_mars
from ephemeris.planetary_states import get_initial_state_earth
from propagator.integrator import propagate_trajectory

# ---------------------------------------------------------------------------
# Mars terminal conditions — computed once at import time
# ---------------------------------------------------------------------------
_TC = terminal_conditions_mars()
_R_MARS: float = _TC['r_f']   # 1.52368 AU (ND)
_U_MARS: float = _TC['u_f']   # 0.0
_V_MARS: float = _TC['v_f']   # sqrt(1 / r_mars) ~ 0.8101

# Penalty returned when integration fails (steers solver away from the region)
_PENALTY: float = 1.0e6

_VALID_MODES = frozenset({'time_optimal', 'mass_optimal'})


def shooting_function(
    x: np.ndarray,
    config_nd: dict,
    mode: str,
    tol: float = 1e-10,
) -> dict:
    """Propagate the trajectory and return objective + terminal constraints.

    Parameters
    ----------
    x : array-like, length N+1
        NLP decision vector.  Layout depends on ``mode``; see module docstring.
    config_nd : dict
        Non-dimensional propulsion and time parameters from
        ``MissionConfig.to_nd_params()``.  Required keys:

        'acc_thrust' : reference ND thrust acceleration [-]
        'mp_dot'     : reference ND mass-flow rate [-]
        'tof_nd'     : nominal ND time of flight (used as fixed TOF in
                        mass-optimal mode) [-]

    mode : str
        'time_optimal' or 'mass_optimal'.

    Returns
    -------
    result : dict
        'objective'    : float
            Accumulated delta-V ``yfinal[5]`` for time-optimal;
            throttle setting ``x[0]`` for mass-optimal.
        'constraints'  : list of 3 floats
            Terminal residuals [r_f - r_mars, u_f, v_f - v_mars_circ].
            Each should be zero at the optimum.
        'final_state'  : np.ndarray, shape (6,)
            State vector at end of transfer [r, u, v, theta, mp, acc_dv].
        'state_history': np.ndarray, shape (N+1, 6)
            State at every shooting node including departure.
        'time_history' : np.ndarray, shape (N+1,)
            Non-dimensional time at every node.

    Notes
    -----
    If the ODE integrator raises a ``RuntimeError`` (e.g. step-size too
    small, singular mass fraction), the function returns ``_PENALTY`` for
    the objective and all three constraint residuals.  This guides SLSQP
    away from the infeasible region without crashing the solver.
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")

    x      = np.asarray(x, dtype=float)
    alphas = x[1:]
    N      = len(alphas)

    # ------------------------------------------------------------------
    # Decode decision vector
    # ------------------------------------------------------------------
    if mode == 'time_optimal':
        tof_nd   = float(x[0])
        throttle = 1.0                     # full thrust, fixed
    else:                                  # mass_optimal
        throttle = float(x[0])
        tof_nd   = config_nd['tof_nd']    # fixed transfer time

    # Guard against degenerate or reversed time spans
    if tof_nd <= 0.0:
        return _penalty_result(N, tof_nd if tof_nd > 0 else 1.0)

    # ------------------------------------------------------------------
    # Propagate
    # ------------------------------------------------------------------
    y0 = get_initial_state_earth()

    try:
        traj = propagate_trajectory(
            y0         = y0,
            tof_nd     = tof_nd,
            alphas     = alphas,
            acc_thrust = config_nd['acc_thrust'],
            mp_dot     = config_nd['mp_dot'],
            throttle   = throttle,
            tol        = tol,
        )
    except RuntimeError:
        return _penalty_result(N, tof_nd)

    yfinal = traj['final_state']
    r_f, u_f, v_f = yfinal[0], yfinal[1], yfinal[2]
    acc_dv         = float(yfinal[5])

    # Catch NaN / Inf in the integrated state (numerical blowup)
    if not np.all(np.isfinite(yfinal)):
        return _penalty_result(N, tof_nd)

    # ------------------------------------------------------------------
    # Assemble output
    # ------------------------------------------------------------------
    objective   = acc_dv   if mode == 'time_optimal' else throttle
    constraints = [r_f - _R_MARS, u_f - _U_MARS, v_f - _V_MARS]

    return {
        'objective':    objective,
        'constraints':  constraints,
        'final_state':  yfinal,
        'state_history': traj['state_history'],
        'time_history':  traj['time_history'],
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _penalty_result(N: int, tof_nd: float) -> dict:
    """Return a penalty result dict when propagation fails or inputs are bad."""
    return {
        'objective':     _PENALTY,
        'constraints':   [_PENALTY, _PENALTY, _PENALTY],
        'final_state':   np.full(6, np.nan),
        'state_history': np.full((N + 1, 6), np.nan),
        'time_history':  np.linspace(0.0, abs(tof_nd), N + 1),
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import numpy as np
    from config.mission_config import MissionConfig

    print('=== shooting.py self-test ===\n')

    cfg = MissionConfig.default_time_optimal()
    nd  = cfg.to_nd_params()
    N   = 10   # small N for speed

    # ------------------------------------------------------------------
    # 1. Coasting trajectory (throttle=0) — constraints NOT satisfied,
    #    but function should not crash and shapes must be correct
    # ------------------------------------------------------------------
    print('Test 1: coasting run (no thrust), checking output structure')
    alpha_guess = cfg.alpha_guess_rad()
    x_coast = np.concatenate([[nd['tof_nd']], np.zeros(N)])  # all alpha=0

    # Override acc_thrust=0 to disable thrust independently of mode
    nd_coast = dict(nd, acc_thrust=0.0, mp_dot=0.0)
    res = shooting_function(x_coast, nd_coast, mode='time_optimal')

    assert set(res.keys()) == {'objective', 'constraints', 'final_state',
                                'state_history', 'time_history'}
    assert len(res['constraints'])   == 3
    assert res['state_history'].shape == (N + 1, 6)
    assert res['time_history'].shape  == (N + 1,)
    assert res['time_history'][0]  == 0.0
    assert abs(res['time_history'][-1] - nd['tof_nd']) < 1e-12

    print(f"  state_history shape : {res['state_history'].shape}  (expect ({N+1}, 6))")
    print(f"  constraints         : {[f'{c:.6f}' for c in res['constraints']]}")
    print(f"  (all non-zero — coasting does not reach Mars)")
    print('  PASSED\n')

    # ------------------------------------------------------------------
    # 2. Objective and constraint types are plain Python floats / lists
    # ------------------------------------------------------------------
    print('Test 2: output types')
    assert isinstance(res['objective'],   float)
    assert isinstance(res['constraints'], list)
    assert all(isinstance(c, float) for c in res['constraints'])
    assert isinstance(res['final_state'], np.ndarray)
    print('  PASSED\n')

    # ------------------------------------------------------------------
    # 3. Penalty path: tof_nd <= 0
    # ------------------------------------------------------------------
    print('Test 3: penalty result for tof_nd <= 0')
    x_bad = np.concatenate([[-1.0], np.zeros(N)])
    res_bad = shooting_function(x_bad, nd, mode='time_optimal')
    assert res_bad['objective'] == _PENALTY
    assert all(c == _PENALTY for c in res_bad['constraints'])
    print(f"  objective  = {res_bad['objective']}  (penalty)")
    print('  PASSED\n')

    # ------------------------------------------------------------------
    # 4. mass_optimal mode: x[0] is throttle, tof is fixed
    # ------------------------------------------------------------------
    print('Test 4: mass_optimal mode — throttle fixed at 0.5')
    nd_mo = dict(nd, tof_nd=nd['tof_nd'])
    x_mo  = np.concatenate([[0.5], np.full(N, 0.0)])
    res_mo = shooting_function(x_mo, nd_mo, mode='mass_optimal')
    assert isinstance(res_mo['objective'], float)
    assert abs(res_mo['objective'] - 0.5) < 1e-12, \
        "mass_optimal objective must equal throttle x[0]"
    print(f"  objective  = {res_mo['objective']:.4f}  (= throttle = 0.5)")
    print('  PASSED\n')

    # ------------------------------------------------------------------
    # 5. Prograde-thrust run: r should exceed 1.0 after tof
    # ------------------------------------------------------------------
    print('Test 5: prograde thrust (alpha=0) raises orbit radius')
    x_thrust = np.concatenate([[nd['tof_nd']], np.zeros(N)])
    res_t = shooting_function(x_thrust, nd, mode='time_optimal')
    r_final = res_t['final_state'][0]
    print(f"  r_final = {r_final:.5f}  (expect > 1.0 for prograde thrust)")
    assert r_final > 1.0, f"Orbit should expand: r_final = {r_final}"
    print('  PASSED')
