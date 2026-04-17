"""Equations of motion for 2D heliocentric low-thrust trajectory optimization.

Coordinate system
-----------------
Heliocentric polar (r, theta), non-dimensional.

    Quantity        Unit (dimensional)   ND unit
    --------------- -------------------- -------------------
    Distance        AU                   1 AU  (r* = 1 AU)
    Time            day                  t_cf = sqrt(1/mu_sun)  ~58.13 day
    Velocity        AU/day               AU / t_cf
    Acceleration    AU/day^2             AU / t_cf^2  = mu* / r*^2

With this normalization, mu* = 1, so circular velocity at r* is exactly 1.

State vector  y = [r, u, v, theta, mp, acc_dv]
-------------------------------------------------
    r       : heliocentric radial distance               [-]   (= AU value)
    u       : radial velocity component (r-direction)    [-]
    v       : transverse velocity component (th-direction) [-]
    theta   : polar angle                                [rad]
    mp      : accumulated propellant mass fraction        [-]   in [0, 1)
                mp = integral( throttle * mp_dot_ref  dt )
    acc_dv  : accumulated thrust delta-V                 [-]   (ND velocity)
                acc_dv = integral( a_m  dt )

Control variable
-----------------
    alpha   : thrust steering angle [rad]
              Measured from the local transverse (tangential) direction.
              Positive alpha rotates the thrust vector clockwise toward the
              outward radial direction (i.e. away from the Sun).

              alpha = 0        →  purely prograde (tangential) thrust
              alpha = +pi/2   →  purely outward radial thrust
              alpha = -pi/2   →  purely inward radial thrust

              Thrust components:
                  F_r  = a_m * sin(alpha)    [radial, +outward]
                  F_th = a_m * cos(alpha)    [transverse, +prograde]

Propulsion parameters (from core.nondimensional)
-------------------------------------------------
    acc_thrust  : reference ND thrust acceleration = (T/m0) / (mu_sun/r0^2)  [-]
    mp_dot_ref  : reference ND mass-flow rate =
                  (T/(g0*Isp) * 86400 / m0) * t_cf                           [-]
    throttle    : throttle setting in [0, 1]
"""

import numpy as np

from .constants import r_mars


def eom_2d(
    t: float,
    y: list,
    alpha: float,
    acc_thrust: float,
    mp_dot: float,
    throttle: float,
) -> list:
    """Evaluate the 2D heliocentric polar equations of motion.

    Parameters
    ----------
    t : float
        Current non-dimensional time [-].
    y : list or array-like, length 6
        State vector [r, u, v, theta, mp, acc_dv].

        r       : radial distance [-] (1 AU = 1 in ND)
        u       : radial velocity component [-]
        v       : transverse velocity component [-]
        theta   : polar angle [rad]
        mp      : accumulated propellant mass fraction [-], starts at 0
        acc_dv  : accumulated thrust delta-V [-]

    alpha : float
        Thrust steering angle [rad], measured from the transverse direction
        toward the outward radial direction (positive = clockwise above
        the local horizontal when viewed from the north ecliptic pole).
    acc_thrust : float
        Reference non-dimensional thrust acceleration (T/m0 normalised) [-].
        Computed via NonDimensional.compute_thrust_acc(T_N, m0_kg).
    mp_dot : float
        Reference non-dimensional propellant mass-flow rate [-].
        Computed via NonDimensional.compute_mass_flow(T_N, Isp_s, m0_kg).
        Positive scalar; the throttle factor is applied internally.
    throttle : float
        Throttle setting in [0, 1].  throttle = 1 → full thrust;
        throttle = 0 → coasting (ballistic) arc.

    Returns
    -------
    dydt : list, length 6
        Time derivatives [r_dot, u_dot, v_dot, theta_dot, mp_dot, acc_dv_dot].

    Notes
    -----
    Equations of motion (non-dimensional, mu* = 1):

        r_dot     = u
        u_dot     = v^2/r  -  1/r^2  +  a_m * sin(alpha)
        v_dot     = -u*v/r           +  a_m * cos(alpha)
        theta_dot = v / r
        mp_dot    = throttle * mp_dot_ref
        acc_dv_dot= a_m

    where the instantaneous thrust acceleration accounts for propellant
    depletion through the current mass fraction (1 - mp):

        a_m = throttle * acc_thrust / (1 - mp)

    Using the state variable mp (rather than mp_dot_ref * t) correctly
    handles variable-throttle arcs in which the denominator evolves as the
    throttle history dictates.  For constant full-throttle burns, the two
    are equivalent: mp = mp_dot_ref * t.
    """
    r, u, v, theta, mp, acc_dv = y

    # Current mass fraction; guard against numerical overshoot past 1
    mass_frac = max(1.0 - mp, 1e-6)

    # Current non-dimensional thrust acceleration [AU/t_cf^2]
    a_m = throttle * acc_thrust / mass_frac

    # Precompute trig
    sin_a = np.sin(alpha)
    cos_a = np.cos(alpha)

    # --- Equations of motion ---
    r_dot      = u
    u_dot      = (v * v) / r  -  1.0 / (r * r)  +  a_m * sin_a
    v_dot      = -(u * v) / r                    +  a_m * cos_a
    theta_dot  = v / r
    mp_dot_val = throttle * mp_dot
    dvdt       = a_m                # rate of delta-V accumulation

    return [r_dot, u_dot, v_dot, theta_dot, mp_dot_val, dvdt]


def eom_2d_ivp(t: float, y: list, params: dict) -> list:
    """Wrapper around eom_2d for use with scipy.integrate.solve_ivp.

    ``solve_ivp`` requires a callable with signature ``f(t, y)``.  This
    wrapper closes over a parameter dictionary and delegates to eom_2d,
    allowing the steering angle and propulsion constants to be updated
    between integration segments without changing the function signature.

    Parameters
    ----------
    t : float
        Current non-dimensional time [-].
    y : list or array-like, length 6
        State vector [r, u, v, theta, mp, acc_dv] — see eom_2d for details.
    params : dict
        Must contain:

        'alpha'      : float
            Thrust steering angle [rad].
        'acc_thrust' : float
            Reference non-dimensional thrust acceleration [-].
        'mp_dot'     : float
            Reference non-dimensional mass-flow rate [-].
        'throttle'   : float
            Throttle setting in [0, 1].

    Returns
    -------
    dydt : list, length 6
        Time derivatives — see eom_2d.

    Example
    -------
    >>> from scipy.integrate import solve_ivp
    >>> params = {'alpha': 0.0, 'acc_thrust': 3e-2, 'mp_dot': 3e-2, 'throttle': 1.0}
    >>> sol = solve_ivp(
    ...     lambda t, y: eom_2d_ivp(t, y, params),
    ...     t_span=[0.0, 6.283],
    ...     y0=[1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    ... )
    """
    return eom_2d(
        t, y,
        alpha      = params['alpha'],
        acc_thrust = params['acc_thrust'],
        mp_dot     = params['mp_dot'],
        throttle   = params['throttle'],
    )


def terminal_conditions_mars() -> dict:
    """Return the non-dimensional target state at Mars arrival (circular orbit).

    The target orbit is a Keplerian circular orbit at Mars's mean
    semi-major axis.  In the non-dimensional system (mu* = 1):

        r_f = r_mars                  [-]   (= 1.52368, same numerical value as AU)
        u_f = 0                       [-]   (no radial velocity in circular orbit)
        v_f = sqrt(mu* / r_f)
            = sqrt(1 / r_mars)        [-]   (circular transverse velocity)

    theta_f is unconstrained (free final angle).

    Returns
    -------
    dict with keys:
        'r_f'  : float — target radial distance [-]
        'u_f'  : float — target radial velocity [-]
        'v_f'  : float — target transverse velocity [-]

    Notes
    -----
    The circular velocity at Mars in dimensional units:
        v_circ = sqrt(mu_sun / r_mars)
               = sqrt(2.959e-4 AU^3/day^2 / 1.52368 AU)
               ~ 0.01394 AU/day  ~ 24.13 km/s

    In non-dimensional units (divide by AU/t_cf = sqrt(mu_sun)):
        v_f = sqrt(1 / r_mars) = sqrt(1 / 1.52368) ~ 0.8104
    """
    r_f = r_mars                      # 1.52368 AU (ND)
    u_f = 0.0
    v_f = np.sqrt(1.0 / r_f)         # circular speed at Mars in ND units
    return {'r_f': r_f, 'u_f': u_f, 'v_f': v_f}


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from scipy.integrate import solve_ivp
    import numpy as np

    print("=== equations_of_motion.py self-test ===\n")

    # ------------------------------------------------------------------
    # 1. Keplerian check: coasting circular orbit at 1 AU stays circular
    # ------------------------------------------------------------------
    print("Test 1: coasting circular orbit at Earth (alpha irrelevant, throttle=0)")
    y0 = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]   # r=1, u=0, v=1 (ND Earth circ.)
    params_coast = {'alpha': 0.0, 'acc_thrust': 0.0, 'mp_dot': 0.0, 'throttle': 0.0}

    one_orbit_nd = 2 * np.pi                # one full period in ND time (= 2pi for r=1)
    sol = solve_ivp(
        lambda t, y: eom_2d_ivp(t, y, params_coast),
        t_span=[0.0, one_orbit_nd],
        y0=y0,
        method='DOP853',
        rtol=1e-10,
        atol=1e-12,
    )
    r_final = sol.y[0, -1]
    u_final = sol.y[1, -1]
    v_final = sol.y[2, -1]
    print(f"  r after 1 orbit : {r_final:.8f}  (expect 1.0)")
    print(f"  u after 1 orbit : {u_final:.2e}   (expect ~0)")
    print(f"  v after 1 orbit : {v_final:.8f}  (expect 1.0)")
    assert abs(r_final - 1.0) < 1e-8, "Circular orbit radius drift!"
    assert abs(u_final)       < 1e-8, "Radial velocity non-zero!"
    assert abs(v_final - 1.0) < 1e-8, "Tangential velocity drift!"
    print("  PASSED\n")

    # ------------------------------------------------------------------
    # 2. Terminal conditions
    # ------------------------------------------------------------------
    print("Test 2: terminal conditions at Mars")
    tc = terminal_conditions_mars()
    print(f"  r_f = {tc['r_f']:.5f}  (expect 1.52368)")
    print(f"  u_f = {tc['u_f']:.1f}     (expect 0.0)")
    print(f"  v_f = {tc['v_f']:.6f}  (expect sqrt(1/1.52368) = {np.sqrt(1/1.52368):.6f})")
    assert tc['r_f'] == r_mars
    assert tc['u_f'] == 0.0
    assert abs(tc['v_f'] - np.sqrt(1.0 / r_mars)) < 1e-12
    print("  PASSED\n")

    # ------------------------------------------------------------------
    # 3. Thrust derivative sanity check (alpha=0, full prograde)
    # ------------------------------------------------------------------
    print("Test 3: thrust derivative at alpha=0 (prograde), full throttle")
    T_N, Isp_s, m0_kg = 0.09, 3000.0, 500.0
    from core.nondimensional import NonDimensional
    at = NonDimensional.compute_thrust_acc(T_N, m0_kg)
    md = NonDimensional.compute_mass_flow(T_N, Isp_s, m0_kg)
    dydt = eom_2d(0.0, y0, alpha=0.0, acc_thrust=at, mp_dot=md, throttle=1.0)
    print(f"  acc_thrust      = {at:.6e}")
    print(f"  mp_dot_ref      = {md:.6e}")
    print(f"  r_dot           = {dydt[0]:.6f}  (= u = 0.0)")
    print(f"  u_dot (grav)    = {-1.0:.6f}, +thrust_r = {at*np.sin(0):.6f}")
    print(f"  v_dot (thrust)  = {dydt[2]:.6e}  (= a_m = {at:.6e})")
    print(f"  mp_dot          = {dydt[4]:.6e}  (= throttle * mp_dot_ref)")
    assert abs(dydt[4] - md) < 1e-15, "mp_dot mismatch!"
    assert abs(dydt[2] - at) < 1e-12, "v_dot / a_m mismatch!"
    print("  PASSED")
