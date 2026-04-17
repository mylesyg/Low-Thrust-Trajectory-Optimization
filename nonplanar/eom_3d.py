"""3D Cartesian equations of motion for nonplanar low-thrust trajectory optimization.

Extends the 2D polar EOM from core/equations_of_motion.py to full 3D
heliocentric inertial Cartesian coordinates.  The same AU/t_cf
non-dimensionalisation is used throughout, so mu* = 1 and the distance unit
is 1 AU.

State vector
------------
y = [x, y, z, vx, vy, vz, mp, acc_dv]   length 8

    x, y, z    : heliocentric Cartesian position [AU, ND]
    vx, vy, vz : heliocentric Cartesian velocity [AU/t_cf, ND]
    mp         : propellant mass fraction consumed [-], mp in [0, 1)
    acc_dv     : accumulated delta-V magnitude [AU/t_cf, ND]

Control variables (per segment, piecewise constant)
---------------------------------------------------
    alpha : in-plane azimuth [rad] — same role as the 2D steering angle;
            measured from the heliocentric +x axis in the ecliptic plane.
    beta  : out-of-plane declination [rad], beta in [-pi/2, pi/2].
            beta > 0 thrusts toward the ecliptic north pole.

Thrust unit vector (inertial frame)
------------------------------------
    u_hat = [cos(beta)*cos(alpha), cos(beta)*sin(alpha), sin(beta)]

Non-dimensionalisation (identical to core.nondimensional)
----------------------------------------------------------
    Length unit : 1 AU   (so r* = 1 AU = ND unit)
    Time unit   : t_cf = sqrt(AU^3 / mu_sun)  ~58.13 days
    Velocity    : AU / t_cf
    mu*         = 1  (gravitational parameter is unity)

References
----------
Morante, D., et al. (2021). A Survey on Low-Thrust Trajectory Optimization
    Approaches. Aerospace, 8(3), 88. Equation (22).
"""

import numpy as np
from scipy.integrate import solve_ivp

# ND gravitational parameter: mu* = 1 in the AU/t_cf unit system
_MU_ND: float = 1.0


# ---------------------------------------------------------------------------
# Equations of motion
# ---------------------------------------------------------------------------

def eom_3d(
    t: float,
    y: np.ndarray,
    alpha: float,
    beta: float,
    acc_thrust: float,
    mp_dot: float,
    throttle: float,
) -> list:
    """3D Cartesian EOM for a heliocentric low-thrust spacecraft.

    Parameters
    ----------
    t : float
        Current non-dimensional time (not used explicitly; required by
        ``scipy.integrate.solve_ivp``).
    y : array-like, length 8
        State vector [x, y_c, z, vx, vy, vz, mp, acc_dv] (non-dimensional).
        (Internal variable name ``y_c`` avoids shadowing the parameter ``y``.)
    alpha : float
        In-plane thrust azimuth angle [rad].  Measured from the heliocentric
        +x axis (vernal equinox direction) in the ecliptic plane.
    beta : float
        Out-of-plane thrust declination angle [rad], in [-pi/2, pi/2].
        Positive beta thrusts toward the ecliptic north (ecliptic +z).
    acc_thrust : float
        Non-dimensional full-throttle thrust acceleration at unit mass
        fraction: a_T = T / (m0 * mu_sun_per_AU2) [ND].
        Computed by ``NonDimensional.compute_thrust_acc()``.
    mp_dot : float
        Non-dimensional mass-flow reference:
        mp_dot = (T / (g0 * Isp)) * (86400 s/day) / m0 * t_cf [ND/t_cf].
        Computed by ``NonDimensional.compute_mass_flow()``.
    throttle : float
        Throttle setting in [0, 1].  For time-optimal mode throttle = 1.

    Returns
    -------
    dydt : list of float, length 8
        Time derivatives [dx, dy, dz, dvx, dvy, dvz, dmp, dacc_dv].

    Notes
    -----
    Mass depletion:
        dmp/dt = throttle * mp_dot
    Effective thrust acceleration (accounts for depleted propellant):
        a_m = throttle * acc_thrust / max(1 - mp, 1e-6)
    Accumulated delta-V:
        dacc_dv/dt = a_m   (integral of thrust magnitude, ND velocity units)
    """
    xp, yc, zp, vx, vy, vz, mp, acc_dv = y

    r_sq = xp*xp + yc*yc + zp*zp
    r3   = r_sq ** 1.5
    if r3 < 1.0e-30:
        r3 = 1.0e-30   # guard against singularity at origin

    # Remaining mass fraction (propellant consumed -> decreasing total mass)
    mass_frac = max(1.0 - mp, 1.0e-6)

    # Effective thrust acceleration magnitude
    a_m = throttle * acc_thrust / mass_frac

    # Thrust unit vector in inertial Cartesian frame
    cb, sb = np.cos(beta),  np.sin(beta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    ux = cb * ca
    uy = cb * sa
    uz = sb

    # Equations of motion (point-mass gravity + thrust)
    dx     = vx
    dy     = vy
    dz     = vz
    dvx    = -_MU_ND * xp / r3 + a_m * ux
    dvy    = -_MU_ND * yc / r3 + a_m * uy
    dvz    = -_MU_ND * zp / r3 + a_m * uz
    dmp    = throttle * mp_dot
    dacc_dv = a_m

    return [dx, dy, dz, dvx, dvy, dvz, dmp, dacc_dv]


def eom_3d_ivp(t: float, y: np.ndarray, params: dict) -> list:
    """Wrapper for ``eom_3d`` compatible with ``scipy.integrate.solve_ivp``.

    Parameters
    ----------
    t : float
        Current non-dimensional time.
    y : np.ndarray, length 8
        State vector [x, y_c, z, vx, vy, vz, mp, acc_dv].
    params : dict
        Must contain keys: ``'alpha'``, ``'beta'``, ``'acc_thrust'``,
        ``'mp_dot'``, ``'throttle'``.

    Returns
    -------
    dydt : list of float, length 8
    """
    return eom_3d(
        t, y,
        params['alpha'],
        params['beta'],
        params['acc_thrust'],
        params['mp_dot'],
        params['throttle'],
    )


# ---------------------------------------------------------------------------
# Cartesian to orbital elements conversion
# ---------------------------------------------------------------------------

def cartesian_to_orbital_elements(
    r_vec: np.ndarray,
    v_vec: np.ndarray,
    mu: float = _MU_ND,
) -> dict:
    """Convert a Cartesian state to classical orbital elements.

    Parameters
    ----------
    r_vec : array-like, shape (3,)
        Position vector.  Units must be consistent with ``mu`` and ``v_vec``.
        For ND units: [AU].
    v_vec : array-like, shape (3,)
        Velocity vector.  For ND units: [AU/t_cf].
    mu : float, optional
        Gravitational parameter [length^3 / time^2].  Default: 1.0 (ND).

    Returns
    -------
    elements : dict
        'a'         : float — semi-major axis (same length units as r_vec);
                      np.inf for a parabolic orbit.
        'e'         : float — eccentricity [-].
        'i_deg'     : float — inclination [deg], in [0, 180].
        'raan_deg'  : float — RAAN [deg], in [0, 360); np.nan if equatorial.
        'omega_deg' : float — argument of periapsis [deg], in [0, 360);
                      np.nan if circular (e < 1e-8) or equatorial.
        'nu_deg'    : float — true anomaly [deg], in [0, 360).
        'h'         : float — specific angular momentum magnitude.
        'energy'    : float — specific orbital energy (negative = elliptic).

    Notes
    -----
    Algorithm follows Curtis (2020) "Orbital Mechanics for Engineering
    Students", 4th ed., Algorithm 4.2.

    Degenerate cases:

    * Equatorial orbit (|sin(i)| < 1e-5): RAAN is undefined -> np.nan.
    * Circular orbit (e < 1e-8): argument of periapsis is undefined -> np.nan.
      True anomaly is replaced by argument of latitude.
    * Parabolic orbit (|energy| < 1e-30): semi-major axis -> np.inf.
    """
    r_vec = np.asarray(r_vec, dtype=float)
    v_vec = np.asarray(v_vec, dtype=float)

    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    # Specific angular momentum
    h_vec = np.cross(r_vec, v_vec)
    h     = np.linalg.norm(h_vec)

    # Specific orbital energy and semi-major axis
    energy = 0.5 * v * v - mu / r
    a = np.inf if abs(energy) < 1.0e-30 else -mu / (2.0 * energy)

    # Eccentricity vector and magnitude
    e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r
    e     = np.linalg.norm(e_vec)

    # Inclination: angle between h and z-axis
    i_rad = np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0))
    i_deg = np.degrees(i_rad)

    # Node vector: N = z_hat x h  (points toward ascending node)
    n_vec = np.array([-h_vec[1], h_vec[0], 0.0])   # = cross([0,0,1], h)
    n     = np.linalg.norm(n_vec)

    # ------------------------------------------------------------------
    # RAAN: angle from +x to ascending node, measured in equatorial plane
    # ------------------------------------------------------------------
    if n < 1.0e-10:   # equatorial orbit (i ~ 0 or ~180 deg)
        raan_deg = np.nan
    else:
        cos_raan = np.clip(n_vec[0] / n, -1.0, 1.0)
        raan_deg = np.degrees(np.arccos(cos_raan))
        if n_vec[1] < 0.0:
            raan_deg = 360.0 - raan_deg   # quadrant check: N_y < 0 -> 2nd half

    # ------------------------------------------------------------------
    # Argument of periapsis: angle from ascending node to periapsis
    # ------------------------------------------------------------------
    if e < 1.0e-8:
        omega_deg = np.nan   # circular: periapsis undefined
    elif n < 1.0e-10:
        # Equatorial non-circular: angle measured from +x
        cos_omega = np.clip(e_vec[0] / e, -1.0, 1.0)
        omega_deg = np.degrees(np.arccos(cos_omega))
        if e_vec[1] < 0.0:
            omega_deg = 360.0 - omega_deg
    else:
        cos_omega = np.clip(np.dot(n_vec, e_vec) / (n * e), -1.0, 1.0)
        omega_deg = np.degrees(np.arccos(cos_omega))
        if e_vec[2] < 0.0:
            omega_deg = 360.0 - omega_deg   # e_z < 0: periapsis below equator

    # ------------------------------------------------------------------
    # True anomaly: angle from periapsis to spacecraft
    # ------------------------------------------------------------------
    if e < 1.0e-8:
        # Circular: use argument of latitude (angle from ascending node)
        if n < 1.0e-10:
            # Equatorial circular: use true longitude (angle from +x)
            cos_nu = np.clip(r_vec[0] / r, -1.0, 1.0)
            nu_deg = np.degrees(np.arccos(cos_nu))
            if v_vec[0] > 0.0:
                nu_deg = 360.0 - nu_deg
        else:
            cos_nu = np.clip(np.dot(n_vec, r_vec) / (n * r), -1.0, 1.0)
            nu_deg = np.degrees(np.arccos(cos_nu))
            if np.dot(n_vec, v_vec) < 0.0:
                nu_deg = 360.0 - nu_deg   # moving away from ascending node
    else:
        cos_nu = np.clip(np.dot(e_vec, r_vec) / (e * r), -1.0, 1.0)
        nu_deg = np.degrees(np.arccos(cos_nu))
        if np.dot(r_vec, v_vec) < 0.0:
            nu_deg = 360.0 - nu_deg   # negative radial velocity -> past apoapsis

    return {
        'a':         a,
        'e':         e,
        'i_deg':     i_deg,
        'raan_deg':  raan_deg,
        'omega_deg': omega_deg,
        'nu_deg':    nu_deg,
        'h':         h,
        'energy':    energy,
    }


# ---------------------------------------------------------------------------
# Propagation utilities
# ---------------------------------------------------------------------------

def propagate_3d_segment(
    y0: np.ndarray,
    t_start: float,
    t_end: float,
    alpha: float,
    beta: float,
    acc_thrust: float,
    mp_dot: float,
    throttle: float,
    tol: float = 1.0e-10,
) -> np.ndarray:
    """Integrate one constant-steering segment of the 3D EOM.

    Parameters
    ----------
    y0 : np.ndarray, length 8
        Initial state [x, y_c, z, vx, vy, vz, mp, acc_dv].
    t_start, t_end : float
        Non-dimensional start and end times of the segment.
    alpha : float
        In-plane azimuth [rad].
    beta : float
        Out-of-plane declination [rad].
    acc_thrust, mp_dot, throttle : float
        Thrust parameters (same meaning as ``eom_3d``).
    tol : float, optional
        Relative and absolute ODE tolerance (default 1e-10).

    Returns
    -------
    y_end : np.ndarray, length 8
        State at the end of the segment.

    Raises
    ------
    RuntimeError
        If ``scipy.integrate.solve_ivp`` fails to integrate the segment.
    """
    params = {
        'alpha':      alpha,
        'beta':       beta,
        'acc_thrust': acc_thrust,
        'mp_dot':     mp_dot,
        'throttle':   throttle,
    }
    sol = solve_ivp(
        fun          = eom_3d_ivp,
        t_span       = (t_start, t_end),
        y0           = y0,
        method       = 'DOP853',
        args         = (params,),
        rtol         = tol,
        atol         = tol,
        dense_output = False,
    )
    if not sol.success:
        raise RuntimeError(
            f"3D segment integration failed in "
            f"[{t_start:.4f}, {t_end:.4f}]: {sol.message}"
        )
    return sol.y[:, -1]


def propagate_3d_trajectory(
    y0: np.ndarray,
    tof_nd: float,
    alphas: np.ndarray,
    betas: np.ndarray,
    acc_thrust: float,
    mp_dot: float,
    throttle: float,
    tol: float = 1.0e-10,
) -> dict:
    """Propagate a piecewise-constant-steering 3D trajectory.

    Divides the total transfer time into N equal segments (N = len(alphas)),
    integrating each with constant alpha and beta.

    Parameters
    ----------
    y0 : np.ndarray, length 8
        Initial state [x, y_c, z, vx, vy, vz, mp, acc_dv].
    tof_nd : float
        Total non-dimensional transfer time.
    alphas : array-like, length N
        In-plane azimuth angles, one per segment [rad].
    betas : array-like, length N
        Out-of-plane declination angles, one per segment [rad].
    acc_thrust, mp_dot, throttle : float
        Thrust parameters (same meaning as ``eom_3d``).
    tol : float, optional
        ODE solver tolerance (default 1e-10).

    Returns
    -------
    result : dict
        'final_state'  : np.ndarray (8,) — state at arrival
        'state_history': np.ndarray (N+1, 8) — state at each node
        'time_history' : np.ndarray (N+1,) — ND node times

    Raises
    ------
    RuntimeError
        Propagated from ``propagate_3d_segment`` on ODE failure.
    """
    alphas = np.asarray(alphas, dtype=float)
    betas  = np.asarray(betas,  dtype=float)
    N = len(alphas)

    # Use np.linspace to avoid accumulated float-point drift in time grid
    times   = np.linspace(0.0, tof_nd, N + 1)
    history = np.empty((N + 1, 8))
    history[0] = y0
    y_cur = np.array(y0, dtype=float)

    for k in range(N):
        y_cur = propagate_3d_segment(
            y_cur, times[k], times[k + 1],
            alphas[k], betas[k],
            acc_thrust, mp_dot, throttle,
            tol=tol,
        )
        history[k + 1] = y_cur

    return {
        'final_state':   y_cur,
        'state_history': history,
        'time_history':  times,
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print('=== eom_3d.py self-test ===\n')

    # ------------------------------------------------------------------
    # 1. eom_3d at zero thrust: kinematics match gravity only
    # ------------------------------------------------------------------
    print('Test 1: zero-thrust EOM — gravity only, no thrust acceleration')
    # State: Earth circular orbit at x-axis: [1, 0, 0, 0, 1, 0, 0, 0]
    y0 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    dydt = eom_3d(0.0, y0, alpha=0.0, beta=0.0,
                  acc_thrust=0.118, mp_dot=0.12, throttle=0.0)
    # With throttle=0: no thrust, pure gravity
    # dx = vx = 0, dy = vy = 1, dz = vz = 0
    # dvx = -mu*x/r^3 = -1, dvy = -mu*y/r^3 = 0, dvz = 0
    # dmp = 0, dacc_dv = 0
    print(f"  dydt = {dydt}")
    assert abs(dydt[0] - 0.0) < 1e-12, f"dx != vy: {dydt[0]}"
    assert abs(dydt[1] - 1.0) < 1e-12, f"dy != vy: {dydt[1]}"
    assert abs(dydt[2] - 0.0) < 1e-12, f"dz != 0: {dydt[2]}"
    assert abs(dydt[3] - (-1.0)) < 1e-12, f"dvx != -mu/r^2: {dydt[3]}"
    assert abs(dydt[4] - 0.0)   < 1e-12, f"dvy != 0: {dydt[4]}"
    assert abs(dydt[5] - 0.0)   < 1e-12, f"dvz != 0: {dydt[5]}"
    assert abs(dydt[6] - 0.0)   < 1e-12, f"dmp != 0: {dydt[6]}"
    assert abs(dydt[7] - 0.0)   < 1e-12, f"dacc_dv != 0: {dydt[7]}"
    print('  PASSED\n')

    # ------------------------------------------------------------------
    # 2. eom_3d thrust direction: verify u_hat components
    # ------------------------------------------------------------------
    print('Test 2: thrust acceleration direction for alpha=0, beta=pi/4')
    alpha, beta = 0.0, np.pi / 4.0
    acc_thrust, throttle = 0.118, 1.0
    mp0 = 0.0   # no propellant consumed yet (mass_frac = 1.0)
    y_test = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, mp0, 0.0]
    dydt_t = eom_3d(0.0, y_test, alpha, beta, acc_thrust, 0.12, throttle)
    # u_hat = [cos(45)*cos(0), cos(45)*sin(0), sin(45)] = [1/sqrt(2), 0, 1/sqrt(2)]
    cb, sb = np.cos(beta), np.sin(beta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    expected_ax = -_MU_ND / 1.0 + acc_thrust * cb * ca
    expected_ay =               + acc_thrust * cb * sa
    expected_az =               + acc_thrust * sb
    print(f"  dvx = {dydt_t[3]:.6f}  (expect {expected_ax:.6f})")
    print(f"  dvy = {dydt_t[4]:.6f}  (expect {expected_ay:.6f})")
    print(f"  dvz = {dydt_t[5]:.6f}  (expect {expected_az:.6f})")
    assert abs(dydt_t[3] - expected_ax) < 1e-12
    assert abs(dydt_t[4] - expected_ay) < 1e-12
    assert abs(dydt_t[5] - expected_az) < 1e-12
    print('  PASSED\n')

    # ------------------------------------------------------------------
    # 3. cartesian_to_orbital_elements: Earth circular orbit
    # ------------------------------------------------------------------
    print('Test 3: OE conversion — Earth circular orbit at [1,0,0], v=[0,1,0]')
    r_e = np.array([1.0, 0.0, 0.0])
    v_e = np.array([0.0, 1.0, 0.0])   # circular speed at r=1 in ND units (mu*=1)
    oe = cartesian_to_orbital_elements(r_e, v_e)
    print(f"  a         = {oe['a']:.6f}  (expect 1.0)")
    print(f"  e         = {oe['e']:.2e}   (expect ~0)")
    print(f"  i_deg     = {oe['i_deg']:.2f}  (expect 0.0)")
    print(f"  raan_deg  = {oe['raan_deg']}  (expect NaN, equatorial)")
    print(f"  energy    = {oe['energy']:.6f}  (expect -0.5)")
    assert abs(oe['a'] - 1.0)      < 1e-10, f"a mismatch: {oe['a']}"
    assert oe['e']                  < 1e-8,  f"e not near zero: {oe['e']}"
    assert abs(oe['i_deg'] - 0.0)  < 1e-8,  f"i_deg mismatch: {oe['i_deg']}"
    assert np.isnan(oe['raan_deg']),          "RAAN should be NaN for equatorial"
    assert abs(oe['energy'] - (-0.5)) < 1e-10
    print('  PASSED\n')

    # ------------------------------------------------------------------
    # 4. cartesian_to_orbital_elements: inclined circular orbit
    # ------------------------------------------------------------------
    print('Test 4: OE conversion — inclined circular orbit (i=30, RAAN=45)')
    i_ref   = np.radians(30.0)
    raan_ref = np.radians(45.0)
    r1 = 1.0
    v1 = np.sqrt(_MU_ND / r1)   # circular speed
    # State at ascending node (argument of latitude = 0)
    r_vec = r1 * np.array([np.cos(raan_ref), np.sin(raan_ref), 0.0])
    v_vec = v1 * np.array([
        -np.cos(i_ref) * np.sin(raan_ref),
         np.cos(i_ref) * np.cos(raan_ref),
         np.sin(i_ref),
    ])
    oe2 = cartesian_to_orbital_elements(r_vec, v_vec)
    print(f"  a         = {oe2['a']:.6f}  (expect 1.0)")
    print(f"  e         = {oe2['e']:.2e}   (expect ~0)")
    print(f"  i_deg     = {oe2['i_deg']:.4f}  (expect 30.0)")
    print(f"  raan_deg  = {oe2['raan_deg']:.4f}  (expect 45.0)")
    assert abs(oe2['a']        - 1.0)   < 1e-9,  f"a mismatch: {oe2['a']}"
    assert oe2['e']                       < 1e-8,  f"e not zero: {oe2['e']}"
    assert abs(oe2['i_deg']    - 30.0)  < 1e-6,  f"i mismatch: {oe2['i_deg']}"
    assert abs(oe2['raan_deg'] - 45.0)  < 1e-6,  f"RAAN mismatch: {oe2['raan_deg']}"
    print('  PASSED\n')

    # ------------------------------------------------------------------
    # 5. Energy conservation over a free-flight arc (throttle = 0)
    # ------------------------------------------------------------------
    print('Test 5: energy conservation for zero-thrust propagation')
    from core.nondimensional import NonDimensional
    from core.constants import r_mars

    tof_nd  = NonDimensional.time_to_nd(50.0)   # 50 days ND
    y0_free = np.array([r_mars, 0.0, 0.0, 0.0, np.sqrt(_MU_ND / r_mars), 0.0, 0.0, 0.0])
    traj    = propagate_3d_trajectory(
        y0_free, tof_nd,
        alphas=np.zeros(5), betas=np.zeros(5),
        acc_thrust=0.118, mp_dot=0.12, throttle=0.0,
    )
    yf = traj['final_state']
    r_f = np.linalg.norm(yf[:3])
    v_f = np.linalg.norm(yf[3:6])
    E0 = 0.5 * np.linalg.norm(y0_free[3:6])**2 - _MU_ND / np.linalg.norm(y0_free[:3])
    Ef = 0.5 * v_f**2 - _MU_ND / r_f
    print(f"  Initial energy : {E0:.10f}")
    print(f"  Final energy   : {Ef:.10f}")
    print(f"  Delta energy   : {abs(Ef - E0):.2e}  (expect < 1e-9)")
    assert abs(Ef - E0) < 1e-9, f"Energy not conserved: dE = {abs(Ef-E0):.2e}"
    print(f"  state_history shape : {traj['state_history'].shape}  (expect (6, 8))")
    assert traj['state_history'].shape == (6, 8)
    print('  PASSED\n')

    # ------------------------------------------------------------------
    # 6. RAAN constraint formula verification
    # ------------------------------------------------------------------
    print('Test 6: RAAN constraint formula  h_x*cos(R) + h_y*sin(R) = 0')
    for i_deg, raan_deg in [(30.0, 0.0), (45.0, 90.0), (60.0, 135.0), (90.0, 270.0)]:
        i_r = np.radians(i_deg)
        R_r = np.radians(raan_deg)
        r_v = np.array([np.cos(R_r), np.sin(R_r), 0.0])
        v_v = np.array([-np.cos(i_r)*np.sin(R_r), np.cos(i_r)*np.cos(R_r), np.sin(i_r)])
        h_v = np.cross(r_v, v_v)
        c5  = h_v[0]*np.cos(R_r) + h_v[1]*np.sin(R_r)
        print(f"  i={i_deg:6.1f}, RAAN={raan_deg:6.1f}: c5 = {c5:.2e}  (expect 0)")
        assert abs(c5) < 1e-14, f"RAAN constraint nonzero at ({i_deg},{raan_deg}): {c5}"
    print('  PASSED')
