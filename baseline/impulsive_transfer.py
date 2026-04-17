"""Hohmann transfer baseline for Earth-Mars comparison.

The Hohmann transfer is the analytically optimal two-impulse manoeuvre
between coplanar circular orbits.  It provides the reference delta-V budget
against which low-thrust trajectories are compared:

    delta_v_hohmann ~ 5.6 km/s  (Earth -> Mars)
    tof_hohmann     ~ 259 days

Because the low-thrust optimizer works in non-dimensional units, and the
Hohmann result is pure Keplerian mechanics, this module operates entirely in
dimensional units (AU, days, km/s) and converts where needed.

Analytical formulae
-------------------
Transfer orbit semi-major axis:
    a_t = (r1 + r2) / 2

Transfer time (half-period):
    tof = pi * sqrt(a_t^3 / mu)

Departure burn (from circular orbit at r1 to ellipse periapsis):
    delta_v1 = sqrt(mu/r1) * [sqrt(2*r2 / (r1+r2)) - 1]

Arrival burn (from ellipse apoapsis to circular orbit at r2):
    delta_v2 = sqrt(mu/r2) * [1 - sqrt(2*r1 / (r1+r2))]

Both burns are prograde (positive tangential direction).

References
----------
Bate, R.R., Mueller, D.D., White, J.E. (1971).
    Fundamentals of Astrodynamics. Dover.
    Chapter 6, Section 6.3.
Battin, R.H. (1999).
    An Introduction to the Mathematics and Methods of Astrodynamics.
    AIAA Education Series.
"""

import numpy as np

from core.constants import r_earth, r_mars, mu_sun_au3day2, AU_m
from config.mission_config import MissionConfig

# Conversion: 1 AU/day -> m/s
_AU_DAY_TO_MS: float = AU_m / 86_400.0      # ~1 731 457 m/s


# ---------------------------------------------------------------------------
# Hohmann transfer
# ---------------------------------------------------------------------------

def hohmann_transfer(r1_au: float, r2_au: float) -> dict:
    """Compute the optimal two-impulse Hohmann transfer between circular orbits.

    Both orbits are assumed to be coplanar, circular, and prograde.  The
    Hohmann transfer is the globally optimal two-impulse manoeuvre for this
    geometry when r2/r1 <= 15.58 (always satisfied for inner solar system).

    Parameters
    ----------
    r1_au : float
        Departure orbit radius [AU].  For Earth: r1_au = 1.0.
    r2_au : float
        Arrival orbit radius [AU].   For Mars:  r2_au = 1.52368.

    Returns
    -------
    result : dict
        'a_au'               : float -- transfer orbit semi-major axis [AU]
        'delta_v1_auday'     : float -- departure burn [AU/day]
        'delta_v2_auday'     : float -- arrival burn [AU/day]
        'total_delta_v_auday': float -- total delta-V [AU/day]
        'tof_days'           : float -- transfer time [days]

    Notes
    -----
    The departure and arrival burns are both tangential (prograde).
    delta_v1 adds energy (circular orbit at r1 -> ellipse periapsis).
    delta_v2 adds energy (ellipse apoapsis -> circular orbit at r2).
    Both are positive for an outward transfer (r2 > r1).

    For an inward transfer (r2 < r1) the formulae still give positive
    magnitudes; both burns are physically retrograde.

    Raises
    ------
    ValueError
        If r1_au or r2_au are non-positive, or if r1_au == r2_au.
    """
    if r1_au <= 0.0 or r2_au <= 0.0:
        raise ValueError(
            f"Orbital radii must be positive; got r1={r1_au}, r2={r2_au}"
        )
    if abs(r1_au - r2_au) < 1.0e-10:
        raise ValueError(
            f"r1 = r2 ({r1_au:.6f} AU); Hohmann transfer requires r1 != r2."
        )

    mu  = mu_sun_au3day2
    a_t = (r1_au + r2_au) / 2.0                      # transfer orbit SMA [AU]

    # Circular-orbit speeds
    v_circ_1 = np.sqrt(mu / r1_au)                   # [AU/day]
    v_circ_2 = np.sqrt(mu / r2_au)                   # [AU/day]

    # Transfer orbit periapsis and apoapsis speeds (vis-viva)
    v_peri = np.sqrt(mu * (2.0 / r1_au - 1.0 / a_t))   # at r1 [AU/day]
    v_apo  = np.sqrt(mu * (2.0 / r2_au - 1.0 / a_t))   # at r2 [AU/day]

    delta_v1 = abs(v_peri - v_circ_1)   # departure burn magnitude [AU/day]
    delta_v2 = abs(v_circ_2 - v_apo)    # arrival burn magnitude   [AU/day]

    # Transfer time = half the period of the transfer ellipse
    tof_days = np.pi * np.sqrt(a_t ** 3 / mu)        # [days]

    return {
        'a_au':                a_t,
        'delta_v1_auday':      delta_v1,
        'delta_v2_auday':      delta_v2,
        'total_delta_v_auday': delta_v1 + delta_v2,
        'tof_days':            tof_days,
    }


# ---------------------------------------------------------------------------
# Mission-level baseline
# ---------------------------------------------------------------------------

def compute_baseline(config: MissionConfig) -> dict:
    """Compute the Hohmann transfer baseline for comparison with low-thrust results.

    Computes the optimal impulsive (two-burn) Earth-Mars Hohmann transfer
    and returns a summary dictionary with values in SI units (m/s, days, kg)
    suitable for direct comparison with the low-thrust optimizer output.

    Parameters
    ----------
    config : MissionConfig
        Mission configuration.  Used for ``initial_mass_kg`` and ``isp_s``
        (Tsiolkovsky propellant estimate).  Orbital radii are taken from
        ``core.constants`` (1.0 AU for Earth, 1.52368 AU for Mars).

    Returns
    -------
    baseline : dict
        'tof_days'            : float -- Hohmann transfer time [days]
        'delta_v1_ms'         : float -- departure burn [m/s]
        'delta_v2_ms'         : float -- arrival burn [m/s]
        'total_delta_v_ms'    : float -- total delta-V [m/s]
        'a_transfer_au'       : float -- transfer ellipse SMA [AU]
        'propellant_fraction' : float -- propellant mass / initial mass [-]
        'propellant_mass_kg'  : float -- propellant consumed [kg]
        'final_mass_kg'       : float -- remaining spacecraft mass [kg]
        'mass_ratio'          : float -- m0 / mf (Tsiolkovsky) [-]

    Notes
    -----
    Uses the sequential Tsiolkovsky rocket equation:

        m1  = m0  * exp(-dv1 / (g0 * Isp))    after departure burn
        mf  = m1  * exp(-dv2 / (g0 * Isp))    after arrival burn

    The sequential form correctly accounts for the mass already consumed
    in the first burn before computing the second burn cost.
    """
    from core.constants import g0   # 9.80665 m/s^2

    ht = hohmann_transfer(r_earth, r_mars)

    # Convert to m/s
    dv1_ms   = ht['delta_v1_auday']          * _AU_DAY_TO_MS
    dv2_ms   = ht['delta_v2_auday']          * _AU_DAY_TO_MS
    total_ms = ht['total_delta_v_auday']     * _AU_DAY_TO_MS

    # Sequential Tsiolkovsky: burn 1 then burn 2
    isp = config.isp_s                   # [s]
    m0  = config.initial_mass_kg         # [kg]
    ve  = g0 * isp                       # effective exhaust speed [m/s]
    m1  = m0 * np.exp(-dv1_ms / ve)     # mass after departure burn
    mf  = m1 * np.exp(-dv2_ms / ve)     # mass after arrival burn
    mp  = m0 - mf                        # total propellant consumed
    mr  = m0 / mf                        # overall mass ratio

    return {
        'tof_days':            ht['tof_days'],
        'delta_v1_ms':         dv1_ms,
        'delta_v2_ms':         dv2_ms,
        'total_delta_v_ms':    total_ms,
        'a_transfer_au':       ht['a_au'],
        'propellant_fraction': mp / m0,
        'propellant_mass_kg':  mp,
        'final_mass_kg':       mf,
        'mass_ratio':          mr,
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print('=== impulsive_transfer.py self-test ===\n')

    mu = mu_sun_au3day2

    # ------------------------------------------------------------------
    # 1. Hohmann transfer Earth -> Mars
    # ------------------------------------------------------------------
    print('Test 1: Earth-Mars Hohmann transfer')
    ht = hohmann_transfer(r_earth, r_mars)
    dv1_ms = ht['delta_v1_auday'] * _AU_DAY_TO_MS
    dv2_ms = ht['delta_v2_auday'] * _AU_DAY_TO_MS
    tot_ms = ht['total_delta_v_auday'] * _AU_DAY_TO_MS

    print(f"  Semi-major axis  : {ht['a_au']:.5f} AU  (expect {(r_earth+r_mars)/2:.5f})")
    print(f"  Transfer time    : {ht['tof_days']:.2f} days  (expect ~259.0)")
    print(f"  delta_v1         : {dv1_ms:.1f} m/s  (expect ~2945 m/s)")
    print(f"  delta_v2         : {dv2_ms:.1f} m/s  (expect ~2648 m/s)")
    print(f"  total delta_v    : {tot_ms:.1f} m/s  (expect ~5593 m/s)")

    assert abs(ht['a_au'] - (r_earth + r_mars) / 2.0) < 1e-10,  "SMA mismatch"
    assert abs(ht['tof_days'] - 258.9) < 1.0,                    f"TOF off: {ht['tof_days']:.2f}"
    assert 2800 < dv1_ms < 3100,                                  f"delta_v1 out of range: {dv1_ms:.1f}"
    assert 2500 < dv2_ms < 2800,                                  f"delta_v2 out of range: {dv2_ms:.1f}"
    assert 5200 < tot_ms  < 6000,                                 f"total dv out of range: {tot_ms:.1f}"
    print('  PASSED\n')

    # ------------------------------------------------------------------
    # 2. Vis-viva energy consistency
    # ------------------------------------------------------------------
    print('Test 2: vis-viva energy consistency')
    a_t    = ht['a_au']
    v_peri = np.sqrt(mu * (2.0 / r_earth - 1.0 / a_t))
    v_apo  = np.sqrt(mu * (2.0 / r_mars  - 1.0 / a_t))
    E_peri = 0.5 * v_peri ** 2 - mu / r_earth
    E_apo  = 0.5 * v_apo  ** 2 - mu / r_mars
    E_ref  = -mu / (2.0 * a_t)
    print(f"  E at periapsis : {E_peri:.8e}  AU^2/day^2")
    print(f"  E at apoapsis  : {E_apo:.8e}   AU^2/day^2")
    print(f"  E from SMA     : {E_ref:.8e}   AU^2/day^2")
    print(f"  Max deviation  : {max(abs(E_peri-E_ref), abs(E_apo-E_ref)):.2e}  (expect < 1e-12)")
    assert abs(E_peri - E_ref) < 1e-12, "Periapsis energy mismatch"
    assert abs(E_apo  - E_ref) < 1e-12, "Apoapsis energy mismatch"
    print('  PASSED\n')

    # ------------------------------------------------------------------
    # 3. Input validation
    # ------------------------------------------------------------------
    print('Test 3: input validation')
    for bad_args, label in [
        ((0.0, 1.5),  'r1=0'),
        ((1.0, -0.5), 'r2<0'),
        ((1.0, 1.0),  'r1==r2'),
    ]:
        try:
            hohmann_transfer(*bad_args)
            assert False, f"Should have raised ValueError for {label}"
        except ValueError as e:
            print(f"  Caught expected error ({label}): {e}")
    print('  PASSED\n')

    # ------------------------------------------------------------------
    # 4. compute_baseline
    # ------------------------------------------------------------------
    print('Test 4: compute_baseline with default spacecraft params')
    from config.mission_config import MissionConfig

    cfg = MissionConfig(
        initial_mass_kg      = 5000.0,
        thrust_N             = 3.5,
        isp_s                = 3000.0,
        opt_mode             = 'time_optimal',
        n_segments           = 10,
        time_guess_days      = 200.0,
        time_lb_days         = 175.0,
        time_ub_days         = 225.0,
        throttle_guess       = 1.0,
        throttle_lb          = 1.0,
        throttle_ub          = 1.0,
        alpha_guess_deg      = 0.0,
        alpha_lb_deg         = -180.0,
        alpha_ub_deg         =  180.0,
        departure_date_guess = '2020-01-01',
    )

    bl = compute_baseline(cfg)

    required_keys = {
        'tof_days', 'delta_v1_ms', 'delta_v2_ms', 'total_delta_v_ms',
        'a_transfer_au', 'propellant_fraction', 'propellant_mass_kg',
        'final_mass_kg', 'mass_ratio',
    }
    assert required_keys.issubset(bl.keys()), f"Missing keys: {required_keys - bl.keys()}"

    print(f"  tof_days            : {bl['tof_days']:.2f} d")
    print(f"  delta_v1            : {bl['delta_v1_ms']:.1f} m/s")
    print(f"  delta_v2            : {bl['delta_v2_ms']:.1f} m/s")
    print(f"  total_delta_v       : {bl['total_delta_v_ms']:.1f} m/s")
    print(f"  transfer SMA        : {bl['a_transfer_au']:.5f} AU")
    print(f"  propellant fraction : {bl['propellant_fraction']:.4f}  ({bl['propellant_fraction']*100:.2f}%)")
    print(f"  propellant mass     : {bl['propellant_mass_kg']:.1f} kg")
    print(f"  final mass          : {bl['final_mass_kg']:.1f} kg")
    print(f"  mass ratio m0/mf    : {bl['mass_ratio']:.4f}")

    assert bl['tof_days'] > 250 and bl['tof_days'] < 270, "TOF not in expected range"
    assert bl['total_delta_v_ms'] > 5000,                  "Total dv too low"
    assert 0 < bl['propellant_fraction'] < 1,              "Propellant fraction unphysical"
    mass_check = abs(bl['propellant_mass_kg'] + bl['final_mass_kg'] - cfg.initial_mass_kg)
    assert mass_check < 1e-6,                              f"Mass budget violated: {mass_check:.2e}"
    print('  PASSED\n')

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print('=== Earth-Mars Hohmann Baseline Summary ===')
    print(f"  Transfer time   : {bl['tof_days']:.1f} days")
    print(f"  delta_v1 (dep)  : {bl['delta_v1_ms']/1e3:.3f} km/s")
    print(f"  delta_v2 (arr)  : {bl['delta_v2_ms']/1e3:.3f} km/s")
    print(f"  Total delta_v   : {bl['total_delta_v_ms']/1e3:.3f} km/s")
    print(f"  Propellant used : {bl['propellant_mass_kg']:.1f} kg "
          f"({bl['propellant_fraction']*100:.1f}% of {cfg.initial_mass_kg:.0f} kg)")
    print(f"  Delivered mass  : {bl['final_mass_kg']:.1f} kg")
