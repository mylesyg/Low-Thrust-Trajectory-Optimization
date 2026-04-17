"""Planetary state vectors and departure date estimation for Earth-Mars transfers.

Replicates and extends the functionality of departure.m and julian.m from the
MATLAB reference code.  The dynamical model assumes circular, coplanar orbits
for both Earth and Mars (consistent with core/equations_of_motion.py), while
astropy is used for accurate calendar date handling.

Coordinate conventions
----------------------
All angles are heliocentric ecliptic longitudes measured from the vernal
equinox (J2000.0 ecliptic frame).  Non-dimensional states follow the scheme
in core.nondimensional:

    Distance unit : 1 AU  (r* = 1 AU)
    Time unit     : t_cf = sqrt(1/mu_sun)  ~58.13 days
    Velocity unit : AU / t_cf  (circular speed at 1 AU = 1 exactly)

References
----------
Longitude polynomials after Meeus, "Astronomical Algorithms", 2nd ed.,
Table 31.a (mean longitudes of the planets, J2000.0 epoch).
MATLAB reference: departure.m, julian.m from ilt_opt_snopt.
"""

import numpy as np
from astropy.time import Time

from core.constants import r_earth, r_mars, mu_sun_au3day2, t_cf

# ---------------------------------------------------------------------------
# Mean motion constants (circular orbit, Keplerian)
# ---------------------------------------------------------------------------
# Orbital period: tau = 2*pi * sqrt(r^3 / mu_sun)  [days]
_TAU_EARTH: float = 2.0 * np.pi * np.sqrt(r_earth**3 / mu_sun_au3day2)  # ~365.25 days
_TAU_MARS:  float = 2.0 * np.pi * np.sqrt(r_mars**3  / mu_sun_au3day2)  # ~686.97 days

# Mean angular velocity: omega = 360 / tau  [deg/day]
_OMEGA_EARTH: float = 360.0 / _TAU_EARTH   # ~0.9856 deg/day
_OMEGA_MARS:  float = 360.0 / _TAU_MARS    # ~0.5240 deg/day

# J2000.0 Julian Date epoch
_JD_J2000: float = 2451545.0

# Julian centuries per day
_DAYS_PER_JC: float = 36525.0


def get_planet_longitude(planet_name: str, jd: float) -> float:
    """Return the approximate heliocentric ecliptic longitude of a planet [deg].

    Uses low-precision polynomial expressions in Julian centuries from J2000.0.
    Accurate to within ~1 degree over the range 1800–2050.

    Parameters
    ----------
    planet_name : str
        'earth' or 'mars' (case-insensitive).
    jd : float
        Julian Date (TDB/TT).

    Returns
    -------
    longitude : float
        Heliocentric ecliptic longitude in degrees, in [0, 360).

    Raises
    ------
    ValueError
        If planet_name is not 'earth' or 'mars'.

    Notes
    -----
    Julian centuries from J2000.0:
        T = (JD - 2451545.0) / 36525.0

    Polynomial expressions (Meeus, Astronomical Algorithms, 2nd ed.):
        Earth : L = 100.466449 + 35999.3728519*T - 0.00000568*T^2  [deg]
        Mars  : L = 355.433275 + 19140.2993313*T + 0.00000261*T^2  [deg]

    Result is reduced modulo 360 to [0, 360).
    """
    T = (jd - _JD_J2000) / _DAYS_PER_JC   # Julian centuries from J2000.0

    name = planet_name.lower()
    if name == 'earth':
        lon = 100.466449 + 35999.3728519 * T - 0.00000568 * T**2
    elif name == 'mars':
        lon = 355.433275 + 19140.2993313 * T + 0.00000261 * T**2
    else:
        raise ValueError(
            f"Unsupported planet '{planet_name}'. Choose 'earth' or 'mars'."
        )

    return lon % 360.0


def compute_departure_date(
    jd_guess: float,
    transfer_time_days: float,
    transfer_angle_deg: float,
) -> tuple:
    """Estimate the departure Julian Date using the orbital phasing equation.

    Finds the departure epoch at which Earth and Mars are in the correct
    angular relationship so that a transfer of duration `transfer_time_days`
    spanning `transfer_angle_deg` of heliocentric arc connects the two
    planets.  Uses a first-order linearization about `jd_guess`.

    Phasing derivation
    ------------------
    At departure JD `jd_dep`, Earth is at longitude theta_E.  The spacecraft
    travels theta_T degrees and arrives at `jd_dep + t_T`.  For Mars to be
    at the arrival longitude:

        theta_M(jd_dep) + omega_M * t_T  =  theta_E(jd_dep) + theta_T

    Linearising about jd_guess (setting jd_dep = jd_guess + delta_t):

        delta_t = (theta_M + omega_M * t_T - theta_E - theta_T)
                  / (omega_E - omega_M)

    where theta_E, theta_M are evaluated at jd_guess and omega = 360/tau.
    This is a first-order estimate; the result is exact in the limit where
    the longitude polynomials are linear in time (a good approximation over
    short intervals of a few synodic periods).

    Parameters
    ----------
    jd_guess : float
        Julian Date of the initial estimate for the departure epoch.
        A reasonable guess is any date near a known launch window
        (e.g. JD 2460000, which is around 2023 May 26).
    transfer_time_days : float
        Assumed transfer duration t_T [days].
    transfer_angle_deg : float
        Heliocentric arc swept by the spacecraft during transfer [deg].
        For a classical Hohmann transfer: ~180 deg.
        For low-thrust trajectories: typically 200–500 deg.

    Returns
    -------
    jd_depart : float
        Estimated departure Julian Date.
    jd_arrive : float
        Estimated arrival Julian Date = jd_depart + transfer_time_days.
    synodic_period_days : float
        Earth-Mars synodic period [days], returned for reference.

    Notes
    -----
    If |delta_t| > synodic_period, the departure opportunity lies more than
    one synodic period away from jd_guess.  Consider iterating with the
    returned jd_depart as the new guess, or choose a jd_guess closer to
    the actual launch window.
    """
    theta_earth = get_planet_longitude('earth', jd_guess)
    theta_mars  = get_planet_longitude('mars',  jd_guess)
    syn_days, _ = compute_synodic_period()

    delta_t = (
        theta_mars + _OMEGA_MARS * transfer_time_days - theta_earth - transfer_angle_deg
    ) / (_OMEGA_EARTH - _OMEGA_MARS)

    jd_depart = jd_guess + delta_t
    jd_arrive = jd_depart + transfer_time_days

    return jd_depart, jd_arrive, syn_days


def jd_to_calendar_string(jd: float) -> str:
    """Convert a Julian Date to a human-readable UTC calendar date string.

    Parameters
    ----------
    jd : float
        Julian Date.

    Returns
    -------
    date_str : str
        Calendar date formatted as 'YYYY-MMM-DD HH:MM' (UTC), e.g.
        '2026-Jan-15 06:00'.

    Examples
    --------
    >>> jd_to_calendar_string(2451545.0)
    '2000-Jan-01 12:00'
    """
    t = Time(jd, format='jd', scale='utc')
    # iso format: '2000-01-01 12:00:00.000' -> convert month to abbreviated name
    dt = t.to_datetime()
    month_abbr = dt.strftime('%b')      # 'Jan', 'Feb', ...
    return dt.strftime(f'%Y-{month_abbr}-%d %H:%M')


def compute_synodic_period() -> tuple:
    """Return the Earth-Mars synodic period.

    The synodic period is the time between successive launch windows, derived
    from the difference in mean angular velocities:

        1 / T_syn = 1 / T_earth - 1 / T_mars
        T_syn = T_earth * T_mars / (T_mars - T_earth)

    Returns
    -------
    syn_days : float
        Synodic period [days].  Expected value ~779.95 days.
    syn_years : float
        Synodic period [Julian years of 365.25 days].  Expected ~2.135 years.

    Notes
    -----
    Uses the Keplerian orbital periods derived from the circular-orbit radii
    and mu_sun_au3day2 in core.constants, so the result is fully consistent
    with the dynamical model.
    """
    syn_days = _TAU_EARTH * _TAU_MARS / (_TAU_MARS - _TAU_EARTH)
    syn_years = syn_days / 365.25
    return syn_days, syn_years


def get_initial_state_earth() -> np.ndarray:
    """Return the non-dimensional initial state for an Earth departure.

    Assumes a circular, prograde heliocentric orbit at r = 1 AU with the
    spacecraft initially on the x-axis (theta = 0).  No propellant has been
    consumed and delta-V is zero at the start.

    Returns
    -------
    y0 : np.ndarray, shape (6,)
        [r, u, v, theta, mp, acc_dv]

        r     = 1.0       (Earth orbital radius in AU, ND = same value)
        u     = 0.0       (no radial velocity in circular orbit)
        v     = 1.0       (circular speed at r=1: v = sqrt(mu*/r) = 1)
        theta = 0.0       (reference departure longitude [rad])
        mp    = 0.0       (no propellant consumed yet)
        acc_dv= 0.0       (no delta-V accumulated yet)

    Notes
    -----
    The departure polar angle theta = 0 defines the reference direction.
    The Mars arrival angle theta_f is unconstrained in the optimizer (free
    final angle), so only r_f, u_f, v_f are enforced as terminal constraints.
    """
    return np.array([r_earth, 0.0, 1.0, 0.0, 0.0, 0.0])


def get_final_state_mars() -> np.ndarray:
    """Return the non-dimensional target state for a Mars orbit insertion.

    Specifies the desired terminal conditions for a circular, prograde orbit
    at Mars's mean orbital radius.  The polar angle theta_f is left as NaN
    to indicate that it is a free variable (unconstrained by the optimizer).

    Returns
    -------
    y_f : np.ndarray, shape (6,)
        [r_f, u_f, v_f, theta_f, mp_f, acc_dv_f]

        r_f    = 1.52368            (Mars orbital radius, ND)
        u_f    = 0.0                (circular orbit: no radial velocity)
        v_f    = sqrt(1/r_mars)     (circular transverse speed at Mars, ND)
                 ~0.8101 in ND units
        theta_f= nan                (free; optimizer does not constrain angle)
        mp_f   = nan                (free; final propellant mass is a result)
        acc_dv_f = nan              (free; accumulated delta-V is a result)

    Notes
    -----
    The enforced terminal constraints are [r_f, u_f, v_f].  The NaN values
    signal to the optimizer/constraint builder that these states are free.
    Use ``terminal_conditions_mars()`` from core.equations_of_motion for the
    dict form expected by the constraint functions.
    """
    v_f = np.sqrt(1.0 / r_mars)     # = sqrt(mu* / r_mars) with mu* = 1
    return np.array([r_mars, 0.0, v_f, np.nan, np.nan, np.nan])


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("=== planetary_states.py self-test ===\n")

    # ------------------------------------------------------------------
    # 1. Synodic period
    # ------------------------------------------------------------------
    print("Test 1: Earth-Mars synodic period")
    syn_d, syn_y = compute_synodic_period()
    print(f"  Orbital period Earth : {_TAU_EARTH:.4f} days  (expect ~365.25)")
    print(f"  Orbital period Mars  : {_TAU_MARS:.4f} days  (expect ~686.97)")
    print(f"  Synodic period       : {syn_d:.4f} days  (expect ~779.95)")
    print(f"  Synodic period       : {syn_y:.4f} years  (expect ~2.135)")
    assert abs(syn_d - 779.95) < 1.0, f"Synodic period off: {syn_d:.2f}"
    print("  PASSED\n")

    # ------------------------------------------------------------------
    # 2. Longitude at J2000.0
    # ------------------------------------------------------------------
    print("Test 2: heliocentric longitudes at J2000.0")
    jd2000 = _JD_J2000
    lon_e  = get_planet_longitude('earth', jd2000)
    lon_m  = get_planet_longitude('mars',  jd2000)
    print(f"  Earth lon at J2000.0 : {lon_e:.4f} deg  (expect ~100.47)")
    print(f"  Mars  lon at J2000.0 : {lon_m:.4f} deg  (expect ~355.43)")
    assert abs(lon_e - 100.466449) < 0.01, f"Earth longitude off: {lon_e:.4f}"
    assert abs(lon_m - 355.433275) < 0.01, f"Mars longitude off: {lon_m:.4f}"
    print("  PASSED\n")

    # ------------------------------------------------------------------
    # 3. Julian Date to calendar string
    # ------------------------------------------------------------------
    print("Test 3: JD to calendar string")
    s = jd_to_calendar_string(_JD_J2000)
    print(f"  JD 2451545.0 -> '{s}'  (expect 2000-Jan-01 12:00)")
    assert '2000' in s and 'Jan' in s
    print("  PASSED\n")

    # ------------------------------------------------------------------
    # 4. Departure date estimation (Hohmann-like, ~180-deg transfer)
    # ------------------------------------------------------------------
    print("Test 4: departure date for 180-deg Hohmann-like transfer")
    # Use a guess near the 2026 launch window (around JD 2461200)
    jd_guess        = 2461200.0
    tof_days        = 259.0       # ~Hohmann transfer time (days)
    transfer_angle  = 180.0       # deg (half-ellipse)

    jd_dep, jd_arr, syn = compute_departure_date(jd_guess, tof_days, transfer_angle)
    date_dep = jd_to_calendar_string(jd_dep)
    date_arr = jd_to_calendar_string(jd_arr)
    print(f"  JD guess   : {jd_guess:.1f}  -> {jd_to_calendar_string(jd_guess)}")
    print(f"  Depart     : JD {jd_dep:.1f}  -> {date_dep}")
    print(f"  Arrive     : JD {jd_arr:.1f}  -> {date_arr}")
    print(f"  delta_t    : {jd_dep - jd_guess:+.1f} days from guess")
    # Verify phasing: at arrival, Mars should be ~180 deg ahead of Earth at departure
    lon_e_dep = get_planet_longitude('earth', jd_dep)
    lon_m_arr = get_planet_longitude('mars',  jd_arr)
    required  = (lon_e_dep + transfer_angle) % 360.0
    print(f"  Earth lon at depart  : {lon_e_dep:.2f} deg")
    print(f"  Mars  lon at arrive  : {lon_m_arr:.2f} deg")
    print(f"  Required Mars arrival: {required:.2f} deg")
    phasing_error = abs((lon_m_arr - required + 180) % 360 - 180)
    print(f"  Phasing error        : {phasing_error:.4f} deg")
    assert phasing_error < 2.0, f"Phasing error too large: {phasing_error:.4f} deg"
    print("  PASSED\n")

    # ------------------------------------------------------------------
    # 5. Initial and final state vectors
    # ------------------------------------------------------------------
    print("Test 5: initial and final state vectors")
    y0 = get_initial_state_earth()
    yf = get_final_state_mars()
    print(f"  y0 (Earth) : {y0}")
    print(f"  yf (Mars)  : {yf}")
    assert y0[0] == 1.0    and y0[1] == 0.0 and y0[2] == 1.0
    assert yf[0] == r_mars and yf[1] == 0.0
    assert abs(yf[2] - np.sqrt(1.0 / r_mars)) < 1e-12
    assert np.isnan(yf[3]) and np.isnan(yf[4]) and np.isnan(yf[5])
    print("  PASSED")
