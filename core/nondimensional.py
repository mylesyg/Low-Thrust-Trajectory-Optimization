"""Non-dimensional scaling for heliocentric low-thrust trajectory optimization.

Normalization scheme (identical to MATLAB ilt_opt_snopt.m):

    Quantity    |  Dimensional unit  |  Non-dimensional unit
    ------------|--------------------|----------------------
    Distance    |  AU                |  1 AU
    Time        |  day               |  t_cf = sqrt(1/mu_sun)  ~ 58.13 day
    Velocity    |  AU/day            |  AU / t_cf
    Acceleration|  AU/day^2          |  AU / t_cf^2  = mu_sun / AU^2

The reference acceleration AU/t_cf^2 equals mu_sun/r0^2 with r0 = 1 AU, so
the non-dimensional gravitational parameter mu* = 1 exactly.

MATLAB equivalents
------------------
    acc_thrust = (0.001 * T / m0) / (xmu / aunit^2)
        — 0.001 converts T [N] / m0 [kg] from m/s^2 to km/s^2;
          xmu/aunit^2 (km^3/s^2 / km^2) is the reference acceleration in km/s^2.
          The ratio is dimensionless.

    mp_dot = (T / (9.80665 * Isp) * 86400 / m0) * tfactor
        — T/(g0*Isp) [kg/s] → *86400 → [kg/day] → /m0 → [1/day] → *tfactor → [—]
"""

import numpy as np

from .constants import mu_sun_au3day2, AU_m, g0, t_cf

# Reference acceleration [AU/day^2]:  a_ref = mu_sun / r0^2,  r0 = 1 AU
_a_ref: float = mu_sun_au3day2   # AU^3/day^2 / AU^2 = AU/day^2


class NonDimensional:
    """Static-method collection for converting between dimensional and ND units.

    All conversions assume the normalization scheme defined at module level.
    Methods accept and return plain floats (or numpy arrays where applicable).
    """

    # ------------------------------------------------------------------
    # Time
    # ------------------------------------------------------------------

    @staticmethod
    def time_to_nd(t_days: float) -> float:
        """Convert days to non-dimensional time.

        t_nd = t_days / t_cf
        """
        return t_days / t_cf

    @staticmethod
    def time_to_dim(t_nd: float) -> float:
        """Convert non-dimensional time to days.

        t_days = t_nd * t_cf
        """
        return t_nd * t_cf

    # ------------------------------------------------------------------
    # Distance
    # ------------------------------------------------------------------

    @staticmethod
    def distance_to_nd(r_au: float) -> float:
        """Convert AU to non-dimensional distance.

        Distance unit is 1 AU, so the value is unchanged numerically.
        Method provided for API symmetry with the other converters.

        r_nd = r_au / (1 AU)  →  r_nd = r_au
        """
        return r_au

    # ------------------------------------------------------------------
    # Velocity
    # ------------------------------------------------------------------

    @staticmethod
    def velocity_to_nd(v_au_per_day: float) -> float:
        """Convert AU/day to non-dimensional velocity.

        Velocity unit is AU/t_cf, so:
            v_nd = v [AU/day] / (1 AU / t_cf [day]) = v * t_cf

        Sanity check: Earth's circular speed ~ 0.01720 AU/day
            → v_nd = 0.01720 * 58.13 ≈ 1.000  (equals sqrt(mu* / r*) = 1)
        """
        return v_au_per_day * t_cf

    # ------------------------------------------------------------------
    # Thrust acceleration
    # ------------------------------------------------------------------

    @staticmethod
    def compute_thrust_acc(thrust_N: float, mass_kg: float) -> float:
        """Return non-dimensional thrust acceleration.

        Non-dimensionalization:
            acc_nd = (T/m0) [AU/day^2] / (mu_sun / r0^2) [AU/day^2]

        Steps:
            1. a_SI  = T [N] / m0 [kg]                   [m/s^2]
            2. a_nd_units = a_SI * 86400^2 / AU_m        [AU/day^2]
            3. acc_nd = a_nd_units / mu_sun               [—]

        Matches MATLAB:
            (0.001 * T / m0) / (xmu / aunit^2)
            (converts m/s^2 -> km/s^2 via 0.001;
             xmu/aunit^2 is the reference acceleration in km/s^2)

        Parameters
        ----------
        thrust_N : float
            Thrust magnitude [N].
        mass_kg : float
            Spacecraft (reference) mass [kg].

        Returns
        -------
        float
            Non-dimensional thrust acceleration [-].
        """
        a_si = thrust_N / mass_kg                              # m/s^2
        a_au_day2 = a_si * (86_400.0 ** 2) / AU_m             # AU/day^2
        return a_au_day2 / _a_ref

    # ------------------------------------------------------------------
    # Mass-flow rate
    # ------------------------------------------------------------------

    @staticmethod
    def compute_mass_flow(thrust_N: float, isp_s: float, mass_kg: float) -> float:
        """Return non-dimensional propellant mass-flow rate.

        Non-dimensionalization:
            mp_dot_nd = (T / (g0 * Isp)) [kg/s]
                        * 86400 [s/day]
                        / m0    [kg]
                        * t_cf  [day]

        The factor 86400 converts the SI mass-flow rate to a daily rate;
        dividing by m0 makes it a specific rate; multiplying by t_cf (the
        time unit) yields the non-dimensional rate used in the equations of
        motion.

        Matches MATLAB:
            (T / (9.80665 * Isp) * 86400 / m0) * tfactor

        Parameters
        ----------
        thrust_N : float
            Thrust magnitude [N].
        isp_s : float
            Specific impulse [s].
        mass_kg : float
            Spacecraft (reference) mass [kg].

        Returns
        -------
        float
            Non-dimensional mass-flow rate [-].
        """
        m_dot_kg_s = thrust_N / (g0 * isp_s)                  # kg/s
        return m_dot_kg_s * 86_400.0 / mass_kg * t_cf


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Allow direct execution: python core/nondimensional.py from the project root
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.constants import mu_sun_au3day2  # re-import after path fix

    print("=== Non-dimensional scaling self-test ===\n")

    # Spacecraft parameters (typical small SEP mission)
    T   = 0.090    # N    (90 mN thruster)
    Isp = 3000     # s
    m0  = 500.0    # kg

    acc_nd  = NonDimensional.compute_thrust_acc(T, m0)
    mdot_nd = NonDimensional.compute_mass_flow(T, Isp, m0)

    print(f"t_cf              = {t_cf:.4f} days")
    print(f"Thrust acc  [ND]  = {acc_nd:.6e}")
    print(f"Mass flow   [ND]  = {mdot_nd:.6e}")

    # Earth circular velocity check
    import math
    v_earth_au_day = math.sqrt(mu_sun_au3day2 / 1.0)   # sqrt(mu/r) at 1 AU
    print(f"\nEarth circ. vel   = {v_earth_au_day:.5f} AU/day")
    print(f"  -> v_nd         = {NonDimensional.velocity_to_nd(v_earth_au_day):.6f}  (expect 1.0)")

    t_yr = 365.25
    print(f"\n1 year            = {t_yr} days")
    print(f"  -> t_nd         = {NonDimensional.time_to_nd(t_yr):.4f}")
    print(f"  -> back to days = {NonDimensional.time_to_dim(NonDimensional.time_to_nd(t_yr)):.2f}")
