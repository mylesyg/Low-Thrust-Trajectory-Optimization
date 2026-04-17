"""Physical and astronomical constants for low-thrust trajectory optimization.

All values are exposed as plain Python floats (units stripped) so they can be
used directly inside numerical solvers without astropy overhead at call time.
"""

import numpy as np
import astropy.constants as const
import astropy.units as u

# ---------------------------------------------------------------------------
# Gravitational parameter of the Sun (GM_sun)
# ---------------------------------------------------------------------------
_GM_sun = const.GM_sun  # astropy Constant, IAU 2015 value [m^3/s^2]

# In km^3/s^2  (canonical form used in most astrodynamics references)
mu_sun_km3s2: float = _GM_sun.to(u.km**3 / u.s**2).value

# In AU^3/day^2  (natural units for heliocentric propagation)
# Matches MATLAB value: smu = 0.2959122082855912e-3
mu_sun_au3day2: float = _GM_sun.to(u.au**3 / u.day**2).value

# ---------------------------------------------------------------------------
# Length scales
# ---------------------------------------------------------------------------
# 1 AU in kilometres (IAU 2012 exact definition via astropy)
AU_km: float = (1.0 * u.au).to(u.km).value          # ~ 149 597 870.691 km

# 1 AU in metres (used for unit conversions inside numerical routines)
AU_m: float = AU_km * 1e3                             # ~ 1.496e11 m

# ---------------------------------------------------------------------------
# Orbital semi-major axes (mean values, circular-orbit assumption)
# ---------------------------------------------------------------------------
r_earth: float = 1.0       # AU
r_mars:  float = 1.52368   # AU

# ---------------------------------------------------------------------------
# Standard gravity
# ---------------------------------------------------------------------------
g0: float = 9.80665        # m/s^2  (exact, SI definition)

# ---------------------------------------------------------------------------
# Non-dimensional time unit
# ---------------------------------------------------------------------------
# t_cf = sqrt(1 / mu_sun [AU^3/day^2])
# This is the time normalization factor T* such that the ND gravitational
# parameter mu* = 1.  Numerically ~ 58.13 days.
# Matches MATLAB: tfactor = sqrt(1/smu)
t_cf: float = np.sqrt(1.0 / mu_sun_au3day2)          # days


if __name__ == "__main__":
    print(f"mu_sun  = {mu_sun_au3day2:.16e}  AU^3/day^2")
    print(f"mu_sun  = {mu_sun_km3s2:.6e}     km^3/s^2")
    print(f"AU      = {AU_km:.3f}            km")
    print(f"g0      = {g0}              m/s^2")
    print(f"t_cf    = {t_cf:.6f}             days")
