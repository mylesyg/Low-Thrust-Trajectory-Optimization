"""Planetary gravitational parameters and reference radii.

Provides constants for Earth and Mars used by the planetocentric spiral
optimizer.  SI (km, s) units are primary; AU-day^2 equivalents are derived
for reference.

    Constant              Value                   Source
    ----------------      ----------------------  ----------------------------
    mu_earth_km3s2        398600.4418 km^3/s^2    IAU 2009 / astropy GM_earth
    mu_mars_km3s2         42828.375816 km^3/s^2   NASA JPL DE430
    R_earth_km            6371.0 km               IUGG mean radius
    R_mars_km             3389.5 km               IAU 2009 mean radius
    soi_earth_km          925 000 km              Hill-sphere approximation
    soi_mars_km           577 000 km              Hill-sphere approximation
"""

import astropy.constants as _ac
import astropy.units as _u

from .constants import AU_km

# ---------------------------------------------------------------------------
# Earth
# ---------------------------------------------------------------------------
# IAU 2015 geocentric gravitational constant via astropy
mu_earth_km3s2: float = _ac.GM_earth.to(_u.km ** 3 / _u.s ** 2).value
# ~ 398 600.4418 km^3/s^2

# Mean equatorial radius (IUGG)
R_earth_km: float = 6371.0   # km

# Earth sphere of influence (Hill-sphere approximation vs. Sun)
soi_earth_km: float = 925_000.0   # km

# ---------------------------------------------------------------------------
# Mars
# ---------------------------------------------------------------------------
# NASA/JPL DE430 value — astropy does not expose GM_mars in all versions
mu_mars_km3s2: float = 42828.375816   # km^3/s^2

# IAU 2009 mean radius
R_mars_km: float = 3389.5   # km

# Mars sphere of influence (Hill-sphere approximation vs. Sun)
soi_mars_km: float = 577_000.0   # km

# ---------------------------------------------------------------------------
# AU-day^2 equivalents  (for cross-checking with heliocentric solver)
# ---------------------------------------------------------------------------
_S_PER_DAY: float = 86_400.0

mu_earth_au3day2: float = mu_earth_km3s2 * (_S_PER_DAY ** 2) / (AU_km ** 3)
mu_mars_au3day2:  float = mu_mars_km3s2  * (_S_PER_DAY ** 2) / (AU_km ** 3)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print('=== planetary_constants.py self-test ===\n')
    print(f'mu_earth_km3s2  = {mu_earth_km3s2:.4f}  km^3/s^2  (expect ~398600.4)')
    print(f'mu_mars_km3s2   = {mu_mars_km3s2:.6f} km^3/s^2  (expect 42828.375816)')
    print(f'R_earth_km      = {R_earth_km}  km')
    print(f'R_mars_km       = {R_mars_km}   km')
    print(f'soi_earth_km    = {soi_earth_km:.0f}  km')
    print(f'soi_mars_km     = {soi_mars_km:.0f}   km')
    print(f'mu_earth_au3day2= {mu_earth_au3day2:.6e}  AU^3/day^2')
    print(f'mu_mars_au3day2 = {mu_mars_au3day2:.6e}  AU^3/day^2')

    # Sanity: circular speed at Earth surface ~ 7.9 km/s
    import math
    v_leo = math.sqrt(mu_earth_km3s2 / R_earth_km)
    print(f'\nv_circ at Earth surface = {v_leo:.2f} km/s  (expect ~7.91)')
    assert 7.8 < v_leo < 8.1, 'Earth surface circular speed out of range'

    # Sanity: circular speed at Mars surface ~ 3.6 km/s
    v_lmo = math.sqrt(mu_mars_km3s2 / R_mars_km)
    print(f'v_circ at Mars surface  = {v_lmo:.2f} km/s  (expect ~3.55)')
    assert 3.4 < v_lmo < 3.7, 'Mars surface circular speed out of range'

    print('\nPASSED')
