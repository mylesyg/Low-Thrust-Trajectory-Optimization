"""Pork chop plot (C3 and arrival delta-V) for Earth-to-Mars launch windows.

Functions
---------
compute_porkchop_data
    Evaluate departure C3 and arrival delta-V on a (departure, arrival) date
    grid by solving Lambert's problem at every grid point.
plot_porkchop
    Two-panel figure: departure C3 [km²/s²] (left) and arrival ΔV [km/s] (right),
    each with filled contours, iso-contour labels, and constant-TOF diagonals.
"""

import os
import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt

# Allow direct execution from the project root:
#   python visualization/porkchop_plot.py
if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.constants import mu_sun_au3day2, AU_km, r_mars
from baseline.lambert_solver import lambert_solve
from ephemeris.planetary_states import get_planet_longitude, jd_to_calendar_string


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_AU_DAY_TO_KMS: float = AU_km / 86_400.0   # 1 AU/day → km/s  (~1 731.5 km/s)
_C3_CLIP:       float = 200.0              # km²/s²  — clip above this to NaN
_DV_CLIP:       float = 15.0              # km/s    — clip above this to NaN
_TOF_MIN:       float = 50.0              # days    — minimum physical transfer
_JD_J2000:      float = 2_451_545.0       # Julian Date of J2000.0 epoch


# ===========================================================================
# Public API
# ===========================================================================

def compute_porkchop_data(
    departure_jd_range: list,
    arrival_jd_range:   list,
    n_departure:        int = 50,
    n_arrival:          int = 50,
) -> dict:
    """Compute departure C3 and arrival ΔV on a 2-D (departure, arrival) grid.

    For every (jd_dep, jd_arr) grid point, Lambert's problem is solved between
    Earth's heliocentric position at jd_dep and Mars's position at jd_arr.
    Planet positions use the same low-precision mean-longitude polynomials as
    the rest of the mission model (see ``ephemeris.planetary_states``).

    Parameters
    ----------
    departure_jd_range : list[float, float]
        [jd_min, jd_max] Julian Date range for the departure axis.
    arrival_jd_range : list[float, float]
        [jd_min, jd_max] Julian Date range for the arrival axis.
    n_departure : int, optional
        Number of evenly-spaced departure grid points.  Default 50.
    n_arrival : int, optional
        Number of evenly-spaced arrival grid points.  Default 50.

    Returns
    -------
    data : dict
        'departure_jd'       : ndarray (n_departure,)         — departure JDs
        'arrival_jd'         : ndarray (n_arrival,)           — arrival JDs
        'C3_depart'          : ndarray (n_arrival, n_departure) — C3 [km²/s²]
        'dv_arrive'          : ndarray (n_arrival, n_departure) — arrival ΔV [km/s]
        'tof_days'           : ndarray (n_arrival, n_departure) — TOF [days]
        'departure_dates_str': list[str] — calendar strings for x-axis ticks
        'arrival_dates_str'  : list[str] — calendar strings for y-axis ticks

    Notes
    -----
    Grid conventions
        Row index i ↔ arrival date arr_jd[i]   (y-axis)
        Column index j ↔ departure date dep_jd[j]  (x-axis)

    Clipping
        C3 > 100 km²/s² and arrival ΔV > 15 km/s are set to NaN so the
        colour scale focuses on the mission-feasible launch window.

    Failed Lambert solves (very short TOF, near-180° geometry, etc.) are
    caught silently and stored as NaN.
    """
    mu = mu_sun_au3day2
    v_circ_earth = np.sqrt(mu / 1.0)       # AU/day  (r_earth = 1 AU)
    v_circ_mars  = np.sqrt(mu / r_mars)    # AU/day

    dep_jd = np.linspace(departure_jd_range[0], departure_jd_range[1], n_departure)
    arr_jd = np.linspace(arrival_jd_range[0],   arrival_jd_range[1],   n_arrival)

    # Pre-compute planet longitudes (in radians) for all grid dates
    earth_lons_rad = np.radians(
        [get_planet_longitude('earth', jd) for jd in dep_jd]
    )
    mars_lons_rad = np.radians(
        [get_planet_longitude('mars', jd) for jd in arr_jd]
    )

    # Output arrays — shape (n_arrival, n_departure)
    C3_depart = np.full((n_arrival, n_departure), np.nan)
    dv_arrive = np.full((n_arrival, n_departure), np.nan)
    tof_grid  = np.full((n_arrival, n_departure), np.nan)

    for j in range(n_departure):
        th_e = earth_lons_rad[j]
        cos_te, sin_te = np.cos(th_e), np.sin(th_e)

        # Earth heliocentric position and circular-orbit velocity at departure
        r1  = np.array([cos_te,                    sin_te,                0.0])
        v_e = np.array([-sin_te * v_circ_earth,  cos_te * v_circ_earth,  0.0])

        for i in range(n_arrival):
            tof = float(arr_jd[i] - dep_jd[j])
            if tof < _TOF_MIN:
                continue                          # unphysical short transfer

            tof_grid[i, j] = tof

            th_m = mars_lons_rad[i]
            cos_tm, sin_tm = np.cos(th_m), np.sin(th_m)

            # Mars heliocentric position and circular-orbit velocity at arrival
            r2  = np.array([r_mars * cos_tm,        r_mars * sin_tm,       0.0])
            v_m = np.array([-sin_tm * v_circ_mars,  cos_tm * v_circ_mars,  0.0])

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    sol = lambert_solve(r1, r2, tof, mu)

                # Departure v_infinity and C3
                dv_d_kms = np.linalg.norm(sol['v1'] - v_e) * _AU_DAY_TO_KMS
                # Arrival delta-V
                dv_a_kms = np.linalg.norm(sol['v2'] - v_m) * _AU_DAY_TO_KMS

                C3_depart[i, j] = dv_d_kms ** 2
                dv_arrive[i, j] = dv_a_kms

            except Exception:
                pass      # failed Lambert solve → remains NaN

    # Clip extreme values to keep the colour scale on the feasible window
    C3_depart[C3_depart > _C3_CLIP] = np.nan
    dv_arrive[dv_arrive > _DV_CLIP] = np.nan

    return {
        'departure_jd':       dep_jd,
        'arrival_jd':         arr_jd,
        'C3_depart':          C3_depart,
        'dv_arrive':          dv_arrive,
        'tof_days':           tof_grid,
        'departure_dates_str': [jd_to_calendar_string(jd) for jd in dep_jd],
        'arrival_dates_str':   [jd_to_calendar_string(jd) for jd in arr_jd],
    }


def plot_porkchop(
    porkchop_data:  dict,
    optimal_result: dict = None,
    save_path:      str  = None,
) -> plt.Figure:
    """Generate a two-panel Earth-to-Mars pork chop plot.

    Left panel  : Departure C3 [km²/s²] — viridis filled contours with
                  labelled black iso-lines and constant-TOF diagonal guides.
    Right panel : Arrival ΔV [km/s] — plasma filled contours with the same
                  overlays.

    Parameters
    ----------
    porkchop_data : dict
        Output from :func:`compute_porkchop_data`.
    optimal_result : dict, optional
        ``{'departure_jd': float, 'arrival_jd': float}`` — marks the
        low-thrust optimal point on both panels with a white star and label.
    save_path : str, optional
        Full file path for saving (PNG/PDF, dpi=150).  Parent directory is
        created if needed.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    dep_jd = porkchop_data['departure_jd']   # (n_departure,)
    arr_jd = porkchop_data['arrival_jd']     # (n_arrival,)
    C3     = porkchop_data['C3_depart']      # (n_arrival, n_departure)
    dv     = porkchop_data['dv_arrive']      # (n_arrival, n_departure)

    # Meshgrid for contour plotting (x = departure JD, y = arrival JD)
    X, Y = np.meshgrid(dep_jd, arr_jd)

    tof_diag_days = [100, 150, 200, 250, 300, 350]  # constant-TOF guides

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ------------------------------------------------------------------ #
    # Left panel — Departure C3                                            #
    # ------------------------------------------------------------------ #
    ax = axes[0]

    cf1 = ax.contourf(X, Y, C3,
                      levels=np.linspace(0.0, _C3_CLIP, 51),
                      cmap='viridis', extend='max')
    cb1 = fig.colorbar(cf1, ax=ax, pad=0.02, fraction=0.046)
    cb1.set_label('C3 [km²/s²]', fontsize=11)

    cs1 = ax.contour(X, Y, C3,
                     levels=[5, 10, 15, 20, 30, 40, 60, 80],
                     colors='black', linewidths=0.8, alpha=0.85)
    ax.clabel(cs1, fmt='%g', fontsize=7, inline=True)

    _draw_tof_lines(ax, dep_jd, arr_jd, tof_diag_days)
    _mark_optimal(ax, optimal_result)

    ax.set_title('Departure C3 [km²/s²]', fontsize=13)
    ax.set_xlabel('Departure Date', fontsize=12)
    ax.set_ylabel('Arrival Date', fontsize=12)
    _apply_date_ticks(ax, dep_jd, arr_jd)
    ax.grid(False)

    # ------------------------------------------------------------------ #
    # Right panel — Arrival delta-V                                        #
    # ------------------------------------------------------------------ #
    ax = axes[1]

    cf2 = ax.contourf(X, Y, dv,
                      levels=np.linspace(0.0, _DV_CLIP, 51),
                      cmap='plasma', extend='max')
    cb2 = fig.colorbar(cf2, ax=ax, pad=0.02, fraction=0.046)
    cb2.set_label('Arrival ΔV [km/s]', fontsize=11)

    cs2 = ax.contour(X, Y, dv,
                     levels=[1, 2, 3, 4, 5, 6, 8, 10],
                     colors='black', linewidths=0.8, alpha=0.85)
    ax.clabel(cs2, fmt='%g', fontsize=7, inline=True)

    _draw_tof_lines(ax, dep_jd, arr_jd, tof_diag_days)
    _mark_optimal(ax, optimal_result)

    ax.set_title('Arrival ΔV [km/s]', fontsize=13)
    ax.set_xlabel('Departure Date', fontsize=12)
    ax.set_ylabel('Arrival Date', fontsize=12)
    _apply_date_ticks(ax, dep_jd, arr_jd)
    ax.grid(False)

    # ------------------------------------------------------------------ #
    # Figure-level title and layout                                        #
    # ------------------------------------------------------------------ #
    fig.suptitle('Earth-to-Mars Pork Chop Plot', fontsize=15, fontweight='bold')
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# ===========================================================================
# Private helpers
# ===========================================================================

def _jd_ticks(jd_array: np.ndarray, step_days: float = 90.0) -> np.ndarray:
    """Return JD tick values spaced *step_days* apart within *jd_array* bounds.

    Ticks are aligned to multiples of *step_days* measured from J2000.0 so
    they fall on consistent calendar-round dates regardless of the input range.
    """
    jd_min, jd_max = float(jd_array[0]), float(jd_array[-1])
    first = _JD_J2000 + np.ceil((jd_min - _JD_J2000) / step_days) * step_days
    ticks = np.arange(first, jd_max + step_days * 0.5, step_days)
    ticks = ticks[(ticks >= jd_min) & (ticks <= jd_max)]
    # Fallback: always include at least first and last values
    if len(ticks) == 0:
        ticks = np.array([jd_min, jd_max])
    return ticks


def _apply_date_ticks(
    ax:        plt.Axes,
    dep_jd:    np.ndarray,
    arr_jd:    np.ndarray,
    step_days: float = 90.0,
) -> None:
    """Set x / y tick positions to JD values and label them as calendar dates.

    Labels are formatted as 'YYYY-MMM-DD' (first 11 chars of the string
    returned by :func:`jd_to_calendar_string`).  X-labels are rotated 45°.
    """
    xt = _jd_ticks(dep_jd, step_days)
    ax.set_xticks(xt)
    ax.set_xticklabels(
        [jd_to_calendar_string(jd)[:11] for jd in xt],
        rotation=45, ha='right', fontsize=8,
    )

    yt = _jd_ticks(arr_jd, step_days)
    ax.set_yticks(yt)
    ax.set_yticklabels(
        [jd_to_calendar_string(jd)[:11] for jd in yt],
        fontsize=8,
    )


def _draw_tof_lines(
    ax:         plt.Axes,
    dep_jd:     np.ndarray,
    arr_jd:     np.ndarray,
    tof_levels: list,
) -> None:
    """Overlay constant-TOF diagonal guides on a pork chop axes.

    A line of constant TOF = T satisfies ``arrival_jd = departure_jd + T``,
    giving a unit-slope diagonal in (departure_jd, arrival_jd) space.
    Each line is clipped to the visible plot region and labelled at 65 % along
    its length with a white bold annotation.
    """
    dep_min, dep_max = float(dep_jd[0]), float(dep_jd[-1])
    arr_min, arr_max = float(arr_jd[0]), float(arr_jd[-1])

    for tof in tof_levels:
        # Intersection of y = x + T with the rectangular plot domain
        x_lo = max(dep_min, arr_min - tof)
        x_hi = min(dep_max, arr_max - tof)
        if x_lo >= x_hi:
            continue                          # line not visible in this window

        ax.plot([x_lo, x_hi], [x_lo + tof, x_hi + tof],
                'w--', linewidth=0.9, alpha=0.75, zorder=3)

        # Label at 65 % along the visible segment
        x_lbl = x_lo + 0.65 * (x_hi - x_lo)
        y_lbl = x_lbl + tof
        if dep_min <= x_lbl <= dep_max and arr_min <= y_lbl <= arr_max:
            ax.annotate(
                f'{tof:d}d',
                xy=(x_lbl, y_lbl),
                color='white', fontsize=7, fontweight='bold',
                ha='center', va='bottom', zorder=4,
            )


def _mark_optimal(ax: plt.Axes, optimal_result: dict) -> None:
    """Draw a white star and 'Optimal' label at the low-thrust optimal point."""
    if optimal_result is None:
        return
    jd_dep = optimal_result.get('departure_jd')
    jd_arr = optimal_result.get('arrival_jd')
    if jd_dep is None or jd_arr is None:
        return

    ax.plot(jd_dep, jd_arr,
            marker='*', color='white', markersize=15,
            markeredgecolor='black', markeredgewidth=0.5,
            zorder=6, label='Optimal')
    ax.annotate(
        'Optimal',
        xy=(jd_dep, jd_arr),
        xytext=(6, 4), textcoords='offset points',
        color='white', fontsize=9, fontweight='bold',
        zorder=7,
    )


# ===========================================================================
# Quick self-test
# ===========================================================================
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

    print('=== porkchop_plot.py self-test ===\n')

    # Small 10×10 grid around a 2026 Earth-Mars window
    _JD_2026 = 2_461_400.0
    dep_range = [_JD_2026 - 100.0, _JD_2026 + 100.0]
    arr_range = [dep_range[0] + 100.0, dep_range[1] + 400.0]

    # ------------------------------------------------------------------ #
    # Test 1: compute_porkchop_data                                        #
    # ------------------------------------------------------------------ #
    print('Test 1: compute_porkchop_data (10x10 grid)')
    data = compute_porkchop_data(dep_range, arr_range, n_departure=10, n_arrival=10)

    assert data['C3_depart'].shape == (10, 10), \
        f"C3 shape wrong: {data['C3_depart'].shape}"
    assert data['dv_arrive'].shape == (10, 10), \
        f"dv shape wrong: {data['dv_arrive'].shape}"
    assert data['tof_days'].shape  == (10, 10), \
        f"tof shape wrong: {data['tof_days'].shape}"
    assert len(data['departure_jd']) == 10
    assert len(data['arrival_jd'])   == 10
    assert len(data['departure_dates_str']) == 10
    assert len(data['arrival_dates_str'])   == 10

    n_finite = int(np.isfinite(data['C3_depart']).sum())
    c3_min   = float(np.nanmin(data['C3_depart'])) if n_finite > 0 else float('nan')
    c3_max   = float(np.nanmax(data['C3_depart'])) if n_finite > 0 else float('nan')
    print(f'  Finite C3 cells : {n_finite} / 100')
    print(f'  C3 range        : [{c3_min:.2f}, {c3_max:.2f}] km²/s²')
    assert n_finite > 0, 'Expected at least some finite C3 values'
    assert c3_max <= _C3_CLIP + 1e-9, 'C3 clipping failed'
    print('  PASSED\n')

    # ------------------------------------------------------------------ #
    # Test 2: plot_porkchop — figure shape                                 #
    # ------------------------------------------------------------------ #
    print('Test 2: plot_porkchop — returns Figure with >= 2 axes')
    fig = plot_porkchop(data)
    assert fig is not None
    # contourf adds subplot axes + colorbar axes: at least 2 main axes
    assert len(fig.axes) >= 2, f'Expected >=2 axes, got {len(fig.axes)}'
    plt.close(fig)
    print('  PASSED\n')

    # ------------------------------------------------------------------ #
    # Test 3: plot_porkchop with optimal_result marker                     #
    # ------------------------------------------------------------------ #
    print('Test 3: plot_porkchop with optimal_result')
    optimal = {'departure_jd': _JD_2026, 'arrival_jd': _JD_2026 + 220.0}
    fig2 = plot_porkchop(data, optimal_result=optimal)
    assert fig2 is not None
    plt.close(fig2)
    print('  PASSED\n')

    # ------------------------------------------------------------------ #
    # Test 4: plot_porkchop with save_path                                 #
    # ------------------------------------------------------------------ #
    print('Test 4: plot_porkchop with save_path')
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, 'sub', 'porkchop.png')
        fig3 = plot_porkchop(data, save_path=out)
        assert pathlib.Path(out).exists(), 'File was not created'
        plt.close(fig3)
    print('  PASSED\n')

    # ------------------------------------------------------------------ #
    # Test 5: _jd_ticks boundary handling                                  #
    # ------------------------------------------------------------------ #
    print('Test 5: _jd_ticks — at least 1 tick in a 200-day window')
    jd_test = np.linspace(_JD_2026, _JD_2026 + 200.0, 50)
    ticks = _jd_ticks(jd_test, step_days=90.0)
    assert len(ticks) >= 1, f'Expected >=1 tick, got {len(ticks)}'
    assert np.all(ticks >= jd_test[0]) and np.all(ticks <= jd_test[-1]), \
        'Ticks out of bounds'
    print(f'  Got {len(ticks)} tick(s) in 200-day window  PASSED\n')

    print('All porkchop_plot tests passed.')
