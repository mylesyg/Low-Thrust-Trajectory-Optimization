"""2D trajectory and state-history visualizations for low-thrust optimization.

Functions
---------
plot_2d_trajectory
    Heliocentric X-Y transfer orbit with planet orbit circles.
plot_state_history
    2x2 time-history panels: r(t), u(t), v(t), theta(t).
"""

import os
import sys

# Allow direct execution from the project root:
#   python visualization/trajectory_plot.py
if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from core.constants import r_earth, r_mars, t_cf


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_2d_trajectory(result_dict: dict, config, save_path: str = None,
                       n_thrust_arrows: int = 0,
                       opt_label: str = 'Time-Optimal',
                       throttle_scale: float = 1.0) -> plt.Figure:
    """Plot the heliocentric transfer orbit in the 2-D ecliptic plane.

    Converts the polar state history stored in ``result_dict`` to Cartesian
    (X, Y) coordinates and overlays the circular orbits of Earth and Mars.
    Optionally draws thrust-direction arrows at evenly-spaced points along
    the trajectory.

    Parameters
    ----------
    result_dict : dict
        Output from ``solve_time_optimal()`` or ``solve_mass_optimal()``.
        Required keys: ``'state_history'`` (N+1, 6) in ND polar coordinates,
        ``'tof_days'``.  If ``n_thrust_arrows > 0``, ``'alphas_rad'`` (N,)
        must also be present.
    config : MissionConfig
        Mission configuration used for the title annotation.
    save_path : str, optional
        Absolute or relative file path for saving the figure (PNG/PDF).
        Parent directories are created automatically.  If *None* the figure
        is returned without saving.
    n_thrust_arrows : int, optional
        Number of thrust-direction arrows to overlay along the trajectory.
        Points are evenly spaced in arc length.  Default is 0 (no arrows).
    opt_label : str, optional
        Optimization mode label shown in the figure title, e.g.
        ``'Time-Optimal'`` or ``'Mass-Optimal'``.  Default is
        ``'Time-Optimal'``.
    throttle_scale : float, optional
        Multiplier applied to the base arrow length so that arrows are
        proportional to thrust magnitude.  Set to the optimal throttle
        value for mass-optimal trajectories (< 1.0); use 1.0 for
        full-throttle time-optimal trajectories.  Default is 1.0.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    state_history = result_dict['state_history']   # (N+1, 6)  ND
    r_hist  = state_history[:, 0]                  # radial distance  [AU]
    th_hist = state_history[:, 3]                  # polar angle      [rad]

    # Polar → Cartesian [AU]
    x_traj = r_hist * np.cos(th_hist)
    y_traj = r_hist * np.sin(th_hist)

    # Planet orbit circles
    theta_c  = np.linspace(0.0, 2.0 * np.pi, 360)
    x_earth  = r_earth * np.cos(theta_c)
    y_earth  = r_earth * np.sin(theta_c)
    x_mars   = r_mars  * np.cos(theta_c)
    y_mars   = r_mars  * np.sin(theta_c)

    fig, ax = plt.subplots(figsize=(7, 7))

    # Planet orbits (dashed circles)
    ax.plot(x_earth, y_earth, color='royalblue', linewidth=1.2, linestyle='--',
            label=f'Earth orbit  (r = {r_earth:.2f} AU)', zorder=2)
    ax.plot(x_mars,  y_mars,  color='firebrick',  linewidth=1.2, linestyle='--',
            label=f'Mars orbit  (r = {r_mars:.5f} AU)', zorder=2)

    # Transfer trajectory
    ax.plot(x_traj, y_traj, color='black', linewidth=1.5,
            label='Transfer trajectory', zorder=3)

    # Thrust-direction arrow overlay
    if n_thrust_arrows > 0 and 'alphas_rad' in result_dict:
        _draw_thrust_arrows(ax, state_history, result_dict['alphas_rad'],
                            n_thrust_arrows, throttle_scale=throttle_scale)

    # Departure / arrival markers
    ax.plot(x_traj[0],  y_traj[0],  'o', color='royalblue', markersize=8,
            markeredgecolor='navy', zorder=6, label='Departure (Earth)')
    ax.plot(x_traj[-1], y_traj[-1], 's', color='firebrick',  markersize=8,
            markeredgecolor='darkred', zorder=6, label='Arrival (Mars)')

    # Sun at origin
    ax.plot(0.0, 0.0, '*', color='gold', markersize=15, zorder=7,
            markeredgecolor='darkorange', label='Sun')

    # Axes reference lines
    ax.axhline(0.0, color='lightgray', linewidth=0.5)
    ax.axvline(0.0, color='lightgray', linewidth=0.5)

    tof = result_dict.get('tof_days', float('nan'))
    throttle_str = (f'  |  throttle = {throttle_scale:.3f}'
                    if throttle_scale < 1.0 else '')
    ax.set_xlabel('X [AU]', fontsize=12)
    ax.set_ylabel('Y [AU]', fontsize=12)
    ax.set_title(
        f'Earth-Mars {opt_label} Transfer\n'
        f'TOF = {tof:.1f} d  |  '
        f'T = {config.thrust_N:.2f} N  |  '
        f'Isp = {config.isp_s:.0f} s{throttle_str}',
        fontsize=14,
    )
    ax.set_aspect('equal', adjustable='box')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, linestyle=':', linewidth=0.7, alpha=0.7)

    fig.tight_layout()

    if save_path is not None:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_state_history(result_dict: dict, config, save_path: str = None) -> plt.Figure:
    """Plot time histories of the four primary state variables (2x2 grid).

    The four panels show: radial distance r(t) [AU], radial velocity u(t)
    [AU/day], transverse velocity v(t) [AU/day], and polar angle theta(t)
    [degrees], all as functions of mission elapsed time.

    ND velocities are converted to AU/day by dividing by ``t_cf``.

    Parameters
    ----------
    result_dict : dict
        Output from ``solve_time_optimal()`` or ``solve_mass_optimal()``.
        Required keys: ``'state_history'`` (N+1, 6), ``'time_history_days'``
        (N+1,).
    config : MissionConfig
        Mission configuration (for title and segment count annotation).
    save_path : str, optional
        File path for saving.  If *None* the figure is returned unsaved.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    sh       = result_dict['state_history']      # (N+1, 6)  ND
    t_days   = result_dict['time_history_days']  # (N+1,) days

    r_hist    = sh[:, 0]                         # [AU] (ND dist unit = 1 AU)
    u_hist    = sh[:, 1] / t_cf                  # ND vel → AU/day
    v_hist    = sh[:, 2] / t_cf                  # ND vel → AU/day
    theta_deg = np.degrees(sh[:, 3])             # rad → degrees

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    _state_panel(axes[0], t_days, r_hist,    'r(t)  [AU]',     'Radial Distance')
    _state_panel(axes[1], t_days, u_hist,    'u(t)  [AU/day]', 'Radial Velocity')
    _state_panel(axes[2], t_days, v_hist,    'v(t)  [AU/day]', 'Transverse Velocity')
    _state_panel(axes[3], t_days, theta_deg, r'$\theta$(t)  [deg]', 'Polar Angle')

    # Annotate r-panel with planet-orbit reference lines
    axes[0].axhline(r_earth, color='royalblue', linestyle=':', linewidth=1.0,
                    label='Earth orbit')
    axes[0].axhline(r_mars,  color='firebrick',  linestyle=':', linewidth=1.0,
                    label='Mars orbit')
    axes[0].legend(fontsize=9, loc='upper left')

    tof = result_dict.get('tof_days', float('nan'))
    fig.suptitle(
        f'State History - Earth-Mars Transfer  '
        f'(TOF = {tof:.1f} d,  N = {config.n_segments} segments)',
        fontsize=14,
    )
    fig.tight_layout()

    if save_path is not None:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _draw_thrust_arrows(
    ax: plt.Axes,
    state_history: np.ndarray,
    alphas_rad: np.ndarray,
    n_arrows: int,
    throttle_scale: float = 1.0,
) -> None:
    """Overlay thrust-direction arrows on a 2-D trajectory axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which the trajectory has already been drawn.
    state_history : ndarray, shape (N+1, 6)
        ND polar state history [r, u, v, theta, mp, acc_dv].
    alphas_rad : ndarray, shape (N,)
        Piecewise-constant steering angles [rad].  ``alpha = 0`` is purely
        prograde; ``alpha = +pi/2`` is purely outward radial.
        Thrust components: F_r = sin(alpha), F_th = cos(alpha).
    n_arrows : int
        Number of arrows to draw.  Samples are evenly distributed in
        cumulative arc length along the Cartesian trajectory (excluding the
        departure and arrival endpoints to avoid crowding their markers).
    throttle_scale : float, optional
        Fraction of full throttle applied to scale arrow length.  Use 1.0
        for full-throttle (time-optimal) trajectories; set to the optimal
        throttle value for mass-optimal trajectories so arrow lengths are
        directly proportional to thrust magnitude and comparable across
        plots.  Default is 1.0.
    """
    n_nodes = len(state_history)
    N_seg   = len(alphas_rad)

    r_hist  = state_history[:, 0]   # radial distance [AU]
    th_hist = state_history[:, 3]   # polar angle     [rad]

    # --- Arc-length parametrisation ---
    x_traj = r_hist * np.cos(th_hist)
    y_traj = r_hist * np.sin(th_hist)
    ds     = np.hypot(np.diff(x_traj), np.diff(y_traj))
    arc    = np.concatenate([[0.0], np.cumsum(ds)])   # (N+1,) cumulative arc length

    # Exclude endpoints (1 % margin on each side) to avoid crowding markers
    arc_total = arc[-1]
    s_samples = np.linspace(0.01 * arc_total, 0.99 * arc_total, n_arrows)
    # Map each sample arc length to the nearest node index
    indices = np.array([int(np.argmin(np.abs(arc - s))) for s in s_samples],
                       dtype=int)

    # --- Arrow display scale ---
    # Base length = 65 % of the radial span (floor 0.20 AU).  Multiply by
    # throttle_scale so mass-optimal arrows are shorter in proportion to the
    # reduced thrust magnitude, enabling direct visual comparison with
    # full-throttle (time-optimal) plots drawn at the same base scale.
    r_span       = float(r_hist.max() - r_hist.min())
    arrow_len_au = max(0.65 * r_span, 0.20) * throttle_scale

    # --- Build arrow arrays ---
    xs, ys, dxs, dys = [], [], [], []
    for idx in indices:
        r_k  = float(r_hist[idx])
        th_k = float(th_hist[idx])

        # Segment whose constant alpha applies at this node:
        # segment k spans [node k, node k+1].  At node idx use segment
        # min(idx, N_seg-1) so the arrival node maps to the last segment.
        seg     = min(int(idx), N_seg - 1)
        alpha_k = float(alphas_rad[seg])

        # Local polar basis vectors (Cartesian)
        #   r_hat  = ( cos(th),  sin(th))   outward radial
        #   th_hat = (-sin(th),  cos(th))   prograde transverse
        cos_th, sin_th = np.cos(th_k), np.sin(th_k)
        sin_a,  cos_a  = np.sin(alpha_k), np.cos(alpha_k)

        # Thrust unit vector in Cartesian:
        #   T_hat = sin(alpha)*r_hat + cos(alpha)*th_hat
        dir_x = sin_a * cos_th - cos_a * sin_th
        dir_y = sin_a * sin_th + cos_a * cos_th

        xs.append(r_k * cos_th)
        ys.append(r_k * sin_th)
        dxs.append(dir_x * arrow_len_au)
        dys.append(dir_y * arrow_len_au)

    throttle_lbl = (f', throttle={throttle_scale:.2f}' if throttle_scale < 1.0 else '')
    ax.quiver(
        xs, ys, dxs, dys,
        angles='xy', scale_units='xy', scale=1,
        color='darkorange', alpha=0.85,
        width=0.004, headwidth=4, headlength=5, headaxislength=4,
        label=f'Thrust direction ({n_arrows} samples{throttle_lbl})',
        zorder=5,
    )


def _state_panel(ax, t, y, ylabel: str, title: str) -> None:
    ax.plot(t, y, color='black', linewidth=1.4)
    ax.set_xlabel('Mission elapsed time [days]', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.grid(True, linestyle=':', linewidth=0.7, alpha=0.7)
    ax.set_xlim(t[0], t[-1])


def _ensure_dir(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print('=== trajectory_plot.py self-test ===\n')

    from types import SimpleNamespace

    # Synthetic result dict: gradual spiral from r=1 AU to r=1.52 AU
    N = 20
    tof_days = 220.0
    t_hist      = np.linspace(0.0, tof_days, N + 1)
    r_hist      = np.linspace(1.0, r_mars, N + 1)
    theta_hist  = np.linspace(0.0, np.radians(200.0), N + 1)
    v_circ      = 1.0 / np.sqrt(r_hist)                # circular transverse speed (ND)
    mp_hist     = np.linspace(0.0, 0.12, N + 1)

    state_history = np.column_stack([
        r_hist,
        np.zeros(N + 1),          # u (radial vel)
        v_circ,                   # v (transverse vel, ND)
        theta_hist,               # theta [rad]
        mp_hist,                  # mp fraction
        np.zeros(N + 1),          # acc_dv
    ])

    result = {
        'success':            True,
        'tof_days':           tof_days,
        'alphas_rad':         np.zeros(N),
        'propellant_mass_kg': 600.0,
        'final_mass_kg':      4400.0,
        'state_history':      state_history,
        'time_history_days':  t_hist,
        'solver_message':     'Optimization terminated successfully.',
        'n_iterations':       42,
        'wall_time_s':        3.2,
    }

    cfg = SimpleNamespace(
        initial_mass_kg=5000.0,
        thrust_N=3.5,
        isp_s=3000.0,
        n_segments=N,
    )

    print('Test 1: plot_2d_trajectory — returns Figure, no errors')
    fig1 = plot_2d_trajectory(result, cfg)
    assert fig1 is not None
    assert len(fig1.axes) == 1
    plt.close(fig1)
    print('  PASSED\n')

    print('Test 2: plot_state_history — returns Figure with 4 axes')
    fig2 = plot_state_history(result, cfg)
    assert fig2 is not None
    assert len(fig2.axes) == 4
    plt.close(fig2)
    print('  PASSED\n')

    print('Test 3: plot_2d_trajectory with save_path')
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = os.path.join(tmpdir, 'sub', 'traj.png')
        fig3 = plot_2d_trajectory(result, cfg, save_path=outpath)
        assert pathlib.Path(outpath).exists(), 'File was not created'
        plt.close(fig3)
    print('  PASSED\n')

    print('Test 4: velocity unit conversion — v at Earth = 1/t_cf AU/day ~0.01720')
    # ND transverse vel at r=1 is 1.0 (circular); dimensional = 1.0/t_cf AU/day
    v_nd_earth = 1.0
    v_dim = v_nd_earth / t_cf
    assert abs(v_dim - 0.01720) < 5e-5, f'Unexpected Earth speed: {v_dim:.5f}'
    print(f'  v_earth = {v_dim:.5f} AU/day  (expect ~0.01720)  PASSED\n')

    print('All trajectory_plot tests passed.')
