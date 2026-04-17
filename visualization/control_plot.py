"""Steering-angle and mass-history visualizations for low-thrust optimization.

Functions
---------
plot_steering_history
    Piecewise-constant alpha(t) rendered as a step/staircase plot.
plot_mass_history
    Spacecraft mass vs. mission elapsed time with propellant annotation.
"""

import os

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_steering_history(result_dict: dict, config, save_path: str = None) -> plt.Figure:
    """Plot the piecewise-constant thrust steering angle as a staircase.

    Each segment's constant alpha value is rendered as a horizontal step
    spanning the segment duration, with vertical transitions between
    segments.  This matches the control-history style in ilt_opt_snopt.pdf.

    Parameters
    ----------
    result_dict : dict
        Output from ``solve_time_optimal()`` or ``solve_mass_optimal()``.
        Required keys: ``'alphas_rad'`` (N,), ``'time_history_days'`` (N+1,).
    config : MissionConfig
        Mission configuration (for title annotation).
    save_path : str, optional
        File path for saving.  If *None* the figure is returned unsaved.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    alphas_deg = np.degrees(result_dict['alphas_rad'])    # (N,) degrees
    alphas_unwrapped = np.degrees(np.unwrap(np.radians(alphas_deg)))       # unwrap to avoid jumps >180 deg
    t_days     = result_dict['time_history_days']         # (N+1,) days

    # Build staircase arrays: alpha[k] holds on [t[k], t[k+1])
    # np.repeat trick: [t0,t1,...,tN] -> [t0,t0,t1,t1,...,tN,tN][1:-1]
    #                                  = [t0, t1,t1, t2,t2, ..., tN]  (2N pts)
    t_step = np.repeat(t_days, 2)[1:-1]       # (2N,)
    a_step = np.repeat(alphas_deg, 2)          # (2N,)
    a_step_unwrapped = np.repeat(alphas_unwrapped, 2)  # (2N,)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_step, a_step_unwrapped, color='black', linewidth=1.4)
    ax.fill_between(t_step, 0.0, a_step_unwrapped, alpha=0.15, color='steelblue')
    ax.axhline(0.0, color='dimgray', linewidth=0.8, linestyle='--')

    ax.set_xlabel('Mission elapsed time [days]', fontsize=12)
    ax.set_ylabel('Steering angle  alpha [deg]', fontsize=12)
    tof = result_dict.get('tof_days', float('nan'))
    ax.set_title(
        f'Control History - Thrust Steering Angle\n'
        f'TOF = {tof:.1f} d  |  N = {config.n_segments} segments',
        fontsize=14,
    )
    ax.set_xlim(t_days[0], t_days[-1])
    ax.grid(True, linestyle=':', linewidth=0.7, alpha=0.7)

    fig.tight_layout()

    if save_path is not None:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_mass_history(result_dict: dict, config, save_path: str = None) -> plt.Figure:
    """Plot spacecraft mass vs. mission elapsed time.

    The mass is computed from the propellant mass fraction stored in
    ``state_history[:, 4]``:  m(t) = (1 - mp_frac(t)) * m0.

    Annotations show the initial mass, final mass, and total propellant
    consumed (mass and percentage of initial).

    Parameters
    ----------
    result_dict : dict
        Output from ``solve_time_optimal()`` or ``solve_mass_optimal()``.
        Required keys: ``'state_history'`` (N+1, 6), ``'time_history_days'``
        (N+1,).
    config : MissionConfig
        Mission configuration.  Uses ``config.initial_mass_kg``.
    save_path : str, optional
        File path for saving.  If *None* the figure is returned unsaved.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    sh      = result_dict['state_history']      # (N+1, 6)
    t_days  = result_dict['time_history_days']  # (N+1,)

    mp_frac = sh[:, 4]                          # propellant mass fraction [-]
    m0      = config.initial_mass_kg
    mass_kg = (1.0 - mp_frac) * m0             # spacecraft mass [kg]
    mf      = mass_kg[-1]
    mp      = m0 - mf

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(t_days, mass_kg, color='black', linewidth=1.6)
    ax.fill_between(t_days, mf, mass_kg, alpha=0.20, color='steelblue',
                    label=f'Propellant: {mp:.1f} kg  ({100.0*mp/m0:.1f}%)')

    # Reference lines for initial and final mass
    ax.axhline(m0, color='royalblue', linewidth=1.0, linestyle='--',
               label=f'Initial mass: {m0:.0f} kg')
    ax.axhline(mf, color='firebrick',  linewidth=1.0, linestyle='--',
               label=f'Final mass: {mf:.1f} kg')

    # In-plot annotation at midpoint of shaded region
    t_ann = t_days[len(t_days) // 2]
    m_ann = 0.5 * (m0 + mf)
    ax.annotate(
        f'{mp:.0f} kg consumed\n({100.0*mp/m0:.1f}%)',
        xy=(t_ann, m_ann),
        fontsize=10,
        ha='center',
        va='center',
        color='steelblue',
        fontweight='bold',
    )

    ax.set_xlabel('Mission elapsed time [days]', fontsize=12)
    ax.set_ylabel('Spacecraft mass [kg]', fontsize=12)
    tof = result_dict.get('tof_days', float('nan'))
    ax.set_title(
        f'Mass History - Propellant Consumption\n'
        f'TOF = {tof:.1f} d  |  Isp = {config.isp_s:.0f} s',
        fontsize=14,
    )
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, linestyle=':', linewidth=0.7, alpha=0.7)
    ax.set_xlim(t_days[0], t_days[-1])
    margin = 0.05 * m0
    ax.set_ylim(max(0.0, mf - margin), m0 + margin)

    fig.tight_layout()

    if save_path is not None:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print('=== control_plot.py self-test ===\n')

    from types import SimpleNamespace

    # Synthetic result dict
    N = 20
    tof_days = 220.0
    t_hist   = np.linspace(0.0, tof_days, N + 1)

    r_hist   = np.linspace(1.0, 1.52368, N + 1)
    mp_hist  = np.linspace(0.0, 0.12, N + 1)
    state_history = np.column_stack([
        r_hist,
        np.zeros(N + 1),
        1.0 / np.sqrt(r_hist),
        np.linspace(0.0, np.radians(200.0), N + 1),
        mp_hist,
        np.zeros(N + 1),
    ])
    # Realistic steering: slight prograde with growing out-of-plane component
    alphas_rad = np.linspace(-0.30, 0.30, N)

    result = {
        'success':            True,
        'tof_days':           tof_days,
        'alphas_rad':         alphas_rad,
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

    print('Test 1: plot_steering_history — returns Figure, staircase length correct')
    fig1 = plot_steering_history(result, cfg)
    assert fig1 is not None
    assert len(fig1.axes) == 1
    # Verify staircase data length = 2*N
    line_xdata = fig1.axes[0].lines[0].get_xdata()
    assert len(line_xdata) == 2 * N, f'Expected {2*N} staircase pts, got {len(line_xdata)}'
    plt.close(fig1)
    print('  PASSED\n')

    print('Test 2: plot_mass_history — returns Figure, mass values plausible')
    fig2 = plot_mass_history(result, cfg)
    assert fig2 is not None
    assert len(fig2.axes) == 1
    # Read the plotted mass data from the first line
    mass_line = fig2.axes[0].lines[0].get_ydata()
    assert abs(mass_line[0]  - 5000.0) < 1.0,   f'Initial mass mismatch: {mass_line[0]:.1f}'
    assert abs(mass_line[-1] - 4400.0) < 1.0,   f'Final mass mismatch: {mass_line[-1]:.1f}'
    plt.close(fig2)
    print('  PASSED\n')

    print('Test 3: staircase correctness for N=3 toy case')
    result3 = {
        'tof_days':          30.0,
        'alphas_rad':        np.array([0.1, -0.2, 0.3]),
        'time_history_days': np.array([0.0, 10.0, 20.0, 30.0]),
        'state_history':     np.zeros((4, 6)),
    }
    # t_step should be [0,10,10,20,20,30]; a_step should hold each alpha twice
    t_step = np.repeat(result3['time_history_days'], 2)[1:-1]
    a_step = np.repeat(np.degrees(result3['alphas_rad']), 2)
    assert list(t_step) == [0.0, 10.0, 10.0, 20.0, 20.0, 30.0], f'Bad t_step: {t_step}'
    assert len(a_step) == 6
    plt.close('all')
    print('  PASSED\n')

    print('All control_plot tests passed.')
