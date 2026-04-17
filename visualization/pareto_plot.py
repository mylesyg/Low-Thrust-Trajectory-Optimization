"""Pareto frontier and power-sweep visualizations for Earth-Mars trade studies.

Functions
---------
plot_pareto_frontier
    Transfer time vs. propellant fraction for the mass-optimal Pareto curve.
plot_power_sweep
    Specific power vs. minimum transfer time with a Hohmann reference line.
"""

import os

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_pareto_frontier(pareto_results: list, save_path: str = None,
                         selected_idx: int = None) -> plt.Figure:
    """Scatter + line plot of transfer time vs. propellant mass fraction.

    Each point on the curve corresponds to a fixed-TOF mass-optimal solve.
    The curve traces the Pareto frontier between mission duration and
    propellant cost.

    Parameters
    ----------
    pareto_results : list of dict
        Output from ``analysis.pareto.generate_pareto_frontier()``.
        Each entry must contain ``'tof_days'`` and ``'propellant_fraction'``.
        The list must be non-empty.
    save_path : str, optional
        File path for saving.  If *None* the figure is returned unsaved.
    selected_idx : int, optional
        Index into *pareto_results* of the point selected for trajectory
        visualization.  When provided, that point is highlighted with a
        distinct marker and annotated with its TOF and throttle.
        Negative indices count from the end (same convention as Python
        lists).  Default is *None* (no highlight).

    Returns
    -------
    fig : matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``pareto_results`` is empty.
    """
    if not pareto_results:
        raise ValueError("pareto_results is empty — nothing to plot.")

    tof_days  = np.array([p['tof_days']           for p in pareto_results])
    prop_pct  = np.array([p['propellant_fraction'] for p in pareto_results]) * 100.0

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(tof_days, prop_pct, '-o', color='steelblue', linewidth=1.6,
            markersize=6, markerfacecolor='white', markeredgewidth=1.5,
            label='Mass-optimal Pareto points')

    if selected_idx is not None:
        sidx = int(selected_idx) % len(pareto_results)
        pt   = pareto_results[sidx]
        thr  = pt.get('throttle', float('nan'))
        ax.plot(
            tof_days[sidx], prop_pct[sidx],
            'D', color='darkorange', markersize=10,
            markeredgecolor='saddlebrown', zorder=5,
            label=(f'Selected (idx {sidx}: TOF={tof_days[sidx]:.0f} d, '
                   f'throttle={thr:.3f})'),
        )

    ax.set_xlabel('Transfer time [days]', fontsize=12)
    ax.set_ylabel('Propellant mass fraction [%]', fontsize=12)
    ax.set_title(
        'Pareto Frontier: Propellant vs. Transfer Time\n'
        '(Earth-Mars Low-Thrust)',
        fontsize=14,
    )
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=':', linewidth=0.7, alpha=0.7)

    fig.tight_layout()

    if save_path is not None:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_power_sweep(
    sweep_results: list,
    hohmann_baseline: dict,
    save_path: str = None,
) -> plt.Figure:
    """Plot specific power vs. minimum transfer time with a Hohmann reference.

    Each point corresponds to a time-optimal solve at a given SEP thrust
    level.  The horizontal dashed line marks the Hohmann transfer time
    (~259 days) — the break-even point where electric propulsion begins to
    offer a shorter transit than the impulsive Hohmann option.

    Parameters
    ----------
    sweep_results : list of dict
        Output from ``analysis.pareto.generate_power_sweep()``.
        Each entry must contain ``'specific_power_W_kg'`` and ``'tof_days'``.
        The list must be non-empty.
    hohmann_baseline : dict
        Output from ``baseline.impulsive_transfer.compute_baseline()``.
        Must contain ``'tof_days'``.
    save_path : str, optional
        File path for saving.  If *None* the figure is returned unsaved.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``sweep_results`` is empty.
    """
    if not sweep_results:
        raise ValueError("sweep_results is empty — nothing to plot.")

    sp_W_kg  = np.array([s['specific_power_W_kg'] for s in sweep_results])
    tof_days = np.array([s['tof_days']             for s in sweep_results])

    hohmann_tof = float(hohmann_baseline['tof_days'])

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(sp_W_kg, tof_days, '-o', color='firebrick', linewidth=1.6,
            markersize=6, markerfacecolor='white', markeredgewidth=1.5,
            label='Low-thrust (time-optimal)')

    ax.axhline(hohmann_tof, color='steelblue', linewidth=1.4, linestyle='--',
               label=f'Hohmann transfer ({hohmann_tof:.1f} d)')

    # Shade the region where EP beats the Hohmann transfer
    beats_idx = np.where(tof_days < hohmann_tof)[0]
    if len(beats_idx) > 0:
        crossover_sp = sp_W_kg[beats_idx[0]]
        y_lo = tof_days.min() * 0.92
        ax.fill_betweenx(
            [y_lo, hohmann_tof],
            crossover_sp, sp_W_kg.max() * 1.02,
            alpha=0.10, color='green',
            label='EP faster than Hohmann',
        )
        ax.axvline(crossover_sp, color='gray', linewidth=0.9, linestyle=':',
                   label=f'Break-even: {crossover_sp:.1f} W/kg')

    ax.set_xlabel('Specific power [W/kg]', fontsize=12)
    ax.set_ylabel('Minimum transfer time [days]', fontsize=12)
    ax.set_title(
        'Power Sweep: Specific Power vs. Transfer Time\n'
        '(Earth-Mars Low-Thrust)',
        fontsize=14,
    )
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=':', linewidth=0.7, alpha=0.7)

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

    print('=== pareto_plot.py self-test ===\n')

    # Synthetic Pareto data: longer TOF -> less propellant (monotone)
    tof_values = np.linspace(150.0, 320.0, 12)
    pareto_results = [
        {'tof_days': tof,
         'propellant_fraction': 0.35 - 0.10 * (tof - 150.0) / (320.0 - 150.0),
         'throttle': 1.0}
        for tof in tof_values
    ]

    # Synthetic power sweep: more power -> shorter TOF
    sp_values = np.linspace(15.0, 200.0, 12)
    sweep_results = [
        {'specific_power_W_kg': sp,
         'tof_days': 380.0 - 0.65 * sp,
         'thrust_N': 1.0}
        for sp in sp_values
    ]

    # Synthetic Hohmann baseline
    hohmann_baseline = {'tof_days': 258.87, 'total_delta_v_ms': 5593.6}

    # ------------------------------------------------------------------
    print('Test 1: plot_pareto_frontier — returns Figure, correct axis labels')
    fig1 = plot_pareto_frontier(pareto_results)
    assert fig1 is not None
    assert len(fig1.axes) == 1
    ax1 = fig1.axes[0]
    assert 'Transfer time' in ax1.get_xlabel()
    assert 'Propellant' in ax1.get_ylabel()
    plt.close(fig1)
    print('  PASSED\n')

    # ------------------------------------------------------------------
    print('Test 2: plot_power_sweep — returns Figure, Hohmann line present')
    fig2 = plot_power_sweep(sweep_results, hohmann_baseline)
    assert fig2 is not None
    assert len(fig2.axes) == 1
    ax2 = fig2.axes[0]
    # Hohmann reference line is an axhline — check at least 2 lines exist
    # (trajectory + at least one reference line)
    assert len(ax2.lines) >= 2, 'Expected trajectory line + Hohmann reference line'
    plt.close(fig2)
    print('  PASSED\n')

    # ------------------------------------------------------------------
    print('Test 3: empty pareto_results raises ValueError')
    try:
        plot_pareto_frontier([])
        assert False, 'Should have raised ValueError'
    except ValueError as e:
        print(f'  Caught expected error: {e}')
    print('  PASSED\n')

    # ------------------------------------------------------------------
    print('Test 4: empty sweep_results raises ValueError')
    try:
        plot_power_sweep([], hohmann_baseline)
        assert False, 'Should have raised ValueError'
    except ValueError as e:
        print(f'  Caught expected error: {e}')
    print('  PASSED\n')

    # ------------------------------------------------------------------
    print('Test 5: save_path creates file in temp directory')
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as tmpdir:
        out1 = os.path.join(tmpdir, 'nested', 'pareto.png')
        fig3 = plot_pareto_frontier(pareto_results, save_path=out1)
        assert pathlib.Path(out1).exists(), 'Pareto figure not saved'
        plt.close(fig3)

        out2 = os.path.join(tmpdir, 'nested', 'power.png')
        fig4 = plot_power_sweep(sweep_results, hohmann_baseline, save_path=out2)
        assert pathlib.Path(out2).exists(), 'Power sweep figure not saved'
        plt.close(fig4)
    print('  PASSED\n')

    print('All pareto_plot tests passed.')
