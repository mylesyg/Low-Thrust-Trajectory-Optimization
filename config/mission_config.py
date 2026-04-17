"""Mission configuration dataclass for Earth-Mars low-thrust trajectory optimization.

Analogous to the .dat input files in the MATLAB reference (ilt_opt_snopt.m):
    e2m_min_time.dat  ->  MissionConfig.default_time_optimal()
    e2m_max_mass.dat  ->  MissionConfig.default_mass_optimal()

Optimization modes
------------------
iopt = 1 | 'time_optimal'
    Minimize transfer time.  Throttle is fixed at 1.0 (full thrust throughout).
    Free variables: time of flight (bounded) + N steering angles alpha_k.

iopt = 2 | 'mass_optimal'
    Maximize final mass (minimize propellant consumption) for a fixed transfer
    time.  Free variables: N steering angles alpha_k + N throttle values tau_k.

Units
-----
All fields stored in dimensional (SI or day) units.  The ``to_nd_params()``
method converts propulsion parameters and time bounds to the non-dimensional
scheme defined in core.nondimensional.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from core.nondimensional import NonDimensional

_VALID_MODES = {'time_optimal', 'mass_optimal'}


@dataclass
class MissionConfig:
    """All mission and optimizer parameters for an Earth-Mars EP transfer.

    Parameters
    ----------
    Spacecraft
    ----------
    initial_mass_kg : float
        Wet (initial) spacecraft mass [kg].  Used as the reference mass m0
        for non-dimensionalization.  Default: 5000 kg.
    thrust_N : float
        Maximum (full-throttle) SEP thrust magnitude [N].  Default: 3.5 N.
    isp_s : float
        Engine specific impulse [s].  Default: 3000 s.

    Optimization
    ------------
    opt_mode : str
        Optimization objective.  Must be 'time_optimal' or 'mass_optimal'.
    n_segments : int
        Number of equal-duration shooting segments N.  The control vector
        has length N (alphas) for time-optimal or 2*N (alphas + throttles)
        for mass-optimal.

    Time parameters (all in days)
    ------------------------------
    time_guess_days : float
        Initial guess for the transfer time.  For mass-optimal this is
        the *fixed* time of flight.
    time_lb_days : float
        Lower bound on transfer time (used only in time-optimal mode).
    time_ub_days : float
        Upper bound on transfer time (used only in time-optimal mode).

    Throttle parameters
    -------------------
    throttle_guess : float
        Initial throttle for each segment (scalar; applied uniformly).
        For time-optimal: fixed at 1.0.  For mass-optimal: starting guess.
    throttle_lb : float
        Per-segment throttle lower bound.  Must be in [0, 1].
    throttle_ub : float
        Per-segment throttle upper bound.  Must be in [0, 1].

    Steering angle parameters (stored in degrees, converted on demand)
    ------------------------------------------------------------------
    alpha_guess_deg : float
        Initial steering angle guess for all segments [deg].
    alpha_lb_deg : float
        Lower bound on steering angle [deg].  Typically -180.
    alpha_ub_deg : float
        Upper bound on steering angle [deg].  Typically +180.

    Departure date
    --------------
    departure_date_guess : str
        ISO-format date string ('YYYY-MM-DD') used as a starting point for
        launch window estimation via ephemeris.planetary_states.
    """

    # ------------------------------------------------------------------
    # Spacecraft parameters
    # ------------------------------------------------------------------
    initial_mass_kg: float = 5000.0
    thrust_N:        float = 3.5
    isp_s:           float = 3000.0

    # ------------------------------------------------------------------
    # Optimization settings
    # ------------------------------------------------------------------
    opt_mode:   str = 'time_optimal'
    n_segments: int = 200

    # ------------------------------------------------------------------
    # Time parameters [days]
    # ------------------------------------------------------------------
    time_guess_days: float = 200.0
    time_lb_days:    float = 185.0
    time_ub_days:    float = 275.0

    # ------------------------------------------------------------------
    # Throttle parameters [dimensionless]
    # ------------------------------------------------------------------
    throttle_guess: float = 1.0
    throttle_lb:    float = 0.0
    throttle_ub:    float = 1.0

    # ------------------------------------------------------------------
    # Steering angle parameters [degrees]
    # ------------------------------------------------------------------
    alpha_guess_deg: float = 20.0
    alpha_lb_deg:    float = -180.0
    alpha_ub_deg:    float =  180.0

    # ------------------------------------------------------------------
    # Departure date
    # ------------------------------------------------------------------
    departure_date_guess: str = '2026-01-01'

    # ------------------------------------------------------------------
    # Post-init validation
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.opt_mode not in _VALID_MODES:
            raise ValueError(
                f"opt_mode must be one of {_VALID_MODES!r}, got {self.opt_mode!r}"
            )
        if self.n_segments < 1:
            raise ValueError(f"n_segments must be >= 1, got {self.n_segments}")
        if not (0.0 <= self.throttle_lb <= self.throttle_ub <= 1.0):
            raise ValueError(
                f"Throttle bounds must satisfy 0 <= lb <= ub <= 1, "
                f"got lb={self.throttle_lb}, ub={self.throttle_ub}"
            )
        if self.time_lb_days > self.time_guess_days or self.time_guess_days > self.time_ub_days:
            raise ValueError(
                f"Time bounds must satisfy lb <= guess <= ub, "
                f"got lb={self.time_lb_days}, guess={self.time_guess_days}, "
                f"ub={self.time_ub_days}"
            )
        if self.initial_mass_kg <= 0 or self.thrust_N <= 0 or self.isp_s <= 0:
            raise ValueError(
                "initial_mass_kg, thrust_N, and isp_s must all be positive."
            )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def to_nd_params(self) -> Dict[str, float]:
        """Return non-dimensional propulsion and time parameters.

        Converts dimensional spacecraft and time parameters to the
        non-dimensional units used throughout the propagator and optimizers
        (see core.nondimensional for the normalization scheme).

        Returns
        -------
        dict with keys:
            'acc_thrust' : float
                Non-dimensional thrust acceleration = (T/m0) / (mu_sun/r0^2) [-].
            'mp_dot' : float
                Non-dimensional mass-flow rate
                = (T/(g0*Isp) * 86400 / m0) * t_cf [-].
            'tof_nd' : float
                Non-dimensional time of flight (from time_guess_days) [-].
            'tof_lb_nd' : float
                Non-dimensional lower bound on time of flight [-].
            'tof_ub_nd' : float
                Non-dimensional upper bound on time of flight [-].

        Notes
        -----
        For mass-optimal mode, tof_nd is the *fixed* transfer time.
        The optimizer should treat it as an equality constraint rather than
        a free variable.

        Example
        -------
        >>> cfg = MissionConfig.default_time_optimal()
        >>> nd = cfg.to_nd_params()
        >>> nd['acc_thrust']
        0.11804...   # (T/m0) / (mu_sun/AU^2)
        """
        acc_thrust = NonDimensional.compute_thrust_acc(self.thrust_N, self.initial_mass_kg)
        mp_dot     = NonDimensional.compute_mass_flow(self.thrust_N, self.isp_s, self.initial_mass_kg)
        tof_nd     = NonDimensional.time_to_nd(self.time_guess_days)
        tof_lb_nd  = NonDimensional.time_to_nd(self.time_lb_days)
        tof_ub_nd  = NonDimensional.time_to_nd(self.time_ub_days)
        return {
            'acc_thrust': acc_thrust,
            'mp_dot':     mp_dot,
            'tof_nd':     tof_nd,
            'tof_lb_nd':  tof_lb_nd,
            'tof_ub_nd':  tof_ub_nd,
        }

    def get_alpha_bounds_rad(self) -> Tuple[float, float]:
        """Return the steering angle bounds in radians.

        Returns
        -------
        (lb_rad, ub_rad) : tuple of float
            Lower and upper bounds on the per-segment steering angle [rad].
            Derived from alpha_lb_deg and alpha_ub_deg.
        """
        return math.radians(self.alpha_lb_deg), math.radians(self.alpha_ub_deg)

    def alpha_guess_rad(self) -> float:
        """Return the steering angle initial guess in radians."""
        return math.radians(self.alpha_guess_deg)

    def summary(self) -> str:
        """Return a human-readable summary of the mission configuration."""
        nd = self.to_nd_params()
        lb_rad, ub_rad = self.get_alpha_bounds_rad()
        lines = [
            "MissionConfig summary",
            "=" * 44,
            f"  Optimization mode    : {self.opt_mode}",
            f"  N segments           : {self.n_segments}",
            "",
            "  Spacecraft",
            f"    Initial mass       : {self.initial_mass_kg:.1f} kg",
            f"    Thrust             : {self.thrust_N:.3f} N",
            f"    Isp                : {self.isp_s:.1f} s",
            f"    acc_thrust (ND)    : {nd['acc_thrust']:.6f}",
            f"    mp_dot     (ND)    : {nd['mp_dot']:.6f}",
            "",
            "  Time of flight",
            f"    Guess              : {self.time_guess_days:.1f} days"
            f"  ({nd['tof_nd']:.4f} ND)",
            f"    Lower bound        : {self.time_lb_days:.1f} days"
            f"  ({nd['tof_lb_nd']:.4f} ND)",
            f"    Upper bound        : {self.time_ub_days:.1f} days"
            f"  ({nd['tof_ub_nd']:.4f} ND)",
            "",
            "  Throttle",
            f"    Guess              : {self.throttle_guess:.2f}",
            f"    Bounds             : [{self.throttle_lb:.2f}, {self.throttle_ub:.2f}]",
            "",
            "  Steering angle",
            f"    Guess              : {self.alpha_guess_deg:.1f} deg"
            f"  ({self.alpha_guess_rad():.4f} rad)",
            f"    Bounds             : [{self.alpha_lb_deg:.1f}, {self.alpha_ub_deg:.1f}] deg"
            f"  ([{lb_rad:.4f}, {ub_rad:.4f}] rad)",
            "",
            f"  Departure date guess : {self.departure_date_guess}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Factory classmethods
    # ------------------------------------------------------------------

    @classmethod
    def default_time_optimal(cls) -> MissionConfig:
        """Return a MissionConfig for the time-optimal problem.

        Matches the MATLAB reference file e2m_min_time.dat:
            iopt = 1  (minimize transfer time)
            Throttle fixed at 1.0 (full thrust throughout)
            Free variables: time of flight + N steering angles

        Spacecraft: 5000 kg initial mass, 3.5 N SEP thrust, 3000 s Isp.
        Transfer time guess 200 days, bounds [175, 225] days.
        Steering angle guess 20 deg, bounds [-180, 180] deg.
        200 shooting segments.

        Returns
        -------
        MissionConfig
        """
        return cls(
            initial_mass_kg       = 5000.0,
            thrust_N              = 3.5,
            isp_s                 = 3000.0,
            opt_mode              = 'time_optimal',
            n_segments            = 200,
            time_guess_days       = 200.0,
            time_lb_days          = 185.0,
            time_ub_days          = 275.0,
            throttle_guess        = 1.0,
            throttle_lb           = 1.0,   # fixed at full throttle
            throttle_ub           = 1.0,
            alpha_guess_deg       = 20.0,
            alpha_lb_deg          = -180.0,
            alpha_ub_deg          =  180.0,
            departure_date_guess  = '2026-01-01',
        )

    @classmethod
    def default_mass_optimal(cls) -> MissionConfig:
        """Return a MissionConfig for the mass-optimal (maximum final mass) problem.

        Matches the MATLAB reference file e2m_max_mass.dat:
            iopt = 2  (maximize final mass)
            Transfer time is fixed at 200 days (the time-optimal result)
            Free variables: N steering angles + N throttle values in [0, 1]

        Spacecraft parameters identical to default_time_optimal().
        Time bounds collapsed to a fixed value (lb = guess = ub = 200 days).
        Throttle initial guess 0.5 (intermediate), bounds [0, 1].
        200 shooting segments.

        Returns
        -------
        MissionConfig
        """
        fixed_tof = 200.0   # days — fix at the time-optimal result
        return cls(
            initial_mass_kg       = 5000.0,
            thrust_N              = 3.5,
            isp_s                 = 3000.0,
            opt_mode              = 'mass_optimal',
            n_segments            = 200,
            time_guess_days       = fixed_tof,
            time_lb_days          = fixed_tof,   # fixed: lb == guess == ub
            time_ub_days          = fixed_tof,
            throttle_guess        = 0.5,
            throttle_lb           = 0.0,
            throttle_ub           = 1.0,
            alpha_guess_deg       = 20.0,
            alpha_lb_deg          = -180.0,
            alpha_ub_deg          =  180.0,
            departure_date_guess  = '2026-01-01',
        )


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("=== mission_config.py self-test ===\n")

    # ------------------------------------------------------------------
    # 1. Default time-optimal instance
    # ------------------------------------------------------------------
    print("Test 1: default_time_optimal()")
    cfg_t = MissionConfig.default_time_optimal()
    nd_t  = cfg_t.to_nd_params()

    print(cfg_t.summary())
    print()
    assert cfg_t.opt_mode   == 'time_optimal'
    assert cfg_t.n_segments == 200
    assert cfg_t.throttle_lb == cfg_t.throttle_ub == 1.0, "Time-optimal must fix throttle at 1"
    assert nd_t['tof_nd']    >  0.0
    assert nd_t['tof_lb_nd'] <= nd_t['tof_nd'] <= nd_t['tof_ub_nd']
    assert nd_t['acc_thrust'] > 0.0
    assert nd_t['mp_dot']     > 0.0
    print("  PASSED\n")

    # ------------------------------------------------------------------
    # 2. Default mass-optimal instance
    # ------------------------------------------------------------------
    print("Test 2: default_mass_optimal()")
    cfg_m = MissionConfig.default_mass_optimal()
    nd_m  = cfg_m.to_nd_params()

    assert cfg_m.opt_mode        == 'mass_optimal'
    assert cfg_m.throttle_lb     == 0.0
    assert cfg_m.throttle_ub     == 1.0
    # Fixed time: lb == guess == ub
    assert cfg_m.time_lb_days == cfg_m.time_guess_days == cfg_m.time_ub_days
    assert nd_m['tof_lb_nd']  == nd_m['tof_nd'] == nd_m['tof_ub_nd']
    # Same spacecraft -> same propulsion ND params
    assert abs(nd_t['acc_thrust'] - nd_m['acc_thrust']) < 1e-15
    assert abs(nd_t['mp_dot']     - nd_m['mp_dot'])     < 1e-15
    print(f"  acc_thrust           : {nd_m['acc_thrust']:.6f}  (ND)")
    print(f"  mp_dot               : {nd_m['mp_dot']:.6f}  (ND)")
    print(f"  tof_nd (fixed)       : {nd_m['tof_nd']:.4f}  (ND)")
    print("  PASSED\n")

    # ------------------------------------------------------------------
    # 3. Steering angle bounds conversion
    # ------------------------------------------------------------------
    print("Test 3: get_alpha_bounds_rad()")
    lb_rad, ub_rad = cfg_t.get_alpha_bounds_rad()
    assert abs(lb_rad - (-math.pi)) < 1e-12
    assert abs(ub_rad -  math.pi)   < 1e-12
    print(f"  Alpha bounds: [{lb_rad:.6f}, {ub_rad:.6f}] rad  (expect [-pi, +pi])")
    print("  PASSED\n")

    # ------------------------------------------------------------------
    # 4. Validation: bad opt_mode
    # ------------------------------------------------------------------
    print("Test 4: validation catches bad opt_mode")
    try:
        _ = MissionConfig(opt_mode='invalid')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Caught expected error: {e}")
    print("  PASSED\n")

    # ------------------------------------------------------------------
    # 5. ND propulsion sanity check against MATLAB reference values
    # ------------------------------------------------------------------
    print("Test 5: ND values vs MATLAB reference")
    # From ilt_opt_snopt.m: acc_thrust = (0.001*3.5/5000)/(xmu/aunit^2)
    # Expected order of magnitude: ~0.12 (3.5 N / 5000 kg spacecraft)
    print(f"  acc_thrust = {nd_t['acc_thrust']:.6f}")
    print(f"  mp_dot     = {nd_t['mp_dot']:.6f}")
    # Both should be in [0.05, 0.30] for these spacecraft params
    assert 0.05 < nd_t['acc_thrust'] < 0.30, f"acc_thrust out of expected range"
    assert 0.05 < nd_t['mp_dot']     < 0.30, f"mp_dot out of expected range"
    # acc_thrust and mp_dot should be close (same order of magnitude)
    ratio = nd_t['acc_thrust'] / nd_t['mp_dot']
    print(f"  acc_thrust / mp_dot = {ratio:.4f}  (expect ~1, both ~0.12)")
    print("  PASSED")
