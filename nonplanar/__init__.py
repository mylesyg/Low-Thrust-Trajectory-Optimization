"""Nonplanar (3D) low-thrust trajectory optimization — Week 4 extension.

Extends the coplanar 2D polar formulation in ``core/`` and ``optimization/``
to full 3D Cartesian heliocentric inertial coordinates.

Modules
-------
eom_3d
    3D equations of motion, Cartesian-to-orbital-elements conversion,
    and segment/trajectory propagators.
nonplanar_optimizer
    SLSQP-based time-optimal NLP for nonplanar transfers specified by
    classical orbital elements.
"""
