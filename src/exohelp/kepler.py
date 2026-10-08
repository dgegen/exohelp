import astropy.constants as const
import numpy as np
from astropy import units as u

from .type import QuantityLike

__all__ = ["keplers_third_law", "solve_kepler"]


def solve_kepler(
    mean_anomaly: QuantityLike,
    eccentricity: QuantityLike = 0.0,
    tol: float = 1e-10,
    max_iter: int = 20,
) -> np.ndarray | u.Quantity | float:
    """Solve Kepler's equation M = E - e * sin(E) for the eccentric anomaly E.

    Uses Halley's method (third-order Householder method) for fast and robust
    convergence across all eccentricities 0 <= e < 1.

    Parameters
    ----------
    mean_anomaly : QuantityLike
        Mean anomaly M. If given without units, assumed to be in radians.
    eccentricity : QuantityLike, optional
        Orbital eccentricity (0 <= e < 1). Default is 0.0.
    tol : float, optional
        Absolute tolerance on the change in E. Default is 1e-10.
    max_iter : int, optional
        Maximum number of iterations. Default is 20.

    Returns
    -------
    E : np.ndarray, Quantity, or float
        Eccentric anomaly in radians (matching input Quantity type if unit was provided).

    Examples
    --------
    >>> from exohelp.kepler import solve_kepler
    >>> round(solve_kepler(0.5, 0.1), 5)
    0.55248
    """
    is_quantity = isinstance(mean_anomaly, u.Quantity)
    m_val = mean_anomaly.to_value(u.rad) if is_quantity else np.asarray(mean_anomaly, dtype=float)

    if isinstance(eccentricity, u.Quantity):
        e_val = eccentricity.to_value(u.dimensionless_unscaled)
    else:
        e_val = np.asarray(eccentricity, dtype=float)

    # Initial guess
    e_val = np.asarray(e_val)
    m_val = np.asarray(m_val)
    ecc_anom = m_val + e_val * np.sin(m_val) / (1.0 - np.sin(m_val + e_val) + np.sin(m_val) + 1e-12)

    for _ in range(max_iter):
        f = ecc_anom - e_val * np.sin(ecc_anom) - m_val
        f_prime = 1.0 - e_val * np.cos(ecc_anom)
        f_double_prime = e_val * np.sin(ecc_anom)
        delta1 = f / f_prime
        delta = f / (f_prime - 0.5 * delta1 * f_double_prime)
        ecc_anom = ecc_anom - delta
        if np.all(np.abs(delta) < tol):
            break

    # Format return matching scalar vs array vs Quantity
    if m_val.ndim == 0 and e_val.ndim == 0:
        res = float(ecc_anom)
        return u.Quantity(res, "rad") if is_quantity else res
    return u.Quantity(ecc_anom, "rad") if is_quantity else ecc_anom


def keplers_third_law(
    period: QuantityLike | None = None,
    semi_major_axis: QuantityLike | None = None,
    mass: QuantityLike | None = None,
) -> u.Quantity:
    """
    Apply Kepler's third law to solve for the missing orbital parameter.

    Exactly one of ``period``, ``semi_major_axis``, or ``mass`` must be ``None``;
    that quantity is solved for and returned.

    Parameters
    ----------
    period : float or Quantity, optional
        Orbital period. Assumed to be in days if no unit is given.
    semi_major_axis : float or Quantity, optional
        Semi-major axis of the orbit. Assumed to be in AU if no unit is given.
    mass : float or Quantity, optional
        Stellar mass. Assumed to be in M_sun if no unit is given.
        Defaults to 1 M_sun when omitted along with the solved-for quantity.

    Returns
    -------
    Quantity
        The solved-for quantity in days (period), AU (semi-major axis),
        or M_sun (mass).

    Raises
    ------
    ValueError
        If not exactly one parameter is ``None``.

    Examples
    --------
    >>> keplers_third_law(period=365.25, semi_major_axis=1).round(1)
    <Quantity 1. solMass>
    >>> keplers_third_law(semi_major_axis=1).round(2)
    <Quantity 365.26 d>
    >>> keplers_third_law(period=365.25).round(1)
    <Quantity 1. AU>
    """
    if period is None and semi_major_axis is not None:
        return _keplers_third_law_period(semi_major_axis=semi_major_axis, mass=mass)
    elif semi_major_axis is None and period is not None:
        return _keplers_third_law_semi_major_axis(period=period, mass=mass)
    elif mass is None and period is not None and semi_major_axis is not None:
        return _keplers_third_law_mass(period=period, semi_major_axis=semi_major_axis)
    else:
        raise ValueError("Exactly one of period, semi_major_axis, or mass must be None.")


def _keplers_third_law_mass(period: QuantityLike, semi_major_axis: QuantityLike) -> u.Quantity:
    period = u.Quantity(period, "day")
    semi_major_axis = u.Quantity(semi_major_axis, "AU")
    return ((4 * np.pi**2 * semi_major_axis**3) / (const.G * period**2)).to("M_sun")  # type: ignore[attr-defined]


def _keplers_third_law_period(
    semi_major_axis: QuantityLike, mass: QuantityLike | None = None
) -> u.Quantity:
    if mass is None:
        mass = u.Quantity(1.0, "M_sun")
    mass = u.Quantity(mass, "M_sun")
    semi_major_axis = u.Quantity(semi_major_axis, "AU")
    return np.sqrt((4 * np.pi**2 * semi_major_axis**3) / (const.G * mass)).to("day")  # type: ignore[attr-defined]


def _keplers_third_law_semi_major_axis(
    period: QuantityLike, mass: QuantityLike | None = None
) -> u.Quantity:
    if mass is None:
        mass = u.Quantity(1.0, "M_sun")
    mass = u.Quantity(mass, "M_sun")
    period = u.Quantity(period, "day")
    return (((const.G * mass * period**2) / (4 * np.pi**2)) ** (1 / 3)).to("AU")  # type: ignore[attr-defined]
