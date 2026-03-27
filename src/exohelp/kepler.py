import astropy.constants as const
import numpy as np
from astropy import units as u

from .type import QuantityLike

__all__ = ["keplers_third_law"]


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
