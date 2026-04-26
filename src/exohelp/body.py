"""Functions for calculating properties of astronomical bodies, such as density and surface gravity."""

import astropy.constants as const
import astropy.units as u
import numpy as np

from .type import QuantityLike

__all__ = [
    "bulk_density",
    "log_surface_gravity",
    "surface_gravity",
]


def bulk_density(mass: QuantityLike, radius: QuantityLike) -> u.Quantity:
    """Calculate the density of an object given its mass and radius.

    Parameters
    ----------
    mass : float or array-like
        Mass. Assumed to be in Earth masses if no unit is given.
    radius : float or array-like
        Radius. Assumed to be in Earth radii if no unit is given.

    Returns
    -------
    density_g_cm3 : float or array-like
        Density of the body in grams per cubic centimeter (g/cm³).

    Examples
    --------
    >>> from exohelp import bulk_density
    >>> round(float(bulk_density(1.0, 1.0).value), 1)  # Earth
    5.5
    """
    mass = u.Quantity(mass, "M_earth")
    radius = u.Quantity(radius, "R_earth")

    volume = (4 / 3) * np.pi * radius**3
    density = mass / volume

    return density.to(u.g / u.cm**3)


def surface_gravity(mass: QuantityLike, radius: QuantityLike) -> u.Quantity:
    """Compute the surface gravity of a body.

    Parameters
    ----------
    mass : QuantityLike
        Mass. Assumed to be in Earth masses if no unit is given.
    radius : QuantityLike
        Radius. Assumed to be in Earth radii if no unit is given.

    Returns
    -------
    g : Quantity
        Surface gravity in m/s².

    Examples
    --------
    >>> from exohelp import surface_gravity
    >>> round(float(surface_gravity(1.0, 1.0).value), 1)  # Earth
    9.8
    """
    mass = u.Quantity(mass, "M_earth")
    radius = u.Quantity(radius, "R_earth")
    return (const.G * mass / radius**2).to(u.m / u.s**2)  # type: ignore[attr-defined]


def log_surface_gravity(mass: QuantityLike, radius: QuantityLike) -> np.ndarray:
    """Compute the log₁₀ surface gravity of a body in cgs units.

    Parameters
    ----------
    mass : QuantityLike
        Mass. Assumed to be in Earth masses if no unit is given.
    radius : QuantityLike
        Radius. Assumed to be in Earth radii if no unit is given.

    Returns
    -------
    logg : ndarray
        log10(g / [cm s^2]) (dex).

    Examples
    --------
    >>> from exohelp import log_surface_gravity
    >>> round(float(log_surface_gravity(1.0, 1.0)), 2)  # Earth: log10(~980 cm/s²)
    2.99
    """
    g = surface_gravity(mass, radius).to(u.cm / u.s**2)
    return np.log10(g.value)
