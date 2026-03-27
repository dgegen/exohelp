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
    """
    mass = mass if isinstance(mass, u.Quantity) else u.Quantity(mass, "M_earth")
    radius = radius if isinstance(radius, u.Quantity) else u.Quantity(radius, "R_earth")

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
    """
    mass = mass if isinstance(mass, u.Quantity) else u.Quantity(mass, "M_earth")
    radius = radius if isinstance(radius, u.Quantity) else u.Quantity(radius, "R_earth")
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
    """
    g = surface_gravity(mass, radius).to(u.cm / u.s**2)
    return np.log10(g.value)
