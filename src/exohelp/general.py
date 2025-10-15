import astropy.units as u
import numpy as np


def calculate_bulk_density(mass, radius):
    """Calculate the density of a planet given its mass and radius in Earth units.

    Parameters
    ----------
    mass : float or array-like
        Mass of the planet in Earth masses.
    radius : float or array-like
        Radius of the planet in Earth radii.

    Returns
    -------
    density_g_cm3 : float or array-like
        Density of the planet in grams per cubic centimeter (g/cm³).
    """
    mass = mass if isinstance(mass, u.Quantity) else mass * u.M_earth
    radius = radius if isinstance(radius, u.Quantity) else radius * u.R_earth

    volume = (4 / 3) * np.pi * radius**3
    density = mass / volume

    return density.to(u.g / u.cm**3)
