import astropy.units as u
import numpy as np


def get_scale_factor(radius_planet: float | u.Quantity | np.ndarray) -> np.ndarray:
    """Get the scale factor based on the radius of the planet.

    Value taken from Table 1 of Kempton et al. (2018).
    https://doi.org/10.1088/1538-3873/aadf6f

    Parameters
    ----------
    radius_planet : float, astropy.units.Quantity, or np.ndarray
        Radius of the planet in Earth radii.
    Returns
    -------
    scale_factor : np.ndarray
        Scale factor corresponding to the planet radius.

    Example
    --------
    >>> radius_planet = np.array([1.0, 1.6, 2.5, 3.0, 4.5, 8.0, 10.0])
    >>> get_scale_factor(radius_planet)
    """
    if isinstance(radius_planet, u.Quantity):
        radius_planet = radius_planet.to("R_earth").value

    conditions = [
        radius_planet < 1.5,
        (radius_planet >= 1.5) & (radius_planet < 2.75),
        (radius_planet >= 2.75) & (radius_planet < 4),
        (radius_planet >= 4) & (radius_planet < 10),
    ]
    values = [0.190, 1.26, 1.28, 1.15]

    return np.select(conditions, values, default=np.nan)


def calculate_tsm(
    radius_planet: float | u.Quantity,
    mass_planet: float | u.Quantity,
    teq_planet: float | u.Quantity,
    radius_star: float | u.Quantity,
    apparent_magnitude_star: float | u.Quantity,
) -> np.ndarray:
    """Calculate the Transmission Spectroscopy Metric (TSM) of Kempton et al. (2018).

    https://doi.org/10.1088/1538-3873/aadf6f
    """
    if isinstance(radius_planet, u.Quantity):
        radius_planet = radius_planet.to("R_earth").value
    if isinstance(mass_planet, u.Quantity):
        mass_planet = mass_planet.to("M_earth").value
    if isinstance(teq_planet, u.Quantity):
        teq_planet = teq_planet.to("K").value
    if isinstance(radius_star, u.Quantity):
        radius_star = radius_star.to("R_sun").value
    if isinstance(apparent_magnitude_star, u.Quantity):
        apparent_magnitude_star = apparent_magnitude_star.to("mag").value

    scale_factor = get_scale_factor(radius_planet)
    tsm = scale_factor * (
        ((radius_planet) ** 3 * teq_planet)
        / (mass_planet * (radius_star) ** 2)
        * 10 ** (-apparent_magnitude_star / 5)
    )
    return tsm
