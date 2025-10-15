import numpy as np
from astropy import constants as const
from astropy import units as u


PRE_FACTOR: float = (
    (2 * np.pi * const.G) ** (-1 / 3)  # type: ignore[attr-defined]
    * const.M_sun ** (2 / 3)  # type: ignore[attr-defined]
    * (u.day.to("second") * u.second) ** (1 / 3)  # type: ignore[attr-defined]
    * const.M_earth ** (-1)  # type: ignore[attr-defined]
).value


def calculate_planet_mass(
    rv_semi_amplitude: np.ndarray | float | u.Quantity,
    period: np.ndarray | float | u.Quantity,
    eccentricity: np.ndarray | float = 0.0,
    stellar_mass: np.ndarray | float | u.Quantity = 1.0,
    inclination: np.ndarray | float | u.Quantity = 90.0,
    jupiter_mass: bool = False,
) -> np.ndarray:
    """Calculate the minimum mass of a planet given the radial velocity semi-amplitude,
    orbital period, eccentricity, and stellar mass.

    Returns
    -------
    np.ndarray
        Minimum mass of the planet in Earth masses.

    Example
    -------
    >>> from astropy import units as u
    >>> planet_mass = calculate_planet_mass(
    ...     rv_semi_amplitude=8.95 * u.cm / u.s,
    ...     period=1 * u.year,
    ... )
    >>> f"{planet_mass:.1f}"
    '1.0'
    """
    if isinstance(period, u.Quantity):
        period = period.to("day").value
    if isinstance(rv_semi_amplitude, u.Quantity):
        rv_semi_amplitude = rv_semi_amplitude.to("m / s").value
    if isinstance(stellar_mass, u.Quantity):
        stellar_mass = stellar_mass.to("solar mass").value
    if isinstance(inclination, u.Quantity):
        inclination = inclination.to("degree").value

    planet_mass = (
        #  1.898 / 5.9722*10**3 / 28.4329
        PRE_FACTOR
        * rv_semi_amplitude
        * np.sqrt(1 - eccentricity**2)
        * stellar_mass ** (2 / 3)
        * period ** (1 / 3)
        * np.sin(np.radians(inclination)) ** (-1)
    )

    if jupiter_mass:
        planet_mass *= const.M_earth.to("M_jup").value  # type: ignore[attr-defined]

    return planet_mass
