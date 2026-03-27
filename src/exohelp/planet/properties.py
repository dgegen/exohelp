import astropy.units as u
import numpy as np

from ..kepler import keplers_third_law
from ..type import QuantityLike

__all__ = [
    "equilibrium_temperature",
    "hill_sphere_radius",
    "insolation_flux",
]


def insolation_flux(
    luminosity: QuantityLike,
    semi_major_axis: QuantityLike,
) -> u.Quantity:
    """Compute the stellar insolation flux received by a planet relative to Earth's.

        S / S⊕ = (L★ / L⊙) (a / AU)⁻²

    Parameters
    ----------
    luminosity : QuantityLike
        Stellar luminosity. Assumed to be in Solar luminosities if no unit is given.
    semi_major_axis : QuantityLike
        Orbital semi-major axis. Assumed to be in AU if no unit is given.

    Returns
    -------
    S : Quantity
        Insolation flux in units of S⊕ (Earth's insolation).
    """
    luminosity = (
        luminosity if isinstance(luminosity, u.Quantity) else u.Quantity(luminosity, "L_sun")
    )
    semi_major_axis = (
        semi_major_axis
        if isinstance(semi_major_axis, u.Quantity)
        else u.Quantity(semi_major_axis, "AU")
    )
    l_sun = u.Quantity(1, "L_sun")
    return (luminosity / l_sun) * (u.Quantity(1, "AU") / semi_major_axis.to("AU")) ** 2


def hill_sphere_radius(
    semi_major_axis: QuantityLike,
    m_planet: QuantityLike,
    m_star: QuantityLike = 1.0,
    eccentricity: float = 0.0,
) -> u.Quantity:
    """Compute the Hill sphere radius of a planet.

        r_H = a (1 - e) (m_p / 3 M*)^(1/3)

    The Hill sphere marks the region within which the planet dominates over the
    star gravitationally, setting an upper limit on stable satellite orbits and
    atmosphere retention.

    Parameters
    ----------
    semi_major_axis : QuantityLike
        Orbital semi-major axis. Assumed to be in AU if no unit is given.
    m_planet : QuantityLike
        Planet mass. Assumed to be in Earth masses if no unit is given.
    m_star : QuantityLike
        Stellar mass. Assumed to be in Solar masses if no unit is given.
    eccentricity : float
        Orbital eccentricity. Default is 0 (circular).

    Returns
    -------
    r_H : Quantity
        Hill sphere radius in AU.

    References
    ----------
    Hamilton, D. P. & Burns, J. A. (1992), Icarus, 96, 43.
    https://doi.org/10.1016/0019-1035(92)90005-R
    """
    semi_major_axis = (
        semi_major_axis
        if isinstance(semi_major_axis, u.Quantity)
        else u.Quantity(semi_major_axis, "AU")
    )
    m_planet = m_planet if isinstance(m_planet, u.Quantity) else u.Quantity(m_planet, "M_earth")
    m_star = m_star if isinstance(m_star, u.Quantity) else u.Quantity(m_star, "M_sun")
    mass_ratio = (m_planet / (3 * m_star)).decompose().value
    return (semi_major_axis * (1 - eccentricity) * mass_ratio ** (1 / 3)).to("AU")


def equilibrium_temperature(
    teff_star: float | np.ndarray | u.Quantity,
    semi_major_axis: None | float | np.ndarray | u.Quantity = None,
    period: None | float | np.ndarray | u.Quantity = None,
    r_star: float | np.ndarray | u.Quantity = 1,
    m_star: float | np.ndarray | u.Quantity = 1,
    bond_albedo: float | np.ndarray | u.Quantity = 0,
) -> u.Quantity:
    """Compute the planetary equilibrium temperature.

        T_eq = T★ * sqrt(R★ / 2a) * (1 - A)^(1/4)

    Parameters
    ----------
    teff_star : QuantityLike
        Stellar effective temperature. Assumed to be in Kelvin if no unit is given.
    semi_major_axis : QuantityLike, optional
        Orbital semi-major axis. Assumed to be in AU if no unit is given.
        Either ``semi_major_axis`` or ``period`` must be provided.
    period : QuantityLike, optional
        Orbital period. Assumed to be in days if no unit is given.
        Either ``semi_major_axis`` or ``period`` must be provided.
    r_star : QuantityLike
        Stellar radius. Assumed to be in Solar radii if no unit is given.
    m_star : QuantityLike
        Stellar mass. Assumed to be in Solar masses if no unit is given.
        Only used when ``period`` is given to derive the semi-major axis.
    bond_albedo : QuantityLike
        Geometric albedo. Default is 0 (perfect absorber).

    Returns
    -------
    T_eq : Quantity
        Equilibrium temperature in Kelvin.
    """
    if period is not None:
        if not isinstance(period, u.Quantity):
            period = u.Quantity(period, "day")
        semi_major_axis = keplers_third_law(period=period, mass=m_star)
    elif semi_major_axis is not None:
        if not isinstance(semi_major_axis, u.Quantity):
            semi_major_axis = u.Quantity(semi_major_axis, "AU")
    else:
        raise ValueError("Either semi_major_axis or period must be provided.")

    if not isinstance(teff_star, u.Quantity):
        teff_star = u.Quantity(teff_star, "K")
    if not isinstance(r_star, u.Quantity):
        r_star = u.Quantity(r_star, "R_sun")
    if not isinstance(m_star, u.Quantity):
        m_star = u.Quantity(m_star, "M_sun")

    return (
        teff_star
        * np.sqrt(r_star.to("R_sun") / (2 * semi_major_axis.to("R_sun")))
        * (1 - bond_albedo) ** (1 / 4)
    )
