import astropy.units as u
import numpy as np
from typing import NamedTuple


from ..kepler import keplers_third_law
from ..type import QuantityLike, ArrayLike
from ..units import S_earth

__all__ = [
    "EccentricEquilibriumTemperature",
    "equilibrium_temperature",
    "equilibrium_temperature_eccentric",
    "hill_sphere_radius",
    "insolation_flux",
    "periapsis_distance",
    "periastron_distance",
]


class EccentricEquilibriumTemperature(NamedTuple):
    periastron: u.Quantity
    apastron: u.Quantity
    flux_averaged: u.Quantity


def periastron_distance(
    semi_major_axis: QuantityLike, eccentricity: float | np.ndarray = 0.0
) -> u.Quantity:
    """Compute the periastron (periapsis) distance q = a (1 - e).

    Parameters
    ----------
    semi_major_axis : QuantityLike
        Orbital semi-major axis. Assumed to be in AU if no unit is given.
    eccentricity : float or array-like
        Orbital eccentricity. Default is 0 (circular).

    Returns
    -------
    q : Quantity
        Periastron distance in AU.

    Examples
    --------
    >>> from exohelp.planet import periastron_distance
    >>> periastron_distance(1.0, eccentricity=0.5)
    <Quantity 0.5 AU>
    """
    semi_major_axis = u.Quantity(semi_major_axis, "AU")
    return semi_major_axis * (1 - eccentricity)


# alias spelling
periapsis_distance = periastron_distance


def insolation_flux(
    luminosity: QuantityLike,
    semi_major_axis: QuantityLike,
    eccentricity: float | np.ndarray = 0.0,
) -> u.Quantity:
    """Compute the stellar insolation flux received by a planet relative to Earth's.

        <S> / S⊕ = (L★ / L⊙) (a / AU)⁻² / sqrt(1-e²)

    Parameters
    ----------
    luminosity : QuantityLike
        Stellar luminosity. Assumed to be in Solar luminosities if no unit is given.
    semi_major_axis : QuantityLike
        Orbital semi-major axis. Assumed to be in AU if no unit is given.
    eccentricity : float or array-like
        Orbital eccentricity. Default is 0 (circular).

    Returns
    -------
    S : Quantity
        Insolation flux in units of S⊕ (Earth's insolation).

    Examples
    --------
    >>> from exohelp.planet import insolation_flux
    >>> round(float(insolation_flux(1.0, 1.0).value), 1)  # Earth around Sun
    1.0
    """
    luminosity = u.Quantity(luminosity, "L_sun")
    semi_major_axis = u.Quantity(semi_major_axis, "AU")
    s_at_a = (luminosity.to("L_sun") / (semi_major_axis.to("AU")) ** 2).value
    return s_at_a / np.sqrt(1 - eccentricity**2) * S_earth


def hill_sphere_radius(
    semi_major_axis: QuantityLike,
    m_planet: QuantityLike,
    m_star: QuantityLike = 1.0,
    eccentricity: ArrayLike = 0.0,
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
    eccentricity : QuantityLike
        Orbital eccentricity. Default is 0 (circular).

    Returns
    -------
    r_H : Quantity
        Hill sphere radius in AU.

    References
    ----------
    Hamilton, D. P. & Burns, J. A. (1992), Icarus, 96, 43.
    https://doi.org/10.1016/0019-1035(92)90005-R

    Examples
    --------
    >>> from exohelp.planet import hill_sphere_radius
    >>> round(float(hill_sphere_radius(1.0, 1.0).value), 3)  # Earth at 1 AU ≈ 0.01 AU
    0.01
    """
    semi_major_axis = u.Quantity(semi_major_axis, "AU")
    m_planet = u.Quantity(m_planet, "M_earth")
    m_star = u.Quantity(m_star, "M_sun")
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

    Examples
    --------
    >>> from exohelp.planet import equilibrium_temperature
    >>> round(equilibrium_temperature(5778, semi_major_axis=1.0).value)  # Earth (A=0)
    279
    """
    if period is not None:
        period = u.Quantity(period, "day")
        semi_major_axis = keplers_third_law(period=period, mass=m_star)
    elif semi_major_axis is not None:
        semi_major_axis = u.Quantity(semi_major_axis, "AU")
    else:
        raise ValueError("Either semi_major_axis or period must be provided.")

    teff_star = u.Quantity(teff_star, "K")
    r_star = u.Quantity(r_star, "R_sun")
    m_star = u.Quantity(m_star, "M_sun")

    t_eq_circular = (
        teff_star
        * np.sqrt(r_star.to("R_sun") / (2 * semi_major_axis.to("R_sun")))
        * (1 - bond_albedo) ** (1 / 4)
    )

    return t_eq_circular


def equilibrium_temperature_eccentric(
    teff_star: float | np.ndarray | u.Quantity,
    semi_major_axis: None | float | np.ndarray | u.Quantity = None,
    period: None | float | np.ndarray | u.Quantity = None,
    r_star: float | np.ndarray | u.Quantity = 1,
    m_star: float | np.ndarray | u.Quantity = 1,
    bond_albedo: float | np.ndarray | u.Quantity = 0,
    eccentricity: float | np.ndarray = 0.0,
) -> EccentricEquilibriumTemperature:
    """Compute the planetary equilibrium temperature at periastron.

        T_eq,periastron = T_eq,circular / (1-e)^{1/2}
        T_eq,apastron = T_eq,circular / (1+e)^{1/2}
        T_eq,flux-averaged = T_eq,circular / (1-e²)^{1/8}

    Parameters
    ----------
    eccentricity : float or array-like
        Orbital eccentricity. Default is 0 (circular).

    Other parameters see :func:`equilibrium_temperature` for parameter descriptions.

    Returns
    -------
    t_eq_periastron : Quantity
        Equilibrium temperature at periastron in Kelvin.
    t_eq_apastron : Quantity
        Equilibrium temperature at apastron in Kelvin.
    t_eq_flux_averaged : Quantity
        Flux-averaged equilibrium temperature in Kelvin.

    References
    ----------
    Andreas Quirrenbach 2022 Res. Notes AAS 6 56
    https://doi.org/10.3847/2515-5172/ac5f0d

    Examples
    --------
    >>> from exohelp.planet import equilibrium_temperature_eccentric
    >>> result = equilibrium_temperature_eccentric(5778, semi_major_axis=1.0, eccentricity=0.0)
    >>> round(result.periastron.value)  # circular: same as T_eq
    279
    >>> bool(result.periastron.value == result.apastron.value)
    True
    """
    t_eq_circular = equilibrium_temperature(
        teff_star=teff_star,
        semi_major_axis=semi_major_axis,
        period=period,
        r_star=r_star,
        m_star=m_star,
        bond_albedo=bond_albedo,
    )
    t_eq_flux_averaged = t_eq_circular / (1 - eccentricity**2) ** (1 / 8)
    t_eq_periastron = t_eq_circular / np.sqrt(1 - eccentricity)
    t_eq_apastron = t_eq_circular / np.sqrt(1 + eccentricity)

    return EccentricEquilibriumTemperature(
        periastron=t_eq_periastron, apastron=t_eq_apastron, flux_averaged=t_eq_flux_averaged
    )
