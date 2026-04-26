import numpy as np
from astropy import constants as const
from astropy import units as u

from ..type import QuantityLike

__all__ = ["planet_mass_from_rv", "rv_semi_amplitude"]

_PRE_FACTOR = ((2 * np.pi * const.G) ** (-1 / 3)).to("day(2/3) M_sun(1/3) / R_sun")  # type: ignore[attr-defined]


def planet_mass_from_rv(
    rv_semi_amplitude: QuantityLike,
    period: QuantityLike,
    eccentricity: QuantityLike = 0.0,
    m_star: QuantityLike = 1.0,
    inclination: QuantityLike = 90.0,
    n_iterations: int = 5,
) -> u.Quantity:
    """Calculate the planet mass from the radial velocity semi-amplitude.

    Solves the exact Keplerian relation via fixed-point iteration, avoiding the
    small-planet approximation (M_star >> m_p) that breaks down for massive
    companions.

    Parameters
    ----------
    rv_semi_amplitude : QuantityLike
        RV semi-amplitude K. Assumed to be in m/s if no unit is given.
    period : QuantityLike
        Orbital period. Assumed to be in days if no unit is given.
    eccentricity : QuantityLike
        Orbital eccentricity. Default is 0.
    m_star : QuantityLike
        Stellar mass. Assumed to be in Solar masses if no unit is given.
    inclination : QuantityLike
        Orbital inclination. Assumed to be in degrees if no unit is given.
        For minimum mass (m sin i) use the default of 90 degrees.
    n_iterations : int
        Number of fixed-point iterations. Default is 10; fewer than 5 are
        needed for convergence even for brown-dwarf companions.

    Returns
    -------
    m_planet : u.Quantity
        Planet mass in Earth masses.

    Notes
    -----
    The exact relation is:

        K = (2*pi*G/P)^(1/3) * m_p sin(i) / ((M_star + m_p)^(2/3) * sqrt(1 - e^2))

    Rearranging, with _PRE_FACTOR carrying units so that m_planet is in Solar masses:
    throughout the iteration and the final result is converted to Earth masses:

        m_p [M_sun] = _PRE_FACTOR * K * sqrt(1-e^2) * P^(1/3)
                        * (M_star + m_p)^(2/3) / sin(i)

    Solved by fixed-point iteration starting from the small-planet approximation.
    Convergence is reached in fewer than 5 steps even for brown-dwarf companions.

    References
    ----------
    Lovis, C. & Fischer, D. (2010), in Exoplanets, edited by S. Seager. Published by University of Arizona Press, Tucson, AZ, 2010, 526 pp. ISBN 978-0-8165-2945-2., p.27-53
    https://ui.adsabs.harvard.edu/abs/2010exop.book...27L/abstract

    Example
    -------
    >>> from astropy import units as u
    >>> m_planet = planet_mass_from_rv(
    ...     rv_semi_amplitude=u.Quantity(8.95, "cm") / u.Quantity(1, "s"),
    ...     period=u.Quantity(1, "year"),
    ... )
    >>> f"{m_planet:.1f}"
    '1.0 earthMass'
    """
    rv_semi_amplitude = (
        rv_semi_amplitude
        if isinstance(rv_semi_amplitude, u.Quantity)
        else u.Quantity(rv_semi_amplitude, "m/s")
    )
    period = u.Quantity(period, "day")
    m_star = u.Quantity(m_star, "M_sun")
    inclination = u.Quantity(inclination, "deg")

    k = rv_semi_amplitude.to("m/s")
    sin_i = np.sin(inclination.to("rad").value)
    ecc_factor = np.sqrt(1 - eccentricity**2)

    # Fixed-point: m_p [M_earth] = prefactor * (M_star + m_p * M_earth/M_sun)^(2/3)
    # where masses in the sum are in M_sun
    prefactor = _PRE_FACTOR * k * ecc_factor * period ** (1 / 3) / sin_i

    m_planet = prefactor * m_star ** (2 / 3)  # small-planet starting point [M_earth]
    for _ in range(n_iterations):
        m_planet = prefactor * (m_star + m_planet) ** (2 / 3)

    return m_planet.to("M_earth")


def rv_semi_amplitude(
    m_planet: QuantityLike,
    period: QuantityLike,
    eccentricity: QuantityLike = 0.0,
    m_star: QuantityLike = 1.0,
    inclination: QuantityLike = 90.0,
) -> u.Quantity:
    """Calculate the RV semi-amplitude K for a planet on a Keplerian orbit.

        K = (2*pi*G/P)^(1/3) * m_p sin(i) / ((M_star + m_p)^(2/3) * sqrt(1 - e^2))

    Parameters
    ----------
    m_planet : QuantityLike
        Planet mass. Assumed to be in Earth masses if no unit is given.
    period : QuantityLike
        Orbital period. Assumed to be in days if no unit is given.
    eccentricity : QuantityLike
        Orbital eccentricity. Default is 0.
    m_star : QuantityLike
        Stellar mass. Assumed to be in Solar masses if no unit is given.
    inclination : QuantityLike
        Orbital inclination. Assumed to be in degrees if no unit is given.
        For minimum mass (m sin i) use the default of 90 degrees.

    Returns
    -------
    K : Quantity
        RV semi-amplitude in m/s.

    References
    ----------
    Lovis, C. & Fischer, D. (2010), in Exoplanets, edited by S. Seager. Published by University of Arizona Press, Tucson, AZ, 2010, 526 pp. ISBN 978-0-8165-2945-2., p.27-53
    https://ui.adsabs.harvard.edu/abs/2010exop.book...27L/abstract

    Example
    -------
    >>> from astropy import units as u
    >>> K = rv_semi_amplitude(
    ...     m_planet=u.Quantity(1, "M_earth"),
    ...     period=u.Quantity(1, "year"),
    ... )
    >>> f"{K.to('cm/s'):.2f}"
    '8.95 cm / s'
    """
    m_planet = u.Quantity(m_planet, "M_earth")
    period = u.Quantity(period, "day")
    m_star = u.Quantity(m_star, "M_sun")
    inclination = u.Quantity(inclination, "deg")

    sin_i = np.sin(inclination.to("rad").value)
    ecc_factor = np.sqrt(1 - eccentricity**2)

    m_total_msun = m_star + m_planet.to("M_sun")
    k = m_planet * sin_i / (_PRE_FACTOR * ecc_factor * period ** (1 / 3) * m_total_msun ** (2 / 3))
    return k.to("m/s")
