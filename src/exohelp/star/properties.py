import astropy.constants as const
import astropy.units as u
import numpy as np

from ..type import QuantityLike

__all__ = ["kennedy_kenyon_snowline", "luminosity"]


def kennedy_kenyon_snowline(m_star: QuantityLike, x: float = 2.0) -> u.Quantity:
    """Calculates the snow line distance based on Kennedy & Kenyon (2008).

    Reference: Kennedy & Kenyon (2008)
    https://ui.adsabs.harvard.edu/abs/2008ApJ...673..502K/abstract

    Parameters
    ----------
    m_star : QuantityLike
        Stellar mass. Assumed to be in Solar masses if no unit is given.
    x : float
        Scaling exponent (usually between 1.5 and 2.0).
        KK08 suggest x=2 for the early stages of disk evolution.

    Returns
    -------
    a_snow : Quantity
        Distance to the snow line in AU.

    Examples
    --------
    >>> from exohelp.star.properties import kennedy_kenyon_snowline
    >>> round(float(kennedy_kenyon_snowline(1.0).value), 2)
    2.7
    """
    m_star = u.Quantity(m_star, "M_sun").value
    return u.Quantity(2.7 * m_star**x, "au")


def luminosity(teff: QuantityLike, r_star: QuantityLike = 1.0) -> u.Quantity:
    """Compute stellar luminosity from effective temperature and radius via the Stefan-Boltzmann law.

        L = 4pi R*^2 sigma T_eff^4

    Parameters
    ----------
    teff : QuantityLike
        Stellar effective temperature. Assumed to be in Kelvin if no unit is given.
    r_star : QuantityLike
        Stellar radius. Assumed to be in Solar radii if no unit is given.

    Returns
    -------
    L : Quantity
        Stellar luminosity in Solar luminosities.

    Examples
    --------
    >>> from exohelp.star.properties import luminosity
    >>> round(float(luminosity(5778, 1.0).value), 2)  # Sun ≈ 1 L_sun
    1.0
    """
    teff = u.Quantity(teff, "K")
    r_star = u.Quantity(r_star, "R_sun")
    return (4 * np.pi * r_star**2 * const.sigma_sb * teff**4).to("L_sun")  # type: ignore[attr-defined]
