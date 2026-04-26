import astropy.constants as const
import astropy.units as u
import numpy as np

from exohelp.type import QuantityLike

__all__ = ["luminosity"]


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
