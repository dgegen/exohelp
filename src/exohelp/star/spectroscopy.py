"""
stellar_spectroscopy.py

Stellar spectroscopy-related functions.

Sources:
- Bruntt et al. (2010), Monthly Notices of the Royal Astronomical Society, Vol. 405, Issue 3, pp. 1907-1923
  https://ui.adsabs.harvard.edu/abs/2010MNRAS.405.1907B/abstract
- Doyle et al. (2014), Monthly Notices of the Royal Astronomical Society, Vol. 444, Issue 4, pp. 3592-3608
  https://ui.adsabs.harvard.edu/abs/2014MNRAS.444.3592D/abstract

All functions are vectorized and unit-aware where appropriate.
"""

import numpy as np
import astropy.units as u
from astropy.table import QTable


__all__ = ["sample_rotation_period_from_vsini", "sample_v_mic_and_v_mac"]


def calculate_microturbulent_velocity_bruntt2010(teff):
    """
    Micro-turbulent velocity from effective temperature.

    Reference: Bruntt et al. (2010), Eq. 10.

    Valid for teff 5000-6500 K and logg > 4.0.

    Parameters
    ----------
    teff : float
        Effective temperature in K.

    Returns
    -------
    v_mic : Quantity [km/s]

    Examples
    --------
    >>> from exohelp.star.spectroscopy import calculate_microturbulent_velocity_bruntt2010
    >>> calculate_microturbulent_velocity_bruntt2010(5700)  # at calibration point
    <Quantity 1.01 km / s>
    """
    delta_t = teff - 5700
    v_mic = 1.01 + (4.56e-4 * delta_t) + (2.75e-7 * delta_t**2)
    return u.Quantity(v_mic, "km / s")


def calculate_macroturbulent_velocity_bruntt2010(teff):
    """
    Macro-turbulent velocity from effective temperature.

    Reference: Bruntt et al. (2010), Eq. 9.

    Parameters
    ----------
    teff : float
        Effective temperature in K.

    Returns
    -------
    v_mac : Quantity [km/s]

    Examples
    --------
    >>> from exohelp.star.spectroscopy import calculate_macroturbulent_velocity_bruntt2010
    >>> calculate_macroturbulent_velocity_bruntt2010(5700)  # at calibration point
    <Quantity 2.26 km / s>
    """
    delta_t = teff - 5700
    v_mac = 2.26 + (2.90e-3 * delta_t) + (5.86e-7 * delta_t**2)
    return u.Quantity(v_mac, "km / s")


def calculate_v_mac_doyle2014(teff, logg):
    """
    Macro-turbulent velocity from effective temperature and surface gravity.

    Reference: Doyle et al. (2014), Eq. 8.

    Valid for teff in [5200, 6400] K and logg in [4.0, 4.6] dex.

    Parameters
    ----------
    teff : float
        Effective temperature in K.
    logg : float
        Surface gravity in dex (log10 cgs).

    Returns
    -------
    v_mac : Quantity [km/s]

    Examples
    --------
    >>> from exohelp.star.spectroscopy import calculate_v_mac_doyle2014
    >>> calculate_v_mac_doyle2014(5777, 4.44)  # solar values
    <Quantity 3.21 km / s>
    """
    t_diff = teff - 5777
    v_mac = 3.21 + (2.33e-3 * t_diff) + (2.0e-6 * t_diff**2) - (2.0 * (logg - 4.44))
    return u.Quantity(v_mac, "km / s")


def rotation_period_from_vsini(vsini, r_star, inclination_star=90.0):
    """
    Calculate the stellar rotation period from vsini, stellar radius, and inclination angle.

    Parameters
    ----------
    vsini : float or Quantity
        Projected rotational velocity in km/s.
    r_star : float or Quantity
        Stellar radius in solar radii.
    inclination_star : float or Quantity, optional
        Stellar obliquity (inclination of spin axis) in degrees. Default 90°.

    Returns
    -------
    rotation_period : Quantity [days]
        Stellar rotation period. If inclination_star is unknown, this is an upper limit.
        Best guess for unknown inclination: <sin i> = pi/4 ≈ 0.785.

    Examples
    --------
    >>> from exohelp.star.spectroscopy import rotation_period_from_vsini
    >>> round(float(rotation_period_from_vsini(2.0, 1.0).value), 1)  # 2 km/s, 1 R_sun, edge-on
    25.3
    """
    vsini = u.Quantity(vsini, "km / s")
    r_star = u.Quantity(r_star, "R_sun")
    inclination_star = u.Quantity(inclination_star, "deg")

    rotation_velocity = vsini / np.sin(inclination_star.to("rad").value)
    rotation_period = (2 * np.pi * r_star / rotation_velocity).to("day")

    return rotation_period


def sample_v_mic_and_v_mac(teff, teff_err, logg, logg_err, n_samples=100_000, seed=None):
    """
    Monte Carlo sampling of micro-turbulent and macro-turbulent velocities from effective temperature and surface gravity.

    Parameters
    ----------
    teff : float
        Effective temperature in K.
    teff_err : float
        1-sigma uncertainty on effective temperature.
    logg : float
        Surface gravity in dex (log10 cgs).
    logg_err : float
        1-sigma uncertainty on surface gravity.
    n_samples : int
        Number of Monte Carlo samples (default 1000).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    v_mic_samples : Quantity [km/s]
        Micro-turbulent velocity samples.
    v_mac_bruntt_samples : Quantity [km/s]
        Macro-turbulent velocity samples from Bruntt et al. (2010).
    v_mac_doyle_samples : Quantity [km/s]
        Macro-turbulent velocity samples from Doyle et al. (2014).
    """
    rng = np.random.default_rng(seed)
    teff_s = rng.normal(teff, teff_err, n_samples)
    logg_s = rng.normal(logg, logg_err, n_samples)

    v_mic_samples = calculate_microturbulent_velocity_bruntt2010(teff_s)
    v_mac_bruntt_samples = calculate_macroturbulent_velocity_bruntt2010(teff_s)
    v_mac_doyle_samples = calculate_v_mac_doyle2014(teff_s, logg_s)

    table = QTable(
        [teff_s, logg_s, v_mic_samples, v_mac_bruntt_samples, v_mac_doyle_samples],
        names=["teff", "logg", "v_mic", "v_mac_bruntt", "v_mac_doyle"],
    )
    table["v_mic"].description = "Micro-turbulent velocity (Bruntt et al. 2010, Eq. 10)"  # type: ignore
    table["v_mac_bruntt"].description = "Macro-turbulent velocity (Bruntt et al. 2010, Eq. 9)"  # type: ignore
    table["v_mac_doyle"].description = "Macro-turbulent velocity (Doyle et al. 2014, Eq. 8)"  # type: ignore

    return table


def sample_rotation_period_from_vsini(
    vsini, vsini_err, r_star, r_star_err, inclination_star=90.0, n_samples=100_000, seed=None
):
    """
    Monte Carlo sampling of rotation period from vsini, stellar radius, and inclination angle.

    Parameters
    ----------
    vsini : float
        Projected rotational velocity in km/s.
    vsini_err : float
        1-sigma uncertainty on vsini.
    r_star : float
        Stellar radius in solar radii.
    r_star_err : float
        1-sigma uncertainty on stellar radius.
    inclination_star : float or Quantity, optional
        Stellar inclination (angle between spin axis and line of sight) in degrees.
        Default 90° (edge-on). If unknown, 90° delivers an upper limit on the rotation
        period.
    n_samples : int
        Number of Monte Carlo samples (default 1000).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    rotation_period_samples : Quantity [days]
        Rotation period samples, masked where invalid.
    """
    rng = np.random.default_rng(seed)
    vsini_s = rng.normal(vsini, vsini_err, n_samples)
    r_star_s = rng.normal(r_star, r_star_err, n_samples)

    max_rotation_period_samples = rotation_period_from_vsini(vsini_s, r_star_s, inclination_star)

    table = QTable(
        [vsini_s, r_star_s, max_rotation_period_samples],
        names=["vsini", "r_star", "max_rotation_period"],
    )

    return table
