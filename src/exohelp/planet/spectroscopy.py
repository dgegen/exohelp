"""
Atmospheric scale height, transmission signal size, and spectroscopy metrics.

Transmission and Emission Spectroscopy Metrics from Kempton et al. (2018).

Useful source
https://github.com/nespinoza/kevinADS/blob/main/utils.py
"""

import astropy.constants as const
import astropy.units as u
import numpy as np
from astropy.modeling.models import BlackBody

from ..type import QuantityLike

__all__ = [
    "emission_spectroscopy_metric",
    "scale_height",
    "transmission_signal_size",
    "transmission_spectroscopy_metric",
]


def scale_height(
    temperature: QuantityLike,
    gravity: QuantityLike,
    mean_molecular_weight: QuantityLike = 2.3,
) -> u.Quantity:
    """Compute the atmospheric scale height H = k_B T / (μ m_H g).

    Parameters
    ----------
    temperature : QuantityLike
        Atmospheric temperature (representative of the limb for transmission).
        Assumed to be in Kelvin if no unit is given.
    gravity : QuantityLike
        Surface gravity. Assumed to be in m/s² if no unit is given.
    mean_molecular_weight : QuantityLike
        Mean molecular weight in atomic mass units. Default is 2.3,
        appropriate for a H₂-dominated atmosphere.

    Returns
    -------
    H : Quantity
        Atmospheric scale height in km.

    References
    ----------
    Winn, J. N. (2010), in Exoplanets, edited by S. Seager. Published by University of Arizona Press, Tucson, AZ, 2010, 526 pp. ISBN 978-0-8165-2945-2., p.55-77
    https://ui.adsabs.harvard.edu/abs/2010exop.book...55W/abstract
    """
    temperature = (
        temperature if isinstance(temperature, u.Quantity) else u.Quantity(temperature, "K")
    )
    gravity = gravity if isinstance(gravity, u.Quantity) else u.Quantity(gravity, "m / s^2")
    mean_molecular_weight = (
        mean_molecular_weight
        if isinstance(mean_molecular_weight, u.Quantity)
        else u.Quantity(mean_molecular_weight, "u")
    )
    return (const.k_B * temperature / (mean_molecular_weight * gravity)).to("km")  # type: ignore[attr-defined]


def transmission_signal_size(
    scale_height: QuantityLike,
    r_planet: QuantityLike,
    r_star: QuantityLike,
    n_scale_heights: float = 1.0,
) -> u.Quantity:
    """Compute the transit depth change per N atmospheric scale heights in ppm.

    The signal is the annular area of N scale heights of atmosphere projected
    against the stellar disk:

        ΔD = 2 N H R_p / R_*²

    Parameters
    ----------
    scale_height : QuantityLike
        Atmospheric scale height. Assumed to be in km if no unit is given.
    r_planet : QuantityLike
        Planet radius. Assumed to be in Earth radii if no unit is given.
    r_star : QuantityLike
        Stellar radius. Assumed to be in Solar radii if no unit is given.
    n_scale_heights : float
        Number of scale heights contributing to the signal. Default is 1.

    Returns
    -------
    signal : Quantity
        Transmission signal in ppm.

    References
    ----------
    Winn, J. N. (2010), in Exoplanets, edited by S. Seager. Published by University of Arizona Press, Tucson, AZ, 2010, 526 pp. ISBN 978-0-8165-2945-2., p.55-77
    https://ui.adsabs.harvard.edu/abs/2010exop.book...55W/abstract

    de Wit, J. & Seager, S. (2013), Science, 342, 1473.
    https://doi.org/10.1126/science.1245450
    """
    scale_height = (
        scale_height if isinstance(scale_height, u.Quantity) else u.Quantity(scale_height, "km")
    )
    r_planet = r_planet if isinstance(r_planet, u.Quantity) else u.Quantity(r_planet, "R_earth")
    r_star = r_star if isinstance(r_star, u.Quantity) else u.Quantity(r_star, "R_sun")

    signal_ppm = (2 * n_scale_heights * scale_height * r_planet / r_star**2).decompose().value * 1e6
    result = np.asarray(signal_ppm)
    return result.item() if result.ndim == 0 else result


def _get_scale_factor(r_planet: QuantityLike) -> np.ndarray:
    """Get the scale factor based on the radius of the planet.

    Value taken from Table 1 of Kempton et al. (2018).
    https://doi.org/10.1088/1538-3873/aadf6f
    """
    if isinstance(r_planet, u.Quantity):
        r_planet = r_planet.to("R_earth").value

    conditions = [
        r_planet < 1.5,
        (r_planet >= 1.5) & (r_planet < 2.75),
        (r_planet >= 2.75) & (r_planet < 4),
        (r_planet >= 4) & (r_planet < 10),
    ]
    values = [0.190, 1.26, 1.28, 1.15]

    return np.select(conditions, values, default=np.nan)


def transmission_spectroscopy_metric(
    r_planet: QuantityLike,
    m_planet: QuantityLike,
    teq_planet: QuantityLike,
    r_star: QuantityLike,
    jmag_star: QuantityLike,
) -> np.ndarray:
    """
    Compute the Transmission Spectroscopy Metric (TSM) following Kempton et al. (2018).

    The TSM is an empirical ranking metric used to estimate the suitability of transiting
    exoplanets for atmospheric characterization via transmission spectroscopy. It scales
    with the expected signal-to-noise of spectral features, combining planetary size,
    temperature, and surface gravity with host star brightness in the near-infrared.

    Parameters
    ----------
    r_planet : QuantityLike
        Planet radius. If a Quantity, it will be converted to Earth radii.
    m_planet : QuantityLike
        Planet mass. If a Quantity, it will be converted to Earth masses.
    teq_planet : QuantityLike
        Planet equilibrium temperature. If a Quantity, it will be converted to Kelvin.
    r_star : QuantityLike
        Stellar radius. If a Quantity, it will be converted to solar radii.
    jmag_star : QuantityLike
        Apparent magnitude of the host star in the J band (e.g., 2MASS J).
        If a Quantity, it will be converted to magnitudes.

    Returns
    -------
    tsm : ndarray
        Transmission Spectroscopy Metric (dimensionless). Higher values indicate more
        favorable targets for transmission spectroscopy.

    Notes
    -----
    The metric is defined as:

        TSM ∝ (R_p^3 * T_eq) / (M_p * R_*^2) * 10^(-J/5)

    with a scale factor that depends on planetary radius, as defined in
    Kempton et al. (2018). The J-band magnitude is used as a proxy for
    near-infrared brightness relevant to instruments such as JWST/NIRISS.

    References
    ----------
    Kempton, E. M.-R., et al. (2018), PASP, 130, 114401.
    https://doi.org/10.1088/1538-3873/aadf6f

    Example
    -------
    >>> r_planet = 1.5 * u.R_earth
    >>> m_planet = 5.0 * u.M_earth
    >>> teq_planet = 500 * u.K
    >>> r_star = 0.5 * u.R_sun
    >>> jmag_star = 8 * u.mag
    >>> transmission_spectroscopy_metric(r_planet, m_planet, teq_planet, r_star, jmag_star)
    """
    r_planet = u.Quantity(r_planet, "R_earth")
    m_planet = u.Quantity(m_planet, "M_earth")
    teq_planet = u.Quantity(teq_planet, "K")
    r_star = u.Quantity(r_star, "R_sun")
    jmag_star = u.Quantity(jmag_star, "mag")

    scale_factor = _get_scale_factor(r_planet)
    tsm = scale_factor * (
        (r_planet.value**3 * teq_planet.value)
        / (m_planet.value * r_star.value**2)
        * 10 ** (-jmag_star.value / 5)
    )
    return tsm.item() if np.ndim(tsm) == 0 else tsm


def _planck_lambda(wavelength: u.Quantity, temperature: u.Quantity) -> u.Quantity:
    """Planck spectral radiance B_lambda(λ, T) (W m^-3 sr^-1)"""
    return BlackBody(temperature=temperature)(wavelength).to(
        u.W / (u.m**3 * u.sr),  # type: ignore[union-attr]
        equivalencies=u.spectral_density(wavelength),
    )


def emission_spectroscopy_metric(
    r_planet: QuantityLike,
    teq_planet: QuantityLike,
    r_star: QuantityLike,
    kmag_star: QuantityLike,
    teff_star: QuantityLike,
    wavelength: QuantityLike = u.Quantity(7.5, "micron"),
) -> np.ndarray:
    """
    Calculate the Emission Spectroscopy Metric (ESM) from Kempton et al. (2018).

    ESM = 4.29e6 * [B_lambda(λ, T_day) / B_lambda(λ, T_star)] * (Rp / Rstar)^2 * 10^(-m_K/5)

    - T_day = 1.10 * Teq (dayside temperature, as in the paper).
    - Default wavelength is 7.5 micron (Kempton uses mid-IR; paper uses 7.5 micron convention).

    References
    ----------
    Kempton, E. M.-R., et al. (2018), Publications of the Astronomical Society of the Pacific,
    Volume 130, Issue 993, pp. 114401 (2018).
    https://ui.adsabs.harvard.edu/abs/2018PASP..130k4401K/abstract

    Parameters
    ----------
    r_planet : QuantityLike
        Planet radius (R_earth or astropy Quantity convertible to R_earth).
    teq_planet : QuantityLike
        Planet equilibrium temperature (K or astropy Quantity convertible to K).
    r_star : QuantityLike
        Stellar radius (R_sun or astropy Quantity convertible to R_sun).
    kmag_star : QuantityLike
        Stellar apparent magnitude in the K band (mag or astropy Quantity convertible to mag).
    teff_star : QuantityLike
        Stellar effective temperature (K or astropy Quantity convertible to K).
    wavelength : astropy.units.Quantity, optional
        Wavelength to evaluate Planck functions (default 7.5 micron).

    Returns
    -------
    numpy.ndarray
        ESM value(s) — scalar if inputs are scalars.
    """
    r_planet = u.Quantity(r_planet, "R_earth")
    teq_planet = u.Quantity(teq_planet, "K")
    r_star = u.Quantity(r_star, "R_sun")
    kmag_star = u.Quantity(kmag_star, "mag")
    teff_star = u.Quantity(teff_star, "K")
    wavelength = u.Quantity(wavelength, "m")

    day_side_temperature = 1.10 * teq_planet
    b_day = _planck_lambda(wavelength, day_side_temperature)
    b_star = _planck_lambda(wavelength, teff_star)
    planck_ratio = (b_day / b_star).value

    geometric_factor = (r_planet / r_star).decompose().value ** 2
    mag_factor = 10 ** (-kmag_star.to("mag").value / 5.0)
    norm_factor = 4.29e6  # Normalization from Kempton et al. (2018)

    esm = norm_factor * planck_ratio * geometric_factor * mag_factor

    return esm if esm.size > 1 else esm.item()
