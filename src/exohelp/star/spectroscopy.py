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
from astropy.coordinates import LSR, SkyCoord
from astropy.table import QTable

from ..type import QuantityLike


__all__ = [
    "bensby_membership_probabilities",
    "ccf_indicator_uncertainties",
    "classify_td_to_d_ratio",
    "sample_rotation_period_from_vsini",
    "sample_uvw_lsr",
    "sample_v_mic_and_v_mac",
]


BENSBY_POPULATION_PARAMETERS = {
    "Thin Disk": {
        "sigma_U": 35.0,
        "sigma_V": 20.0,
        "sigma_W": 16.0,
        "U_asym": 0.0,
        "V_asym": -15.0,
        "X": 0.85,
    },
    "Thick Disk": {
        "sigma_U": 67.0,
        "sigma_V": 38.0,
        "sigma_W": 35.0,
        "U_asym": 0.0,
        "V_asym": -46.0,
        "X": 0.09,
    },
    "Halo": {
        "sigma_U": 160.0,
        "sigma_V": 90.0,
        "sigma_W": 90.0,
        "U_asym": 0.0,
        "V_asym": -220.0,
        "X": 0.0015,
    },
    "Hercules": {
        "sigma_U": 26.0,
        "sigma_V": 9.0,
        "sigma_W": 17.0,
        "U_asym": -40.0,
        "V_asym": -50.0,
        "X": 0.06,
    },
}


def microturbulent_velocity_bruntt2010(teff: QuantityLike) -> u.Quantity:
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
    >>> from exohelp.star.spectroscopy import microturbulent_velocity_bruntt2010
    >>> microturbulent_velocity_bruntt2010(5700)  # at calibration point
    <Quantity 1.01 km / s>
    """
    if isinstance(teff, u.Quantity):
        teff = teff.to("K").value
    delta_t = teff - 5700
    v_mic = 1.01 + (4.56e-4 * delta_t) + (2.75e-7 * delta_t**2)
    return u.Quantity(v_mic, "km / s")


def macroturbulent_velocity_bruntt2010(teff: QuantityLike) -> u.Quantity:
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
    >>> from exohelp.star.spectroscopy import macroturbulent_velocity_bruntt2010
    >>> macroturbulent_velocity_bruntt2010(5700)  # at calibration point
    <Quantity 2.26 km / s>
    """
    if isinstance(teff, u.Quantity):
        teff = teff.to("K").value
    delta_t = teff - 5700
    v_mac = 2.26 + (2.90e-3 * delta_t) + (5.86e-7 * delta_t**2)
    return u.Quantity(v_mac, "km / s")


def macroturbulent_velocity_doyle2014(teff: QuantityLike, logg: QuantityLike) -> u.Quantity:
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
    >>> from exohelp.star.spectroscopy import macroturbulent_velocity_doyle2014
    >>> macroturbulent_velocity_doyle2014(5777, 4.44)  # solar values
    <Quantity 3.21 km / s>
    """
    if isinstance(teff, u.Quantity):
        teff = teff.to("K").value
    t_diff = teff - 5777
    v_mac = 3.21 + (2.33e-3 * t_diff) + (2.0e-6 * t_diff**2) - (2.0 * (logg - 4.44))
    return u.Quantity(v_mac, "km / s")


def rotation_period_from_vsini(
    vsini: QuantityLike, r_star: QuantityLike, inclination_star: QuantityLike = 90.0
) -> u.Quantity:
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


def sample_v_mic_and_v_mac(
    teff: float,
    teff_err: float,
    logg: float,
    logg_err: float,
    n_samples: int = 100_000,
    seed: int | None = None,
):
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

    if isinstance(teff, u.Quantity):
        teff = teff.to("K").value
    if isinstance(teff_err, u.Quantity):
        teff_err = teff_err.to("K").value
    if isinstance(logg, u.Quantity):
        logg = logg.to("dex").value
    if isinstance(logg_err, u.Quantity):
        logg_err = logg_err.to("dex").value

    teff_s = rng.normal(teff, teff_err, n_samples)
    logg_s = rng.normal(logg, logg_err, n_samples)

    v_mic_samples = microturbulent_velocity_bruntt2010(teff_s)
    v_mac_bruntt_samples = macroturbulent_velocity_bruntt2010(teff_s)
    v_mac_doyle_samples = macroturbulent_velocity_doyle2014(teff_s, logg_s)

    table = QTable(
        [teff_s, logg_s, v_mic_samples, v_mac_bruntt_samples, v_mac_doyle_samples],
        names=["teff", "logg", "v_mic", "v_mac_bruntt", "v_mac_doyle"],
    )
    table["v_mic"].description = "Micro-turbulent velocity (Bruntt et al. 2010, Eq. 10)"  # type: ignore
    table["v_mac_bruntt"].description = "Macro-turbulent velocity (Bruntt et al. 2010, Eq. 9)"  # type: ignore
    table["v_mac_doyle"].description = "Macro-turbulent velocity (Doyle et al. 2014, Eq. 8)"  # type: ignore

    return table


def sample_rotation_period_from_vsini(
    vsini: float,
    vsini_err: float,
    r_star: float,
    r_star_err: float,
    inclination_star: float = 90.0,
    n_samples: int = 100_000,
    seed: int | None = None,
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

    if isinstance(inclination_star, u.Quantity):
        inclination_star = inclination_star.to("deg").value
    if isinstance(vsini, u.Quantity):
        vsini = vsini.to("km/s").value
    if isinstance(vsini_err, u.Quantity):
        vsini_err = vsini_err.to("km/s").value
    if isinstance(r_star, u.Quantity):
        r_star = r_star.to("R_sun").value
    if isinstance(r_star_err, u.Quantity):
        r_star_err = r_star_err.to("R_sun").value

    vsini_s = rng.normal(vsini, vsini_err, n_samples)
    r_star_s = rng.normal(r_star, r_star_err, n_samples)

    max_rotation_period_samples = rotation_period_from_vsini(vsini_s, r_star_s, inclination_star)

    table = QTable(
        [vsini_s, r_star_s, max_rotation_period_samples],
        names=["vsini", "r_star", "max_rotation_period"],
    )

    return table


def sample_uvw_lsr(
    ra: QuantityLike,
    ra_err: QuantityLike,
    dec: QuantityLike,
    dec_err: QuantityLike,
    distance: QuantityLike,
    distance_err: QuantityLike,
    pm_ra_cosdec: QuantityLike,
    pm_ra_cosdec_err: QuantityLike,
    pm_dec: QuantityLike,
    pm_dec_err: QuantityLike,
    radial_velocity: QuantityLike,
    radial_velocity_err: QuantityLike,
    n_samples: int = 100_000,
    seed: int | None = None,
) -> QTable:
    """
    Monte Carlo sampling of Galactic velocity components (U, V, W) in the LSR frame.

    Astrometric and radial-velocity inputs are sampled independently as Gaussian
    distributions with provided 1-sigma uncertainties, converted to ICRS
    coordinates, and transformed to ``LSR()``.

    Parameters
    ----------
    ra, ra_err : QuantityLike
        Right ascension and 1-sigma uncertainty. Assumed deg if unitless.
    dec, dec_err : QuantityLike
        Declination and 1-sigma uncertainty. Assumed deg if unitless.
    distance, distance_err : QuantityLike
        Distance and 1-sigma uncertainty. Assumed pc if unitless.
    pm_ra_cosdec, pm_ra_cosdec_err : QuantityLike
        Proper motion in RA*cos(dec) and 1-sigma uncertainty.
        Assumed mas/yr if unitless.
    pm_dec, pm_dec_err : QuantityLike
        Proper motion in Dec and 1-sigma uncertainty. Assumed mas/yr if unitless.
    radial_velocity, radial_velocity_err : QuantityLike
        Radial velocity and 1-sigma uncertainty. Assumed km/s if unitless.
    n_samples : int
        Number of Monte Carlo samples.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    QTable
        Table with sampled observables and LSR velocities. U is positive toward
        the Galactic center, V along Galactic rotation, and W toward the north
        Galactic pole.
    """
    rng = np.random.default_rng(seed)

    ra = u.Quantity(ra, "deg").value
    ra_err = u.Quantity(ra_err, "deg").value
    dec = u.Quantity(dec, "deg").value
    dec_err = u.Quantity(dec_err, "deg").value
    distance = u.Quantity(distance, "pc").value
    distance_err = u.Quantity(distance_err, "pc").value
    pm_ra_cosdec = u.Quantity(pm_ra_cosdec, "mas / yr").value
    pm_ra_cosdec_err = u.Quantity(pm_ra_cosdec_err, "mas / yr").value
    pm_dec = u.Quantity(pm_dec, "mas / yr").value
    pm_dec_err = u.Quantity(pm_dec_err, "mas / yr").value
    radial_velocity = u.Quantity(radial_velocity, "km / s").value
    radial_velocity_err = u.Quantity(radial_velocity_err, "km / s").value

    ra_s = u.Quantity(rng.normal(ra, ra_err, n_samples), "deg")
    dec_s = u.Quantity(rng.normal(dec, dec_err, n_samples), "deg")
    distance_s = u.Quantity(rng.normal(distance, distance_err, n_samples), "pc")
    pm_ra_cosdec_s = u.Quantity(rng.normal(pm_ra_cosdec, pm_ra_cosdec_err, n_samples), "mas / yr")
    pm_dec_s = u.Quantity(rng.normal(pm_dec, pm_dec_err, n_samples), "mas / yr")
    radial_velocity_s = u.Quantity(
        rng.normal(radial_velocity, radial_velocity_err, n_samples), "km / s"
    )

    coords_icrs = SkyCoord(
        ra=ra_s,
        dec=dec_s,
        distance=distance_s,
        pm_ra_cosdec=pm_ra_cosdec_s,
        pm_dec=pm_dec_s,
        radial_velocity=radial_velocity_s,
        frame="icrs",
    )
    coords_lsr = coords_icrs.transform_to(LSR())
    velocity = coords_lsr.velocity
    if velocity is None:
        raise ValueError("Velocity information is required to compute U, V, W in LSR.")
    u_lsr = u.Quantity(velocity.d_x, "km / s")  # type: ignore[arg-type]
    v_lsr = u.Quantity(velocity.d_y, "km / s")  # type: ignore[arg-type]
    w_lsr = u.Quantity(velocity.d_z, "km / s")  # type: ignore[arg-type]

    table = QTable(
        [
            ra_s,
            dec_s,
            distance_s,
            pm_ra_cosdec_s,
            pm_dec_s,
            radial_velocity_s,
            u_lsr,
            v_lsr,
            w_lsr,
        ],
        names=[
            "ra",
            "dec",
            "distance",
            "pm_ra_cosdec",
            "pm_dec",
            "radial_velocity",
            "U_lsr",
            "V_lsr",
            "W_lsr",
        ],
    )

    return table


def bensby_membership_probabilities(
    u_lsr: QuantityLike,
    v_lsr: QuantityLike,
    w_lsr: QuantityLike,
) -> QTable:
    """
    Compute Bensby-style kinematic membership probabilities from LSR velocities.

    This uses Gaussian velocity ellipsoids with asymmetric drift and population
    fractions for Thin Disk, Thick Disk, Halo, and Hercules stream.

    Parameters
    ----------
    u_lsr, v_lsr, w_lsr : QuantityLike
        Galactic velocity components in the LSR frame. Assumed km/s if unitless.

    Returns
    -------
    QTable
        Probabilities for each population and the Thick-to-Thin ratio ``TD_to_D``.
        Values are relative and need not sum to 1.

    Examples
    --------
    >>> from exohelp.star.spectroscopy import bensby_membership_probabilities
    >>> probs = bensby_membership_probabilities(u_lsr=5.0, v_lsr=-10.0, w_lsr=2.0)
    >>> f"{probs['Thin Disk'].value[0]:.1e}"
    '4.6e-06'
    """
    u_lsr = np.atleast_1d(u.Quantity(u_lsr, "km / s").value)
    v_lsr = np.atleast_1d(u.Quantity(v_lsr, "km / s").value)
    w_lsr = np.atleast_1d(u.Quantity(w_lsr, "km / s").value)

    probabilities: dict[str, np.ndarray] = {}

    for population, params in BENSBY_POPULATION_PARAMETERS.items():
        sigma_u = params["sigma_U"]
        sigma_v = params["sigma_V"]
        sigma_w = params["sigma_W"]
        u_asym = params["U_asym"]
        v_asym = params["V_asym"]
        fraction = params["X"]

        norm = 1.0 / ((2.0 * np.pi) ** 1.5 * sigma_u * sigma_v * sigma_w)
        exponent = -(
            (u_lsr - u_asym) ** 2 / (2.0 * sigma_u**2)
            + (v_lsr - v_asym) ** 2 / (2.0 * sigma_v**2)
            + (w_lsr**2) / (2.0 * sigma_w**2)
        )
        probabilities[population] = np.atleast_1d(fraction * norm * np.exp(exponent))

    td_to_d = probabilities["Thick Disk"] / probabilities["Thin Disk"]

    return QTable(
        [
            probabilities["Thin Disk"],
            probabilities["Thick Disk"],
            probabilities["Halo"],
            probabilities["Hercules"],
            td_to_d,
        ],
        names=["Thin Disk", "Thick Disk", "Halo", "Hercules", "TD_to_D"],
    )


def classify_td_to_d_ratio(
    td_to_d_ratio: QuantityLike,
    thin_threshold: float = 0.5,
    thick_threshold: float = 2.0,
) -> np.ndarray:
    """
    Classify stars from the Thick-to-Thin membership ratio.

    Parameters
    ----------
    td_to_d_ratio : QuantityLike
        Thick-to-Thin ratio (dimensionless).
    thin_threshold : float
        Upper threshold for "Thin Disk" classification.
    thick_threshold : float
        Lower threshold for "Thick Disk" classification.

    Returns
    -------
    ndarray
        Classification labels: "Thin Disk", "Thick Disk", or "In-between".
    """
    ratio = np.asarray(u.Quantity(td_to_d_ratio, u.dimensionless_unscaled).value, dtype=float)

    classifications = np.full(ratio.shape, "In-between", dtype="<U16")
    classifications[ratio < thin_threshold] = "Thin Disk"
    classifications[ratio > thick_threshold] = "Thick Disk"

    return classifications


def ccf_indicator_uncertainties(rv_error: QuantityLike, instrument: str = "HARPS") -> dict:
    """
    Computes photon-noise uncertainties for FWHM, BIS, and Contrast.

    Parameters
    ----------
    rv_error : QuantityLike
        The uncertainty in the radial velocity measurement (e_rv). Assumed to be in m/s if no unit is given.
    instrument : str
        The spectrograph used for the RV measurement.
        Must be one of 'HARPS', 'SOPHIE_HR', or 'SOPHIE_HE'. Default is 'HARPS'.

    Returns
    -------
    dict
        A dictionary containing the uncertainties for FWHM, BIS, and Contrast:
        {
            'sigma_fwhm_err': Quantity,  # Uncertainty in FWHM (m/s)
            'sigma_bis_err': Quantity,   # Uncertainty in BIS (m/s)
            'sigma_contrast_err': Quantity,  # Uncertainty in Contrast (%)
        }
    """
    # Scaling coefficients (epsilon_k) from Table 1 of Santerne et al. (2015)
    # These coefficients assume sigma_RV is in km/s to yield results in m/s or %.
    # Since input is in m/s, we adjust the math accordingly.
    # The source defines epsilon_contrast relative to RV in km/s.
    # To get sigma_contrast in %, we must convert e_rv from m/s to km/s first.
    # (Mathematically: epsilon_k * (e_rv/1000) * 1000)
    configs = {
        "HARPS": {"fwhm": 2.0, "bis": 2.0, "contrast": 11.3 / 1000.0},
        "SOPHIE_HR": {"fwhm": 2.5, "bis": 2.3, "contrast": 10.2 / 1000.0},
        "SOPHIE_HE": {"fwhm": 2.5, "bis": 2.6, "contrast": 7.0 / 1000.0},
    }

    if instrument not in configs:
        raise ValueError(f"Instrument must be one of {list(configs.keys())}")

    eps = configs[instrument]

    rv_error = u.Quantity(rv_error, "m/s")

    # 1. FWHM and BIS uncertainties
    sigma_fwhm = eps["fwhm"] * rv_error
    sigma_bis = eps["bis"] * rv_error

    # 2. Contrast uncertainty (Result in %)
    sigma_contrast = u.Quantity(eps["contrast"] * rv_error.value, "%")

    return {
        "sigma_fwhm_err": sigma_fwhm,
        "sigma_bis_err": sigma_bis,
        "sigma_contrast_err": sigma_contrast,
    }
