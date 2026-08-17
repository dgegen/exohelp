import numpy as np
import astropy.units as u
from astropy.table import QTable

from .spectroscopy import (
    microturbulent_velocity_bruntt2010,
    macroturbulent_velocity_bruntt2010,
    macroturbulent_velocity_doyle2014,
    rotation_period_from_vsini,
)
from ..body import bulk_density
from ..stats import truncated_normal
from .properties import luminosity


def derive_stellar_parameters(
    teff: float,
    teff_err: float,
    logg: float,
    logg_err: float,
    mass: float,
    mass_err: float,
    radius: float,
    radius_err: float,
    vsini: float,
    vsini_err: float,
    inclination_star: float = 90.0,
    n_samples: int = 100_000,
    seed: int | None = None,
) -> QTable:
    """
    Monte Carlo uncertainty propagation for stellar parameters.

    Parameters
    ----------
    teff, teff_err : float
        Effective temperature and 1-sigma uncertainty in K.
    logg, logg_err : float
        Surface gravity and 1-sigma uncertainty in dex.
    mass, mass_err : float
        Stellar mass and 1-sigma uncertainty in solar masses.
    radius, radius_err : float
        Stellar radius and 1-sigma uncertainty in solar radii.
    vsini, vsini_err : float
        Projected rotational velocity and 1-sigma uncertainty in km/s.
    inclination_star : float
        Stellar inclination in degrees (default 90°, edge-on).
    n_samples : int
        Number of Monte Carlo samples.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    QTable
        Sampled input observables and derived quantities.
    """
    rng = np.random.default_rng(seed)

    teff_s = u.Quantity(truncated_normal(teff, teff_err, n_samples, lower=0.0, rng=rng), "K")
    logg_s = u.Dex(rng.normal(logg, logg_err, n_samples) * (u.cm / u.s**2))
    mass_s = u.Quantity(truncated_normal(mass, mass_err, n_samples, lower=0.0, rng=rng), "M_sun")
    radius_s = u.Quantity(
        truncated_normal(radius, radius_err, n_samples, lower=0.0, rng=rng), "R_sun"
    )
    vsini_s = u.Quantity(truncated_normal(vsini, vsini_err, n_samples, lower=0.0, rng=rng), "km/s")

    luminosity_s = luminosity(teff_s, radius_s)
    density_s = bulk_density(mass_s, radius_s)
    v_mic_s = microturbulent_velocity_bruntt2010(teff_s)
    v_mac_bruntt_s = macroturbulent_velocity_bruntt2010(teff_s)
    v_mac_doyle_s = macroturbulent_velocity_doyle2014(teff_s, logg_s)
    prot_s = rotation_period_from_vsini(vsini_s, radius_s, inclination_star)

    table = QTable(
        {
            "teff": teff_s,
            "logg": logg_s,
            "mass": mass_s,
            "radius": radius_s,
            "vsini": vsini_s,
            "luminosity": luminosity_s,
            "density": density_s,
            "v_mic": v_mic_s,
            "v_mac_bruntt": v_mac_bruntt_s,
            "v_mac_doyle": v_mac_doyle_s,
            "rotation_period": prot_s,
        }
    )

    table["v_mic"].description = "Micro-turbulent velocity (Bruntt et al. 2010, Eq. 10)"  # type: ignore
    table["v_mac_bruntt"].description = "Macro-turbulent velocity (Bruntt et al. 2010, Eq. 9)"  # type: ignore
    table["v_mac_doyle"].description = "Macro-turbulent velocity (Doyle et al. 2014, Eq. 8)"  # type: ignore
    table["rotation_period"].description = (  # type: ignore
        "Rotation period from vsini and stellar radius"
    )

    return table
