"""
stellar_activity.py

Rotation period and age estimation from chromospheric activity.

Sources:
- Noyes et al. (1984), Astrophysical Journal, Vol. 279, p. 763-777
  https://ui.adsabs.harvard.edu/abs/1984ApJ...279..763N/abstract
- Mamajek & Hillenbrand (2008), ApJ 687 1264DOI 10.1086/591785
  https://ui.adsabs.harvard.edu/abs/2008ApJ...687.1264M/abstract
- Mittag et al. (2018), Astronomy & Astrophysics, Volume 618, id.A48, 12 pp.
  https://ui.adsabs.harvard.edu/abs/2018A%26A...618A..48M/abstract

All functions are vectorized and unit-aware where appropriate.
"""

import numpy as np
import astropy.units as u
from astropy.table import QTable


__all__ = ["sample_rotation_period_and_age"]


def age_mamajek2008(log_rhk, jitter=None):
    """
    Chromospheric age from log R'_HK.

    Reference: Mamajek & Hillenbrand (2008), Eq. (3).

    Valid for:
        -5.1 < log R'_HK < -4.0
        6.7 < log10(age/yr) < 9.9

    Parameters
    ----------
    log_rhk : array_like
        log10(R'_HK)
    jitter : array_like, optional
        Optional jitter term to account for intrinsic variability in log R'_HK.
        Samples from a Unit Normal Distribution N(0, 1).
        If provided, age is inflated by the empirical model scatter:
        0.11 dex for log_rhk <= -4.3, 0.23 dex for log_rhk > -4.3.

    Returns
    -------
    age : Quantity [Gyr]
        Masked outside valid range.
    """

    log_rhk = np.asarray(log_rhk, dtype=float)

    log_tau = -38.053 - 17.912 * log_rhk - 1.6675 * log_rhk**2

    if jitter is not None:
        jitter = np.broadcast_to(jitter, log_rhk.shape)

        very_active_mask = log_rhk > -4.3
        active_mask = (log_rhk >= -5.1) & (log_rhk <= -4.3)
        very_active_sigma = 0.23  # dex
        active_sigma = 0.11  # dex

        log_tau[very_active_mask] += jitter[very_active_mask] * very_active_sigma
        log_tau[active_mask] += jitter[active_mask] * active_sigma

    age = (10**log_tau) * u.year

    mask_valid = (log_rhk > -5.1) & (log_rhk < -4.0)

    return np.ma.array(age.to(u.Gyr).value, mask=~mask_valid) * u.Gyr


def log_rhk_from_age_mamajek2008(age):
    """
    Inverse chromospheric relation: log R'_HK from age.

    Reference: Mamajek & Hillenbrand (2008), Eq. (4).

    Valid for:
        -5.1 < log R'_HK < -4.0
        6.7 < log10(age/yr) < 9.9

    Parameters
    ----------
    age : Quantity
        Age (must be convertible to years).

    Returns
    -------
    log_rhk : MaskedArray
        log10(R'_HK), masked outside valid range.
    """
    age = u.Quantity(age).to(u.year).value
    log_tau = np.log10(age)

    valid_mask = (log_tau > 6.7) & (log_tau < 9.9)

    log_rhk = 8.94 - 4.849 * log_tau + 0.624 * log_tau**2 - 0.028 * log_tau**3

    return np.ma.array(log_rhk, mask=~valid_mask)


def tau_c_noyes1984(bv):
    """
    Local convective turnover time from B-V color.

    Reference: Noyes et al. (1984), Eq. (4).
    """
    bv = np.asarray(bv, dtype=float)
    x = 1.0 - bv

    # This polynomial calculates log10(tau_c)
    log10_tau_c = np.empty_like(x)

    # Noyes 1984 specifies the polynomial for x > 0 (B-V < 1.0)
    # and a linear relation for x < 0 (B-V > 1.0)
    mask = x > 0
    log10_tau_c[mask] = 1.362 - 0.166 * x[mask] + 0.025 * x[mask] ** 2 - 5.323 * x[mask] ** 3
    log10_tau_c[~mask] = 1.362 - 0.14 * x[~mask]

    return (10**log10_tau_c) * u.day


def tau_c_mittag2018(bv):
    """
    Global convective turnover time from B-V color.

    Reference: Mittag et al. (2018), Eqs. (11) & (12).

    Valid for:
        B-V >= 0.44

    Parameters
    ----------
    bv : array_like
        B-V color.

    Returns
    -------
    tau_c : Quantity [days]
        Masked where B-V < 0.44.
    """
    bv = np.asarray(bv, dtype=float)

    log10_tau_c = np.full_like(bv, np.nan)

    mask_1 = (bv >= 0.44) & (bv <= 0.71)
    mask_2 = bv > 0.71
    mask_valid = bv >= 0.44

    # Eq. 11
    log10_tau_c[mask_1] = 1.06 + 2.33 * (bv[mask_1] - 0.44)

    # Eq. 12
    log10_tau_c[mask_2] = 1.69 + 0.69 * (bv[mask_2] - 0.71)

    # extrapolate (flagged as invalid)
    # log10_tau_c[~mask_valid] = 1.06 + 2.33 * (bv[~mask_valid] - 0.44)

    tau_c = 10**log10_tau_c

    return np.ma.array(tau_c, mask=~mask_valid) * u.day


def rotation_period_mittag2018(rhk_plus, bv, slope=0.15):
    """
    Calculates the rotation period from magnetic activity excess and B-V color.

    Equations:
        log10(P [day]) = log10(tau_c(B-V) [day]) + f(R_HK^+)  (Eq. 9)
        f(R_HK^+) = -A * (R_HK^+ * 10^5)                      (Eq. 10)

    Parameters:
    -----------
    rhk_plus : array_like
        The dimensionless excess flux indicator (R+_HK).
    bv : array_like
        B-V color index.
    slope : float
        Universal slope A (default 0.15 ± 0.02 from Mittag et al. 2018).

    Returns:
    --------
    prot : Quantity [days]
        Estimated rotation period.
    """
    global_convective_turnover_time = tau_c_mittag2018(bv)
    tau_c_val = np.ma.getdata(global_convective_turnover_time.to(u.day).value)
    tau_c_mask = np.ma.getmaskarray(global_convective_turnover_time)

    f_rhk = -slope * (np.asarray(rhk_plus) * 1e5)
    log_p = np.log10(tau_c_val) + f_rhk

    return np.ma.array(10**log_p, mask=tau_c_mask) * u.day


def gyro_age_barnes2010(prot, tau_c):
    """
    Calculates stellar age from rotation period and global convective turnover time.

    Reference: Barnes (2010) as cited in Mittag et al. (2018), Eq. (13).

    Equation:
        t_age = (tau/k_c) * ln(P/P_0) + (k_i / (2*tau)) * (P^2 - P_0^2)

    Note: Standard constants from Barnes (2010) are used here.
    """
    p = prot.to(u.day).value
    tau = tau_c.to(u.day).value

    # Constants from Barnes (2010) / Mittag (2018) snippet
    p0 = 1.1  # [days]
    kc = 0.646  # [day/Myr]
    ki = 452  # [Myr/day]

    term1 = (tau / kc) * np.log(p / p0)
    term2 = (ki / (2 * tau)) * (p**2 - p0**2)

    age_myr = term1 + term2
    return (age_myr * u.Myr).to(u.Gyr)


def rotation_period_noyes1984(log_rhk, tau_c):
    """
    Rotation period from log R'_HK and local convective turnover time.

    Reference: Noyes et al. (1984), Eq. (3).

    Valid for:
        -5.5 < log R'_HK < -4.3

    Parameters
    ----------
    log_rhk : array_like
        log10(R'_HK)
    tau_c : Quantity [days]
        Local convective turnover time.

    Returns
    -------
    prot : Quantity [days]
        Masked where log_rhk is outside [-5.5, -4.3] or tau_c is masked.
    """
    log_rhk = np.asarray(log_rhk, dtype=float)
    tau_c_val = np.ma.getdata(tau_c.to(u.day).value)
    tau_mask = np.ma.getmaskarray(tau_c)

    mask_valid = (log_rhk < -4.3) & (log_rhk > -5.5)

    y = 5 + log_rhk

    log_prot = 0.324 - 0.400 * y - 0.283 * y**2 - 1.325 * y**3 + np.log10(tau_c_val)

    prot = (10**log_prot) * u.day

    combined_mask = (~mask_valid) | tau_mask

    return np.ma.array(prot.value, mask=combined_mask) * u.day


def rossby_number_mamajek2008(log_rhk):
    """
    Rossby number (Ro = P / tau_c) from log R'_HK.

    Reference: Mamajek & Hillenbrand (2008), Eqs. (5) & (7).

    Valid for:
        log R'_HK >= -5.0

    Parameters
    ----------
    log_rhk : array_like
        log10(R'_HK)

    Returns
    -------
    rossby_number : MaskedArray
        Masked where log_rhk < -5.0.
    """
    log_rhk = np.asarray(log_rhk, dtype=float)

    rossby_number = np.full_like(log_rhk, np.nan)

    # Regimes
    mask_very_active = log_rhk > -4.3
    mask_active = (log_rhk >= -5.0) & (log_rhk <= -4.3)
    mask_valid = log_rhk >= -5.0

    # Eq. (5): very active
    rossby_number[mask_very_active] = 0.233 - 0.689 * (log_rhk[mask_very_active] + 4.23)

    # Eq. (7): active
    rossby_number[mask_active] = 0.808 - 2.966 * (log_rhk[mask_active] + 4.52)

    return np.ma.array(rossby_number, mask=~mask_valid)


def rotation_period_mamajek2008(log_rhk, tau_c):
    """
    Rotation period from Rossby number and convective turnover time.

    Reference: Mamajek & Hillenbrand (2008), Eqs. (5) & (7).

    Valid for:
        log R'_HK >= -5.0

    Parameters
    ----------
    log_rhk : array_like
        log10(R'_HK)
    tau_c : Quantity [days]
        Convective turnover time.

    Returns
    -------
    prot : Quantity [days]
        Masked where log_rhk < -5.0 or tau_c is masked.
    """
    if not isinstance(tau_c, u.Quantity):
        raise ValueError("tau_c must be an astropy Quantity with time units.")

    log_rhk = np.asarray(log_rhk, dtype=float)

    rossby_number = rossby_number_mamajek2008(log_rhk)
    ro_val = np.ma.getdata(rossby_number)
    ro_mask = np.ma.getmaskarray(rossby_number)

    tau_c_val = np.ma.getdata(tau_c.to(u.day).value)
    tau_mask = np.ma.getmaskarray(tau_c)

    prot = ro_val * tau_c_val
    combined_mask = ro_mask | tau_mask

    return np.ma.array(prot, mask=combined_mask) * u.day


def gyro_age_mamajek2008(prot, bv, a=0.407, b=0.325, c=0.495, n=0.566):
    """
    Gyrochronological age from rotation period and B-V color.

    Reference: Mamajek & Hillenbrand (2008), Eqs. (12)-(14).

    Valid for:
        0.5 < B-V < 0.9
        prot > 0

    Parameters
    ----------
    prot : Quantity
        Rotation period (must be convertible to days).
    bv : array_like
        B-V color.
    a : float
        Gyrochronology coefficient (default 0.407 ± 0.021).
    b : float
        Gyrochronology coefficient (default 0.325 ± 0.024).
    c : float
        Gyrochronology coefficient (default 0.495 ± 0.010).
    n : float
        Gyrochronology coefficient (default 0.566 ± 0.008).

    Returns
    -------
    age : Quantity [Gyr]
        Masked where B-V or prot are outside valid range.
    """
    prot_val = np.ma.getdata(prot.to(u.day).value)
    prot_mask = np.ma.getmaskarray(prot)
    bv = np.asarray(bv, dtype=float)

    a, b, c, n = (np.broadcast_to(x, prot_val.shape).copy() for x in (a, b, c, n))

    mask_valid = (prot_val > 0) & (bv > 0.5) & (bv < 0.9)

    f_bv = a[mask_valid] * (bv[mask_valid] - c[mask_valid]) ** b[mask_valid]

    age = np.full_like(prot_val, np.nan)
    age[mask_valid] = 1e-3 * (prot_val[mask_valid] / f_bv) ** (1 / n[mask_valid])

    combined_mask = (~mask_valid) | prot_mask

    return np.ma.array(age, mask=combined_mask) * u.Gyr


def sample_rotation_period_and_age(
    log_rhk, log_rhk_err, mag_b, mag_b_err, mag_v, mag_v_err, n_samples=1000, seed=None
):
    """
    Monte Carlo sampling of rotation period and age from chromospheric activity and
    B-V color.

    Parameters
    ----------
    log_rhk : float
        log10(R'_HK)
    log_rhk_err : float
        1-sigma uncertainty on log_rhk.
    mag_b : float
        B magnitude.
    mag_b_err : float
        1-sigma uncertainty on B magnitude.
    mag_v : float
        V magnitude.
    mag_v_err : float
        1-sigma uncertainty on V magnitude.
    n_samples : int
        Number of Monte Carlo samples (default 100_000).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    prot_samples : Quantity [days]
        Rotation period samples, masked where invalid.
    age_samples : Quantity [Gyr]
        Gyrochronological age samples, masked where invalid.

    Example
    -------
    >>> table = sample_rotation_period_and_age(mag_b=9.941, mag_b_err=0.029, mag_v=9.33, mag_v_err=0.023, log_rhk=-5.04, log_rhk_err=0.09, n_samples=100_000)
    >>> summary = table.to_pandas().describe(percentiles=[0.16, 0.5, 0.84])
    >>> summary.loc['upper_error'] = summary.loc['84%'] - summary.loc['50%']
    >>> summary.loc['lower_error'] = summary.loc['50%'] - summary.loc['16%']
    >>> summary.loc[:, ['mag_bv', 'prot', 'age_gyro', 'age_chromo']].round(3)
    """
    rng = np.random.default_rng(seed)
    mag_b_s = rng.normal(mag_b, mag_b_err, n_samples)
    mag_v_s = rng.normal(mag_v, mag_v_err, n_samples)
    log_rhk_s = rng.normal(log_rhk, log_rhk_err, n_samples)

    mag_bv_s = mag_b_s - mag_v_s

    a = rng.normal(0.407, 0.021, n_samples)
    b = rng.normal(0.325, 0.024, n_samples)
    c = rng.normal(0.495, 0.010, n_samples)
    n = rng.normal(0.566, 0.008, n_samples)

    tau_c_noyes_s = tau_c_noyes1984(mag_bv_s)
    prot_mamajek_s = rotation_period_mamajek2008(log_rhk_s, tau_c_noyes_s)
    prot_noyes_s = rotation_period_noyes1984(log_rhk_s, tau_c_noyes_s)
    age_mamajek_s = gyro_age_mamajek2008(prot_mamajek_s, mag_bv_s, a=a, b=b, c=c, n=n)

    age_chromospheric_s = age_mamajek2008(log_rhk_s, jitter=rng.normal(0, 1, n_samples))

    table = QTable(
        [
            log_rhk_s,
            mag_bv_s,
            tau_c_noyes_s,
            prot_mamajek_s,
            prot_noyes_s,
            age_mamajek_s,
            age_chromospheric_s,
        ],
        names=[
            "log_rhk",
            "mag_bv",
            "tau_c_noyes",
            "prot_mamajek",
            "prot_noyes",
            "age_mamajek_gyro",
            "age_mamajek_chromo",
        ],
    )
    table[
        "prot_mamajek"
    ].description = "Rotation period via Rossby number (Mamajek & Hillenbrand 2008, Eqs. 5 & 7)"
    table["prot_noyes"].description = "Rotation period via Noyes et al. (1984), Eq. 3"
    table[
        "age_mamajek_gyro"
    ].description = "Gyrochronological age (Mamajek & Hillenbrand 2008, Eqs. 12-14)"
    table[
        "age_mamajek_chromo"
    ].description = "Chromospheric age (Mamajek & Hillenbrand 2008, Eq. 3 with jitter)"

    return table
