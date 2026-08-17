import astropy.units as u
import numpy as np

from exohelp import truncated_normal
from exohelp.star.activity import sample_rotation_period_and_age
from exohelp.star.spectroscopy import (
    sample_rotation_period_from_vsini,
    sample_uvw_lsr,
    sample_v_mic_and_v_mac,
)
from exohelp.star.summary import derive_stellar_parameters


def test_truncated_normal_basic():
    samples = truncated_normal(mean=5.0, std=2.0, size=1000, lower=0.0, rng=42)
    assert isinstance(samples, np.ndarray)
    assert len(samples) == 1000
    assert np.all(samples >= 0.0)


def test_truncated_normal_two_sided():
    samples = truncated_normal(mean=0.0, std=10.0, size=2000, lower=-1.0, upper=1.0, rng=123)
    assert np.all(samples >= -1.0)
    assert np.all(samples <= 1.0)


def test_truncated_normal_shape_tuple():
    samples = truncated_normal(mean=2.0, std=1.0, size=(10, 20), lower=0.0, rng=42)
    assert samples.shape == (10, 20)
    assert np.all(samples >= 0.0)


def test_truncated_normal_zero_std():
    samples = truncated_normal(mean=7.5, std=0.0, size=50, lower=0.0)
    assert np.all(samples == 7.5)


def test_truncated_normal_quantity():
    mean = 5000.0 * u.K
    std = 200.0 * u.K
    lower = 0.0 * u.K
    samples = truncated_normal(mean=mean, std=std, size=500, lower=lower, rng=99)
    assert isinstance(samples, u.Quantity)
    assert samples.unit == u.K
    assert np.all(samples.value >= 0.0)


def test_truncated_normal_quantity_conversion():
    mean = 1.0 * u.km / u.s
    std = 500.0 * u.m / u.s
    lower = 0.0 * u.m / u.s
    samples = truncated_normal(mean=mean, std=std, size=500, lower=lower, rng=99)
    assert isinstance(samples, u.Quantity)
    assert samples.unit == (u.km / u.s)
    assert np.all(samples.value >= 0.0)


def test_truncated_normal_reproducibility():
    rng1 = np.random.default_rng(42)
    s1 = truncated_normal(mean=1.0, std=1.0, size=100, lower=0.0, rng=rng1)
    rng2 = np.random.default_rng(42)
    s2 = truncated_normal(mean=1.0, std=1.0, size=100, lower=0.0, rng=rng2)
    assert np.allclose(s1, s2)


def test_spectroscopy_sample_positivity():
    # Test sampling when mean is close to zero and error is large
    table = sample_v_mic_and_v_mac(
        teff=500.0,
        teff_err=1000.0,
        logg=4.4,
        logg_err=0.1,
        n_samples=500,
        seed=42,
    )
    assert np.all(table["teff"] >= 0.0)

    rot_table = sample_rotation_period_from_vsini(
        vsini=1.0,
        vsini_err=5.0,
        r_star=0.5,
        r_star_err=2.0,
        n_samples=500,
        seed=42,
    )
    assert np.all(rot_table["vsini"] >= 0.0)
    assert np.all(rot_table["r_star"] >= 0.0)


def test_uvw_lsr_distance_and_dec_bounds():
    table = sample_uvw_lsr(
        ra=10.0,
        ra_err=0.1,
        dec=85.0,
        dec_err=20.0,
        distance=5.0,
        distance_err=20.0,
        pm_ra_cosdec=10.0,
        pm_ra_cosdec_err=1.0,
        pm_dec=-10.0,
        pm_dec_err=1.0,
        radial_velocity=5.0,
        radial_velocity_err=1.0,
        n_samples=500,
        seed=42,
    )
    assert np.all(table["distance"] >= 0.0 * u.pc)
    assert np.all(table["dec"] >= -90.0 * u.deg)
    assert np.all(table["dec"] <= 90.0 * u.deg)


def test_activity_sampling_positivity():
    table = sample_rotation_period_and_age(
        log_rhk=-5.0,
        log_rhk_err=0.1,
        mag_b=10.0,
        mag_b_err=0.05,
        mag_v=9.5,
        mag_v_err=0.05,
        n_samples=500,
        seed=42,
    )
    # Check that rotation period and age columns are non-negative where valid
    for col in ["prot_mamajek", "prot_noyes", "age_gyro_mamajek", "age_gyro_barnes"]:
        valid = ~np.isnan(table[col].value)
        if np.any(valid):
            assert np.all(table[col][valid].value >= 0.0)


def test_derive_stellar_parameters_positivity():
    table = derive_stellar_parameters(
        teff=5000.0,
        teff_err=1000.0,
        logg=4.4,
        logg_err=0.1,
        mass=1.0,
        mass_err=2.0,
        radius=1.0,
        radius_err=2.0,
        vsini=2.0,
        vsini_err=5.0,
        n_samples=500,
        seed=42,
    )
    assert np.all(table["teff"] >= 0.0 * u.K)
    assert np.all(table["mass"] >= 0.0 * u.M_sun)
    assert np.all(table["radius"] >= 0.0 * u.R_sun)
    assert np.all(table["vsini"] >= 0.0 * u.km / u.s)
