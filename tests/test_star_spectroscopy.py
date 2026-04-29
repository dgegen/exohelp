import numpy as np
import astropy.units as u
from astropy.coordinates import LSR, SkyCoord
from typing import Any, cast

from exohelp.star.spectroscopy import (
    bensby_membership_probabilities,
    classify_td_to_d_ratio,
    sample_uvw_lsr,
)


def test_sample_uvw_lsr_columns_and_length():
    km_per_s = u.Unit("km / s")

    table = sample_uvw_lsr(
        ra=120.5,
        ra_err=0.001,
        dec=-45.2,
        dec_err=0.001,
        distance=50.0,
        distance_err=0.1,
        pm_ra_cosdec=120.0,
        pm_ra_cosdec_err=0.2,
        pm_dec=-80.0,
        pm_dec_err=0.2,
        radial_velocity=12.0,
        radial_velocity_err=0.1,
        n_samples=256,
        seed=123,
    )

    assert len(table) == 256
    assert set(table.colnames) == {
        "ra",
        "dec",
        "distance",
        "pm_ra_cosdec",
        "pm_dec",
        "radial_velocity",
        "U_lsr",
        "V_lsr",
        "W_lsr",
    }

    u_col = cast(Any, table["U_lsr"])
    v_col = cast(Any, table["V_lsr"])
    w_col = cast(Any, table["W_lsr"])

    assert u_col.unit == km_per_s
    assert v_col.unit == km_per_s
    assert w_col.unit == km_per_s


def test_sample_uvw_lsr_reproducible_with_seed():
    table_a = sample_uvw_lsr(
        ra=120.5,
        ra_err=0.001,
        dec=-45.2,
        dec_err=0.001,
        distance=50.0,
        distance_err=0.1,
        pm_ra_cosdec=120.0,
        pm_ra_cosdec_err=0.2,
        pm_dec=-80.0,
        pm_dec_err=0.2,
        radial_velocity=12.0,
        radial_velocity_err=0.1,
        n_samples=128,
        seed=77,
    )
    table_b = sample_uvw_lsr(
        ra=120.5,
        ra_err=0.001,
        dec=-45.2,
        dec_err=0.001,
        distance=50.0,
        distance_err=0.1,
        pm_ra_cosdec=120.0,
        pm_ra_cosdec_err=0.2,
        pm_dec=-80.0,
        pm_dec_err=0.2,
        radial_velocity=12.0,
        radial_velocity_err=0.1,
        n_samples=128,
        seed=77,
    )

    u_a = cast(Any, table_a["U_lsr"])
    v_a = cast(Any, table_a["V_lsr"])
    w_a = cast(Any, table_a["W_lsr"])
    u_b = cast(Any, table_b["U_lsr"])
    v_b = cast(Any, table_b["V_lsr"])
    w_b = cast(Any, table_b["W_lsr"])

    assert np.allclose(u_a.value, u_b.value)
    assert np.allclose(v_a.value, v_b.value)
    assert np.allclose(w_a.value, w_b.value)


def test_sample_uvw_lsr_zero_uncertainty_matches_direct_transform():
    deg = u.Unit("deg")
    pc = u.Unit("pc")
    mas_per_yr = u.Unit("mas / yr")
    km_per_s = u.Unit("km / s")

    table = sample_uvw_lsr(
        ra=120.5,
        ra_err=0.0,
        dec=-45.2,
        dec_err=0.0,
        distance=50.0,
        distance_err=0.0,
        pm_ra_cosdec=120.0,
        pm_ra_cosdec_err=0.0,
        pm_dec=-80.0,
        pm_dec_err=0.0,
        radial_velocity=12.0,
        radial_velocity_err=0.0,
        n_samples=32,
        seed=9,
    )

    direct = SkyCoord(
        ra=120.5 * deg,
        dec=-45.2 * deg,
        distance=50.0 * pc,
        pm_ra_cosdec=120.0 * mas_per_yr,
        pm_dec=-80.0 * mas_per_yr,
        radial_velocity=12.0 * km_per_s,
        frame="icrs",
    ).transform_to(LSR())

    velocity = cast(Any, direct.velocity)
    if velocity is None:
        raise ValueError("Velocity information is required for this test.")

    u_expected = velocity.d_x.to_value(km_per_s)
    v_expected = velocity.d_y.to_value(km_per_s)
    w_expected = velocity.d_z.to_value(km_per_s)

    u_col = cast(Any, table["U_lsr"])
    v_col = cast(Any, table["V_lsr"])
    w_col = cast(Any, table["W_lsr"])

    assert np.allclose(u_col.value, u_expected)
    assert np.allclose(v_col.value, v_expected)
    assert np.allclose(w_col.value, w_expected)


def test_bensby_membership_probabilities_columns_and_length():
    probs = bensby_membership_probabilities(
        u_lsr=np.array([5.0, -60.0]),
        v_lsr=np.array([-10.0, -55.0]),
        w_lsr=np.array([2.0, 20.0]),
    )

    assert len(probs) == 2
    assert set(probs.colnames) == {
        "Thin Disk",
        "Thick Disk",
        "Halo",
        "Hercules",
        "TD_to_D",
    }


def test_bensby_td_to_d_tracks_expected_kinematic_regime():
    thin_like = bensby_membership_probabilities(u_lsr=5.0, v_lsr=-10.0, w_lsr=2.0)
    thick_like = bensby_membership_probabilities(u_lsr=70.0, v_lsr=-80.0, w_lsr=40.0)

    thin_ratio = cast(Any, thin_like["TD_to_D"])[0]
    thick_ratio = cast(Any, thick_like["TD_to_D"])[0]

    assert thin_ratio < 0.5
    assert thick_ratio > 2.0


def test_classify_td_to_d_ratio_thresholds():
    labels = classify_td_to_d_ratio(np.array([0.2, 1.2, 4.0]))
    assert list(labels) == ["Thin Disk", "In-between", "Thick Disk"]
