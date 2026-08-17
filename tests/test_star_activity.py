import astropy.units as u
import numpy as np
import pytest

from exohelp.star.activity import (
    SUAREZ_MASCARENO_2015_COEFFICIENTS,
    rotation_period_from_rhk_suarez_mascareno_2015,
    rotation_period_suarez_mascareno2015,
)


def test_suarez_mascareno_basic():
    # log_rhk = -4.5 (full sample)
    prot, err = rotation_period_suarez_mascareno2015(-4.5)
    assert isinstance(prot, u.Quantity)
    assert isinstance(err, u.Quantity)
    assert prot.unit == u.day
    assert err.unit == u.day
    # 10**(-0.808 * -4.5 - 2.536) = 10**1.1 = 12.589254...
    assert np.isclose(prot.value, 12.589, atol=0.01)
    assert err.value > 0.0


def test_suarez_mascareno_alias():
    prot1, err1 = rotation_period_suarez_mascareno2015(-4.8)
    prot2, err2 = rotation_period_from_rhk_suarez_mascareno_2015(-4.8)
    assert np.allclose(prot1.value, prot2.value)
    assert np.allclose(err1.value, err2.value)


def test_suarez_mascareno_all_samples():
    for s in SUAREZ_MASCARENO_2015_COEFFICIENTS:
        prot, err = rotation_period_suarez_mascareno2015(-4.7, sample=s)
        assert prot.unit == u.day
        assert err.unit == u.day
        assert prot.value > 0.0
        assert err.value > 0.0


def test_suarez_mascareno_invalid_sample():
    with pytest.raises(ValueError, match="Unknown sample"):
        rotation_period_suarez_mascareno2015(-4.5, sample="invalid_sample")


def test_suarez_mascareno_return_components():
    prot, fit_err, scatter = rotation_period_suarez_mascareno2015(
        -4.5, sample="full", return_components=True
    )
    assert isinstance(prot, u.Quantity)
    assert isinstance(fit_err, u.Quantity)
    assert isinstance(scatter, u.Quantity)
    assert np.isclose(scatter.value, 0.17 * prot.value)

    # Total error hypot check
    _, total_err = rotation_period_suarez_mascareno2015(
        -4.5, sample="full", return_components=False
    )
    assert np.isclose(total_err.value, np.hypot(fit_err.value, scatter.value))


def test_suarez_mascareno_array_input():
    log_rhk_arr = np.array([-4.4, -4.7, -5.0])
    prot_arr, err_arr = rotation_period_suarez_mascareno2015(log_rhk_arr)
    assert len(prot_arr) == 3
    assert len(err_arr) == 3
    # Prot should increase as activity decreases (more negative log_rhk)
    assert prot_arr[0] < prot_arr[1] < prot_arr[2]


def test_sample_rotation_period_and_age_all_suarez_mascareno_subsets():
    from exohelp.star.activity import sample_rotation_period_and_age

    table = sample_rotation_period_and_age(
        log_rhk=-5.04,
        log_rhk_err=0.09,
        mag_b=9.941,
        mag_b_err=0.029,
        mag_v=9.33,
        mag_v_err=0.023,
        n_samples=500,
        seed=42,
    )
    for s in SUAREZ_MASCARENO_2015_COEFFICIENTS:
        col_name = f"prot_suarez_mascareno_{s}"
        assert col_name in table.colnames
        assert table[col_name].unit == u.day
        assert np.all(table[col_name].value > 0.0)

    assert "prot_suarez_mascareno" in table.colnames


@pytest.mark.parametrize(
    "log_rhk, category, expected_prot, expected_err",
    [
        (-4.90, "full", 26.4972, 5.7847),
        (-4.30, "full", 8.6776, 1.8087),
        (-4.91, "solar_feh", 28.0821, 7.2068),
        (-5.10, "m_dwarfs", 41.8119, 47.0017),
    ],
)
def test_predict_prot_benchmarks(log_rhk, category, expected_prot, expected_err):
    prot, err = rotation_period_suarez_mascareno2015(log_rhk, category=category)
    np.testing.assert_allclose(prot.value, expected_prot, rtol=1e-3)
    np.testing.assert_allclose(err.value, expected_err, rtol=1e-3)


def test_array_input_support():
    log_rhk_arr = np.array([-4.90, -4.30])
    prot, err = rotation_period_suarez_mascareno2015(log_rhk_arr, category="full")
    assert prot.shape == (2,)
    assert err.shape == (2,)
    np.testing.assert_allclose(prot[0].value, 26.4972, rtol=1e-3)
