import astropy.units as u
import numpy as np

from exohelp import keplers_third_law, solve_kepler


def test_earth_period():
    """Earth at 1 AU around 1 solar mass should give ~1 year."""
    period = keplers_third_law(semi_major_axis=1.0 * u.AU, mass=1.0 * u.Msun)
    assert np.isclose(period.to(u.yr).value, 1.0, rtol=1e-3)


def test_solve_kepler_scalar():
    """Verify Kepler equation solution M = E - e*sin(E)."""
    mean_anom = 0.5
    e = 0.3
    ecc_anom = solve_kepler(mean_anom, e)
    # Check that E - e*sin(E) == M
    assert np.isclose(ecc_anom - e * np.sin(ecc_anom), mean_anom, atol=1e-10)


def test_solve_kepler_quantity():
    """Verify solve_kepler preserves Quantity units."""
    mean_anom = 45.0 * u.deg
    e = 0.5
    ecc_anom = solve_kepler(mean_anom, e)
    assert isinstance(ecc_anom, u.Quantity)
    assert ecc_anom.unit == u.rad
    e_val = ecc_anom.to_value(u.rad)
    m_val = mean_anom.to_value(u.rad)
    assert np.isclose(e_val - e * np.sin(e_val), m_val, atol=1e-10)


def test_solve_kepler_vectorized():
    """Verify vectorized solution across array of anomalies and eccentricities."""
    mean_anom = np.linspace(0, 2 * np.pi, 50)
    e = 0.8
    ecc_anom = solve_kepler(mean_anom, e)
    assert np.allclose(ecc_anom - e * np.sin(ecc_anom), mean_anom, atol=1e-10)
