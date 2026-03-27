import astropy.units as u
import numpy as np

from exohelp import keplers_third_law


def test_earth_period():
    """Earth at 1 AU around 1 solar mass should give ~1 year."""
    period = keplers_third_law(semi_major_axis=1.0 * u.AU, mass=1.0 * u.Msun)
    assert np.isclose(period.to(u.yr).value, 1.0, rtol=1e-3)
