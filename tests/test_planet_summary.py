import numpy as np

from exohelp.planet.summary import derived_planet_quantities


def test_planet_index_suffix_and_latex_names():
    table = derived_planet_quantities(
        period=3.0,
        r_planet=2.0,
        r_star=0.8,
        m_star=0.8,
        teff_star=4500,
        m_planet=8.0,
        planet_index="b",
    )

    assert "transit_depth_b" in table.colnames
    assert "surface_gravity_b" in table.colnames
    assert "log_surface_gravity_b" in table.colnames

    latex_names = table.meta["latex_names"]
    assert latex_names["transit_depth_b"] == r"\delta_{b}"
    assert latex_names["surface_gravity_b"] == r"g_{b}"
    assert latex_names["log_surface_gravity_b"] == r"\log g_{b}"


def test_log_surface_gravity_is_dimensionless():
    table = derived_planet_quantities(
        period=3.0,
        r_planet=2.0,
        r_star=0.8,
        m_star=0.8,
        m_planet=8.0,
    )

    assert "log_surface_gravity" in table.colnames
    assert table["log_surface_gravity"].unit is None
    assert np.isfinite(table["log_surface_gravity"][0])
