import numpy as np
from astropy.units.cds import ppm  # type: ignore[import]

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
    assert "planet_star_flux_ratio_mid_ir_b" in table.colnames

    latex_names = table.meta["latex_names"]  # type: ignore[assignment]
    assert latex_names["transit_depth_b"] == r"\delta_{b}"
    assert latex_names["surface_gravity_b"] == r"g_{b}"
    assert latex_names["log_surface_gravity_b"] == r"\log g_{b}"
    assert (
        latex_names["planet_star_flux_ratio_mid_ir_b"] == r"(F_{p}/F_{\star})_{\lambda=7.5\mu m, b}"
    )


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


def test_mid_ir_flux_ratio_is_dimensionless_and_finite():
    table = derived_planet_quantities(
        period=3.0,
        r_planet=2.0,
        r_star=0.8,
        m_star=0.8,
        teff_star=4500,
        m_planet=8.0,
    )

    assert "planet_star_flux_ratio_mid_ir" in table.colnames
    assert table["planet_star_flux_ratio_mid_ir"].unit is None  # type: ignore[assignment]
    assert np.isfinite(table["planet_star_flux_ratio_mid_ir"][0])  # type: ignore[index]


def test_transmission_signal_has_ppm_unit():
    table = derived_planet_quantities(
        period=3.0,
        r_planet=2.0,
        r_star=0.8,
        m_star=0.8,
        teff_star=4500,
        m_planet=8.0,
    )

    assert "transmission_signal_1H" in table.colnames
    assert table["transmission_signal_1H"].unit == ppm  # type: ignore[comparison]
    assert np.isfinite(table["transmission_signal_1H"].value[0])  # type: ignore[index]
