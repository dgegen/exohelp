import astropy.units as u
import numpy as np
import pytest

from exohelp.planet.escape import (
    StellarXUVTrack,
    default_rocky_core_radius,
    energy_limited_mass_loss_rate,
    escape_velocity,
    jeans_parameter,
    lopez_fortney_fraction_lost,
    lopez_fortney_threshold_flux,
    photoevaporation_evolution,
    photoevaporation_static,
    recombination_limited_mass_loss_rate,
    roche_lobe_correction_factor,
    salz_efficiency,
)


def test_escape_velocity():
    # Earth surface escape velocity: ~11.19 km/s
    v_earth = escape_velocity(1.0, 1.0)
    assert np.isclose(v_earth.to(u.km / u.s).value, 11.186, atol=0.01)

    # Unit handling
    v_jup = escape_velocity(1.0 * u.M_jup, 1.0 * u.R_jup)
    assert np.isclose(v_jup.to(u.km / u.s).value, 59.5, atol=0.5)

    # Array handling
    v_arr = escape_velocity([1.0, 317.8], [1.0, 11.2])
    assert len(v_arr) == 2


def test_jeans_parameter():
    # Earth: M=1 M_earth, R=1 R_earth, T=288 K, air (mu=28.97)
    lam_earth_air = jeans_parameter(1.0, 1.0, 288.0, mean_molecular_weight=28.97)
    assert np.isclose(lam_earth_air, 761.6, atol=1.0)

    # HD 81466 b: M=17.6 M_earth, R=2.655 R_earth, T=759 K, atomic H (mu=1.0)
    lam_hd = jeans_parameter(17.6, 2.655, 759.0, mean_molecular_weight=1.0)
    assert np.isclose(lam_hd, 66.2, atol=0.5)

    # H2 (mu=2.0)
    lam_hd_h2 = jeans_parameter(17.6, 2.655, 759.0, mean_molecular_weight=2.0)
    assert np.isclose(lam_hd_h2, 2.0 * lam_hd, atol=0.1)


def test_roche_lobe_correction_factor():
    # Far orbit -> xi >> 1 -> K_tide -> 1.0
    k_far = roche_lobe_correction_factor(10.0, 1.0, 1.0, 1.0)
    assert np.isclose(k_far, 1.0, atol=0.01)

    # HD 81466 b setup: a=0.1759 AU, M_p=17.6 M_earth, M_*=1.045 M_sun, R_xuv = 1.15 * 2.655 R_earth
    r_xuv = 1.15 * 2.655
    k_hd = roche_lobe_correction_factor(0.1759, 17.6, 1.045, r_xuv)
    assert np.isclose(k_hd, 0.957, atol=0.01)

    # Overfilling Roche lobe (xi <= 1) returns 0
    k_overflow = roche_lobe_correction_factor(0.001, 1.0, 10.0, 10.0)
    assert k_overflow == 0.0


def test_stellar_xuv_track():
    track = StellarXUVTrack.nominal_solar(l_bol=1.0 * u.L_sun, m_star=1.0 * u.M_sun)
    assert track.f_sat == pytest.approx(10 ** (-3.5))
    assert track.t_sat.to(u.Myr).value == 100.0
    assert track.beta == 1.24

    # Saturation phase (t <= t_sat)
    l_sat = track.luminosity_xuv(10.0 * u.Myr)
    l_expected_sat = (10 ** (-3.5) * 1.0 * u.L_sun).to(u.erg / u.s)
    assert np.isclose(l_sat.value, l_expected_sat.value)

    # Decay phase (t = 1000 Myr = 10 * t_sat)
    l_decay = track.luminosity_xuv(1000.0 * u.Myr)
    expected_decay = (10 ** (-3.5) * 1.0 * u.L_sun * (10.0 ** (-1.24))).to(u.erg / u.s)
    assert np.isclose(l_decay.value, expected_decay.value, rtol=1e-3)

    # Flux at 1 AU circular vs eccentric
    f_circ = track.flux_xuv(100.0 * u.Myr, 1.0 * u.AU, eccentricity=0.0)
    f_ecc = track.flux_xuv(100.0 * u.Myr, 1.0 * u.AU, eccentricity=0.6)
    assert np.isclose((f_ecc / f_circ).decompose().value, 1.0 / np.sqrt(1 - 0.6**2), rtol=1e-4)

    # Presets
    low = StellarXUVTrack.low_activity()
    mod = StellarXUVTrack.moderate_activity()
    high = StellarXUVTrack.high_activity()
    assert low.t_sat < mod.t_sat < high.t_sat


def test_salz_efficiency():
    # Low escape velocity -> higher efficiency (capped at 0.35)
    eta_low_v = salz_efficiency(5.0)
    assert 0.05 < eta_low_v <= 0.35

    # High escape velocity -> lower efficiency
    eta_high_v = salz_efficiency(50.0)
    assert eta_high_v < eta_low_v
    assert eta_high_v >= 0.01


def test_energy_limited_mass_loss_rate():
    # Test formula directly: dM/dt = eta * pi * R_xuv^3 * F_xuv / (G * M_p * K_tide)
    rate = energy_limited_mass_loss_rate(
        flux_xuv=100.0 * u.erg / (u.s * u.cm**2),
        r_planet=2.0 * u.R_earth,
        m_planet=10.0 * u.M_earth,
        eta=0.10,
        r_xuv_factor=1.1,
        k_tide=0.95,
    )
    assert rate.unit.is_equivalent(u.g / u.s)
    assert rate.value > 0


def test_photoevaporation_static_hd81466b():
    # HD 81466 b parameters
    track = StellarXUVTrack.nominal_solar(l_bol=1.711 * u.L_sun, m_star=1.045 * u.M_sun)
    res = photoevaporation_static(
        m_planet=17.6,
        r_planet=2.655,
        semi_major_axis=0.1759,
        track=track,
        age=6.2 * u.Gyr,
        eccentricity=0.390,
        eta=[0.08, 0.10, 0.12, 0.15],
        r_xuv_factor=1.15,
        apply_tidal_correction=True,
    )

    assert len(res) == 4
    assert "mass_lost" in res.colnames
    assert "mass_fraction_lost" in res.colnames

    # For eta=0.10, mass lost should be around 0.017 M_earth (15% of a 0.8% envelope)
    m_lost_010 = res[res["eta"] == 0.10]["mass_lost"][0]
    assert np.isclose(m_lost_010.to(u.M_earth).value, 0.017, atol=0.005)


def test_photoevaporation_evolution():
    track = StellarXUVTrack.nominal_solar(l_bol=1.0 * u.L_sun, m_star=1.0 * u.M_sun)
    table = photoevaporation_evolution(
        m_planet_init=5.0 * u.M_earth,
        r_planet_init=2.0 * u.R_earth,
        semi_major_axis=0.05 * u.AU,
        track=track,
        m_env_init=0.05 * u.M_earth,  # 1% envelope
        m_core=4.95 * u.M_earth,
        age=1.0 * u.Gyr,
        eta=0.10,
    )

    assert "time" in table.colnames
    assert "m_planet" in table.colnames
    assert "m_env" in table.colnames
    assert "m_lost" in table.colnames

    # Check that m_lost is monotonic non-decreasing
    m_lost_vals = table["m_lost"].to(u.M_earth).value
    assert np.all(np.diff(m_lost_vals) >= -1e-10)

    # Check envelope is depleted or reduced
    assert table["m_env"][-1] <= table["m_env"][0]
    assert table.meta["solver_success"] is True


def test_photoevaporation_evolution_callable_eta_and_radius():
    track = StellarXUVTrack.low_activity()

    def custom_radius(t, m_tot, m_env):
        # Contraction mock
        return (2.0 * (m_tot.value / 5.0) ** (1 / 3)) * u.R_earth

    table = photoevaporation_evolution(
        m_planet_init=5.0,
        r_planet_init=2.0,
        semi_major_axis=0.1,
        track=track,
        m_env_init=0.2,
        age=0.5 * u.Gyr,
        eta=salz_efficiency,
        radius_func=custom_radius,
    )
    assert len(table) > 2
    assert table["radius"][0].unit.is_equivalent(u.R_earth)


def test_lopez_fortney_scaling():
    # Threshold flux for 5 M_earth core: F_th = 0.5 * 5^2.4 * (0.1/0.1)^-0.7 = 0.5 * 47.59 = 23.8 F_earth
    f_th = lopez_fortney_threshold_flux(5.0, eta=0.10)
    assert np.isclose(f_th.value, 23.8, atol=0.2)

    # Young system age elevation (140 Myr)
    f_th_young = lopez_fortney_threshold_flux(5.0, eta=0.10, age=140.0 * u.Myr)
    assert np.isclose(f_th_young.value, 23.8 + 3.4, atol=0.2)

    # Fraction lost at F = 10 F_earth: f_lost = 0.5 * (10 / 23.8)^1.1 = 0.19
    f_lost = lopez_fortney_fraction_lost(10.0, 5.0, eta=0.10)
    assert np.isclose(f_lost, 0.193, atol=0.01)

    # Unit input handling with W/m^2
    f_lost_unit = lopez_fortney_fraction_lost((10.0 * 1361.0) * u.W / u.m**2, 5.0, eta=0.10)
    assert np.isclose(f_lost_unit, f_lost, atol=1e-4)

    # Fraction lost at high flux clamped to 1.0
    f_lost_high = lopez_fortney_fraction_lost(1000.0, 1.0)
    assert f_lost_high == 1.0


def test_default_rocky_core_radius():
    r_1m = default_rocky_core_radius(1.0)
    assert np.isclose(r_1m.to(u.R_earth).value, 1.0, atol=1e-3)

    r_5m = default_rocky_core_radius(5.0 * u.M_earth)
    assert np.isclose(r_5m.to(u.R_earth).value, 5.0**0.27, atol=1e-3)


def test_recombination_limited_mass_loss_rate():
    rate_fid = recombination_limited_mass_loss_rate(5.0e5 * u.erg / (u.s * u.cm**2))
    assert np.isclose(rate_fid.to(u.g / u.s).value, 4.0e12, rtol=1e-3)

    # Scale with flux
    rate_half = recombination_limited_mass_loss_rate(2.5e5 * u.erg / (u.s * u.cm**2), exponent=0.6)
    assert np.isclose(rate_half.value, 4.0e12 * (0.5**0.6), rtol=1e-3)
