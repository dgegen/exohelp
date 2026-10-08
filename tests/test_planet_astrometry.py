import astropy.units as u
import numpy as np

from astropy.table import QTable

from exohelp.planet import (
    BinaryAstrometricContamination,
    along_scan_displacement,
    astrometric_detection_thresholds,
    astrometric_orbit,
    astrometric_semi_amplitude,
    binary_astrometric_contamination,
    delta_chi2_to_sigma,
    expected_astrometric_excess_noise,
    expected_ruwe,
    gaia_astrometric_detectability,
    gaia_scan_events,
    planet_mass_from_astrometry,
    thiele_innes_constants,
)


def test_astrometric_semi_amplitude():
    """Jupiter at 1 AU around 1 M_sun at 10 pc."""
    # alpha = (m_p / (M_* + m_p)) * (a / d)
    # m_p ~ 0.000954588 M_sun, M_* = 1 M_sun -> a_* = 1.0 * (0.000954588 / 1.000954588) AU = 0.000953678 AU
    # at 10 pc -> alpha = 0.000953678 / 10 arcsec = 95.3678 uarcsec
    alpha = astrometric_semi_amplitude(
        m_planet=1.0 * u.M_jup,
        m_star=1.0 * u.M_sun,
        distance=10.0 * u.pc,
        semi_major_axis=1.0 * u.AU,
    )
    assert np.isclose(alpha.to_value("uarcsec"), 95.368, atol=1e-2)


def test_astrometric_semi_amplitude_with_parallax_and_period():
    """Verify input handling with parallax and period."""
    alpha = astrometric_semi_amplitude(
        m_planet=1.0,  # assumed M_jup
        m_star=1.0,  # assumed M_sun
        parallax=100.0,  # 100 mas = 10 pc
        period=365.25,  # ~1 AU
    )
    assert np.isclose(alpha.to_value("uarcsec"), 95.37, atol=1e-1)


def test_planet_mass_from_astrometry_roundtrip():
    """Verify round-trip between alpha and mass."""
    m_p_true = 3.5 * u.M_jup
    m_star = 0.95 * u.M_sun
    dist = 50.0 * u.pc
    a = 2.0 * u.AU

    alpha = astrometric_semi_amplitude(
        m_planet=m_p_true,
        m_star=m_star,
        distance=dist,
        semi_major_axis=a,
    )

    m_derived = planet_mass_from_astrometry(
        alpha=alpha,
        m_star=m_star,
        distance=dist,
        semi_major_axis=a,
    )
    assert np.isclose(m_derived.to_value("M_jup"), m_p_true.to_value("M_jup"), rtol=1e-5)


def test_planet_mass_from_astrometry_with_period():
    """Verify round-trip mass inversion when period is provided (exact Kepler's third law)."""
    # 1. Standard planet
    m_p = 5.0 * u.M_jup
    alpha = astrometric_semi_amplitude(m_planet=m_p, m_star=1.0, distance=20.0, period=200.0)
    m_rec = planet_mass_from_astrometry(alpha=alpha, m_star=1.0, distance=20.0, period=200.0)
    assert np.isclose(m_rec.to_value("M_jup"), m_p.to_value("M_jup"), rtol=1e-4)

    # 2. Massive companion (100 M_jup ~ 0.1 M_sun)
    m_massive = 100.0 * u.M_jup
    alpha_m = astrometric_semi_amplitude(
        m_planet=m_massive, m_star=1.0, distance=10.0, period=365.25
    )
    m_rec_m = planet_mass_from_astrometry(alpha=alpha_m, m_star=1.0, distance=10.0, period=365.25)
    assert np.isclose(m_rec_m.to_value("M_jup"), m_massive.to_value("M_jup"), rtol=1e-4)


def test_planet_mass_from_astrometry_unphysical():
    """Verify that unphysical astrometric amplitude (a_* >= a) returns a masked quantity."""
    # a_* = alpha * d = 2.0 AU, but orbital semi-major axis is only 1.0 AU
    m_inv = planet_mass_from_astrometry(
        alpha=2.0 * 1e6,  # 2 arcsec at 1 pc = 2 AU
        m_star=1.0,
        distance=1.0,
        semi_major_axis=1.0,  # a < a_*
    )
    assert m_inv.mask


def test_astrometric_photocenter_reduction():
    """Verify luminous companion reduces photocentric wobble."""
    alpha_dark = astrometric_semi_amplitude(
        m_planet=100.0, m_star=1.0, distance=10.0, semi_major_axis=1.0, flux_ratio=0.0
    )
    alpha_luminous = astrometric_semi_amplitude(
        m_planet=100.0, m_star=1.0, distance=10.0, semi_major_axis=1.0, flux_ratio=0.01
    )
    assert alpha_luminous < alpha_dark


def test_thiele_innes_constants():
    """Verify Thiele-Innes constants for face-on and edge-on cases."""
    a_star = 100.0 * u.uarcsec
    # Face-on, omega=0, Omega=0: cos i = 1
    # A = a_*, B = 0, F = 0, G = a_*
    ti_face = thiele_innes_constants(a_star=a_star, inclination=0.0, omega=0.0, node=0.0)
    assert np.isclose(ti_face.A.to_value("uarcsec"), 100.0)
    assert np.isclose(ti_face.B.to_value("uarcsec"), 0.0)
    assert np.isclose(ti_face.F.to_value("uarcsec"), 0.0)
    assert np.isclose(ti_face.G.to_value("uarcsec"), 100.0)

    # Edge-on, omega=0, Omega=0: cos i = 0
    # A = a_*, B = 0, F = 0, G = 0
    ti_edge = thiele_innes_constants(a_star=a_star, inclination=90.0, omega=0.0, node=0.0)
    assert np.isclose(ti_edge.A.to_value("uarcsec"), 100.0)
    assert np.isclose(ti_edge.B.to_value("uarcsec"), 0.0)
    assert np.isclose(ti_edge.F.to_value("uarcsec"), 0.0)
    assert np.isclose(ti_edge.G.to_value("uarcsec"), 0.0)


def test_astrometric_orbit():
    """Verify projected orbit displacements and alias."""
    times = np.array([0.0, 91.3125, 182.625, 273.9375])  # 4 quarters of a 365.25-day orbit
    proj = astrometric_orbit(
        times=times,
        period=365.25,
        m_planet=1.0,
        m_star=1.0,
        distance=10.0,
        inclination=0.0,  # face-on circular
        eccentricity=0.0,
    )
    # Displacement radius should be constant ~ 95.40 uarcsec
    assert np.allclose(proj.separation.to_value("uarcsec"), 95.40, atol=1e-2)
    assert len(proj.position_angle) == 4
    # Along-scan projection method
    d_al_from_method = proj.along_scan(0.0)
    assert np.allclose(d_al_from_method.to_value("uarcsec"), proj.d_dec.to_value("uarcsec"))


def test_along_scan_displacement():
    """Verify 1D projection onto scan direction."""
    d_ra = 100.0 * u.uarcsec
    d_dec = 0.0 * u.uarcsec
    psi = np.pi / 2  # scan along RA -> d_AL = d_RA
    d_al = along_scan_displacement(d_ra, d_dec, psi)
    assert np.isclose(d_al.to_value("uarcsec"), 100.0)

    psi_dec = 0.0  # scan along Dec -> d_AL = d_Dec = 0
    d_al_dec = along_scan_displacement(d_ra, d_dec, psi_dec)
    assert np.isclose(d_al_dec.to_value("uarcsec"), 0.0)


def test_thiele_innes_project():
    """Verify Thiele-Innes project method."""
    ti = thiele_innes_constants(a_star=100.0, inclination=90.0, omega=0.0, node=0.0)
    proj = ti.project(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert isinstance(proj.d_ra, u.Quantity)
    assert isinstance(proj.d_dec, u.Quantity)
    assert len(proj.d_ra) == 2


def test_gaia_scan_events():
    """Verify Gaia scan event generation, methods, and epoch_start."""
    ra = 125.18875
    dec = 46.20360
    events = gaia_scan_events(ra=ra, dec=dec, duration_years=5.5)
    assert events.n_transits > 0
    assert len(events.times) == events.n_transits
    assert events.time_baseline.unit == u.day
    assert events.time_baseline > 0 * u.day

    # Project method on GaiaScanEvents
    d_al = events.project(
        events.times.value * 0.0 * u.uarcsec, events.times.value * 0.0 * u.uarcsec
    )
    assert len(d_al) == events.n_transits

    # Epoch start shift
    epoch_ref = 2456860.5
    events_shifted = gaia_scan_events(ra=ra, dec=dec, duration_years=5.5, epoch_start=epoch_ref)
    assert np.isclose(
        events_shifted.times[0].to_value("day") - events.times[0].to_value("day"), epoch_ref
    )

    # Verify scan angle depends on target sky coordinates (not pure time function)
    events_other = gaia_scan_events(ra=10.0, dec=-30.0, duration_years=5.5)
    assert not np.allclose(events.scan_angles[0].value, events_other.scan_angles[0].value)
    assert np.all(events.scan_angles.to_value("rad") >= 0.0)
    assert np.all(events.scan_angles.to_value("rad") < 2 * np.pi)


def test_expected_ruwe_and_excess_noise():
    """Verify analytical RUWE and excess noise calculations."""
    # Under null hypothesis (delta_chi2 = 0): RUWE = 1.0, excess_noise = 0
    assert np.isclose(expected_ruwe(0.0, n_transits=80), 1.0)
    assert np.isclose(
        expected_astrometric_excess_noise(0.0, n_transits=80).to_value("uarcsec"), 0.0
    )

    # With significant signal:
    # Transit level (on_ccd=False): Delta_chi2 = 75 with dof = 75 gives RUWE = sqrt(2) ~ 1.41
    assert expected_ruwe(75.0, n_transits=80, on_ccd=False) > 1.4
    # Official CCD level (on_ccd=True, default): dof = 9 * 80 - 5 = 715; Delta_chi2 = 750 gives RUWE > 1.4
    assert expected_ruwe(750.0, n_transits=80) > 1.4
    excess = expected_astrometric_excess_noise(75.0, n_transits=80, sigma_al=200.0)
    assert excess.to_value("uarcsec") > 0.0


def test_gaia_astrometric_detectability_table():
    """Verify Gaia detectability returns a QTable with grid broadcasting and correct columns."""
    tab = gaia_astrometric_detectability(
        period=109.0,
        m_planet=5.54,
        m_star=0.989,
        distance=206.9,
        inclination=[90.0, 60.0, 45.0],
        node=[0.0, 90.0],
        eccentricity=0.1,
        omega=45.0,
        t0=10.0,
        ra=125.18875,
        dec=46.20360,
        is_msini=True,
    )
    assert isinstance(tab, QTable)
    assert len(tab) == 6  # 3 inclinations x 2 nodes grid

    # Check columns
    expected_cols = {
        "inclination",
        "node",
        "true_mass",
        "alpha_star",
        "delta_chi2",
        "snr",
        "equivalent_sigma",
        "ruwe",
        "excess_noise",
    }
    assert set(tab.colnames) == expected_cols

    # Check units
    assert tab["inclination"].unit == u.deg
    assert tab["node"].unit == u.deg
    assert tab["true_mass"].unit == u.M_jup
    assert tab["alpha_star"].unit == u.uarcsec
    assert tab["excess_noise"].unit == u.uarcsec

    # Check relation between snr and delta_chi2
    assert np.allclose(tab["snr"], np.sqrt(tab["delta_chi2"]))
    assert np.all(tab["equivalent_sigma"] > 0)
    assert np.all(tab["ruwe"] >= 1.0)
    assert np.all(tab["excess_noise"].value >= 0.0)

    # Check metadata and citation provenance
    assert set(tab.meta["references"]) == {
        "Perryman2014",
        "Ranalli2018",
        "Lindegren2021_edr3_astro",
    }
    assert tab["alpha_star"].info.meta["references"] == ["Perryman2014"]
    assert tab["delta_chi2"].info.meta["references"] == ["Perryman2014", "Ranalli2018"]
    assert tab["equivalent_sigma"].info.meta["references"] == ["Perryman2014", "Ranalli2018"]


def test_gaia_astrometric_detectability_marginalized():
    """Verify node=None marginalizes over ascending node Omega."""
    tab = gaia_astrometric_detectability(
        period=109.0,
        m_planet=5.54,
        m_star=0.989,
        distance=206.9,
        inclination=[90.0, 60.0, 30.0],
        eccentricity=0.1,
        omega=45.0,
        t0=5.0,
        ra=125.18875,
        dec=46.20360,
        is_msini=True,
        # node=None is default!
        n_node_samples=24,
    )
    assert isinstance(tab, QTable)
    assert len(tab) == 3  # 1 row per inclination

    # Verify marginalized columns exist
    expected_cols = {
        "inclination",
        "true_mass",
        "alpha_star",
        "delta_chi2",
        "delta_chi2_min",
        "delta_chi2_max",
        "snr",
        "snr_min",
        "snr_max",
        "equivalent_sigma",
        "equivalent_sigma_min",
        "equivalent_sigma_max",
        "ruwe",
        "ruwe_min",
        "ruwe_max",
        "excess_noise",
        "excess_noise_min",
        "excess_noise_max",
    }
    assert set(tab.colnames) == expected_cols

    # Verify min <= median <= max
    assert np.all(tab["delta_chi2_min"] <= tab["delta_chi2"])
    assert np.all(tab["delta_chi2"] <= tab["delta_chi2_max"])
    assert np.all(tab["snr_min"] <= tab["snr"])
    assert np.all(tab["snr"] <= tab["snr_max"])
    assert np.all(tab["equivalent_sigma_min"] <= tab["equivalent_sigma"])
    assert np.all(tab["equivalent_sigma"] <= tab["equivalent_sigma_max"])
    assert np.all(tab["ruwe_min"] <= tab["ruwe"])
    assert np.all(tab["ruwe"] <= tab["ruwe_max"])
    assert np.all(tab["excess_noise_min"] <= tab["excess_noise"])
    assert np.all(tab["excess_noise"] <= tab["excess_noise_max"])


def test_gaia_astrometric_detectability_face_on_singularity():
    """Verify face-on singularity (i=0 or 180) with is_msini=True returns masked arrays."""
    tab = gaia_astrometric_detectability(
        period=109.0,
        m_planet=5.54,
        m_star=0.989,
        distance=206.9,
        inclination=[90.0, 0.0, 180.0],
        ra=125.18875,
        dec=46.20360,
        is_msini=True,
    )
    # Valid row
    assert not tab["true_mass"].mask[0]
    assert not tab["delta_chi2"].mask[0]
    # Singularity rows
    assert tab["true_mass"].mask[1]
    assert tab["true_mass"].mask[2]
    assert tab["delta_chi2"].mask[1]
    assert tab["delta_chi2"].mask[2]
    assert tab["ruwe"].mask[1]
    assert tab["excess_noise"].mask[1]


def test_delta_chi2_to_sigma():
    """Verify conversion from delta_chi2 to equivalent Gaussian sigma."""
    # df=2 (rv mode)
    assert np.isclose(delta_chi2_to_sigma(6.18, mode="rv"), 2.0, atol=0.02)
    assert np.isclose(delta_chi2_to_sigma(11.83, mode="rv"), 3.0, atol=0.02)
    assert np.isclose(delta_chi2_to_sigma(28.74, mode="rv"), 5.0, atol=0.02)

    # df=7 (blind mode, Perryman et al. 2014)
    assert np.isclose(delta_chi2_to_sigma(14.34, mode="blind"), 2.0, atol=0.02)
    assert np.isclose(delta_chi2_to_sigma(21.85, mode="blind"), 3.0, atol=0.02)

    # Ranalli et al. (2018) Table 4 effective degrees of freedom:
    # df=11 (5-yr mission): 3-sigma is Delta chi2 ~ 28.51 (empirical ~28)
    assert np.isclose(delta_chi2_to_sigma(28.51, mode="blind_5yr"), 3.0, atol=0.05)
    # df=16 (10-yr mission): 3-sigma is Delta chi2 ~ 36.22 (empirical ~34)
    assert np.isclose(delta_chi2_to_sigma(36.22, mode="blind_10yr"), 3.0, atol=0.05)

    # Vectorized
    arr = delta_chi2_to_sigma([6.18, 11.83], mode="rv")
    assert len(arr) == 2


def test_astrometric_detection_thresholds():
    """Verify critical thresholds table."""
    thresh_rv = astrometric_detection_thresholds(mode="rv")
    assert isinstance(thresh_rv, QTable)
    assert len(thresh_rv) == 7
    assert thresh_rv.meta["degrees_of_freedom"] == 2
    assert "sigma" in thresh_rv.colnames
    assert "delta_chi2" in thresh_rv.colnames
    assert "snr" in thresh_rv.colnames
    assert "Ranalli Benchmark" in thresh_rv["level"]
    ranalli_row = thresh_rv[thresh_rv["level"] == "Ranalli Benchmark"][0]
    assert np.isclose(ranalli_row["delta_chi2"], 80.0)

    thresh_blind = astrometric_detection_thresholds(mode="blind")
    assert thresh_blind.meta["degrees_of_freedom"] == 7

    # Ranalli 5-yr mission: 11 dof with 3-sigma ~ 28.5 (Ranalli Table 4 empirical: 28)
    thresh_5yr = astrometric_detection_thresholds(mode="blind_5yr")
    assert thresh_5yr.meta["degrees_of_freedom"] == 11
    three_sig_5yr = thresh_5yr[thresh_5yr["level"] == "3-sigma"][0]
    assert np.isclose(three_sig_5yr["delta_chi2"], 28.51, atol=0.1)

    # Ranalli 10-yr mission: 16 dof
    thresh_10yr = astrometric_detection_thresholds(mode="blind_10yr")
    assert thresh_10yr.meta["degrees_of_freedom"] == 16


def test_ranalli_thiele_innes_and_scan_projection():
    """Verify Thiele-Innes constants and along-scan projection against Ranalli et al. (2018).

    Cross-checks Eqs. (5)-(8) for A, B, F, G and Eq. (14) for along-scan projection.
    """
    alpha = 100.0 * u.uarcsec
    # Test face-on circular orbit (i=0, e=0, omega=0, Omega=0)
    ti = thiele_innes_constants(alpha, inclination=0.0, omega=0.0, node=0.0)
    assert np.isclose(ti.A.to_value("uarcsec"), 100.0)
    assert np.isclose(ti.B.to_value("uarcsec"), 0.0)
    assert np.isclose(ti.F.to_value("uarcsec"), 0.0)
    assert np.isclose(ti.G.to_value("uarcsec"), 100.0)

    # Sky projection (Ranalli Eqs. 3-4):
    proj = ti.project(orb_x=1.0, orb_y=0.0)
    assert np.isclose(proj.d_ra.to_value("uarcsec"), 0.0)
    assert np.isclose(proj.d_dec.to_value("uarcsec"), 100.0)

    # Along-scan projection (Ranalli Eq. 14 / standard psi):
    # Scan toward North (psi=0): d_AL = d_Dec = 100 uas
    assert np.isclose(along_scan_displacement(proj.d_ra, proj.d_dec, 0.0 * u.rad).value, 100.0)
    # Scan toward East (psi=pi/2): d_AL = d_RA* = 0 uas
    assert np.isclose(along_scan_displacement(proj.d_ra, proj.d_dec, np.pi / 2 * u.rad).value, 0.0)


def test_binary_astrometric_contamination():
    """Verify visual binary contamination model."""
    # Wide binary (10 arcsec) -> clean
    wide = binary_astrometric_contamination(10.0 * u.arcsec, 5.0)
    assert isinstance(wide, BinaryAstrometricContamination)
    assert wide.excess_noise.to_value("uarcsec") == 0.0
    assert np.isclose(wide.ruwe, 1.0)
    assert wide.contamination_flag == "clean"

    # Close binary TOI-1699A (rho = 1.68 arcsec, delta_G = 5.25 mag)
    toi1699a = binary_astrometric_contamination(1.68 * u.arcsec, 5.25)
    assert toi1699a.contamination_flag == "blended_window_conflict"
    # Should predict excess noise ~ 0.22 mas and RUWE ~ 1.46 - 1.48
    assert np.isclose(toi1699a.excess_noise.to_value("mas"), 0.22, atol=0.05)
    assert np.isclose(toi1699a.ruwe, 1.48, atol=0.08)
    assert toi1699a.max_centroid_shift.to_value("mas") > 10.0

    # Secondary TOI-1699B (evaluating faint star with brighter companion, delta_G = -5.25)
    toi1699b = binary_astrometric_contamination(1.68 * u.arcsec, -5.25, phot_g_mean_mag=16.58)
    assert toi1699b.contamination_flag == "blended_window_conflict"
    assert np.isclose(toi1699b.excess_noise.to_value("mas"), 1.04, atol=0.20)
    assert 1.2 <= toi1699b.ruwe <= 1.6


def test_gaia_detectability_binary_baseline():
    """Verify detectability table when incorporating visual binary baseline."""
    # Run with explicit baseline RUWE and excess noise
    tab = gaia_astrometric_detectability(
        period=108.75,
        m_planet=5.54,
        is_msini=True,
        m_star=0.989,
        distance=206.9,
        inclination=[90.0, 3.0],
        ra=125.18875,
        dec=46.20360,
        baseline_ruwe=1.46,
        baseline_excess_noise=0.22 * u.mas,
    )
    assert "Fabricius2021_edr3_val" in tab.meta["references"]
    assert "ElBadry2021_binaries" in tab.meta["references"]

    # Coplanar (i=90): planetary wobble is small, RUWE remains ~ 1.46
    ruwe_coplanar = tab["ruwe"][0]
    assert np.isclose(ruwe_coplanar, 1.46, atol=0.02)
    assert np.isclose(tab["excess_noise"][0].to_value("mas"), 0.22, atol=0.02)

    # Face-on (i=3): massive ~0.1 M_sun companion elevates RUWE to > 2.0 at CCD level
    ruwe_faceon = tab["ruwe"][1]
    assert ruwe_faceon > 2.0

    # With on_ccd=False (transit level), RUWE is > 4.5
    tab_trans = gaia_astrometric_detectability(
        period=108.75,
        m_planet=5.54,
        is_msini=True,
        m_star=0.989,
        distance=206.9,
        inclination=[90.0, 3.0],
        ra=125.18875,
        dec=46.20360,
        baseline_ruwe=1.46,
        baseline_excess_noise=0.22 * u.mas,
        on_ccd=False,
    )
    assert tab_trans["ruwe"][1] > 4.0

    # Test automatic binary calculation from separation and delta_mag
    tab_auto = gaia_astrometric_detectability(
        period=108.75,
        m_planet=5.54,
        is_msini=True,
        m_star=0.989,
        distance=206.9,
        inclination=90.0,
        ra=125.18875,
        dec=46.20360,
        binary_separation=1.68 * u.arcsec,
        binary_delta_mag=5.25,
    )
    assert np.isclose(tab_auto["ruwe"][0], 1.48, atol=0.08)
