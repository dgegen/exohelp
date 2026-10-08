import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import QTable
from astropy.utils.masked import Masked
from scipy import special as sc
from scipy import stats
from typing import NamedTuple

from ..citations import cites
from ..kepler import keplers_third_law, solve_kepler
from ..type import QuantityLike

__all__ = [
    "BinaryAstrometricContamination",
    "GaiaScanEvents",
    "ProjectedAstrometry",
    "ThieleInnesConstants",
    "along_scan_displacement",
    "astrometric_detection_thresholds",
    "astrometric_orbit",
    "astrometric_semi_amplitude",
    "binary_astrometric_contamination",
    "delta_chi2_to_sigma",
    "expected_astrometric_excess_noise",
    "expected_ruwe",
    "gaia_astrometric_detectability",
    "gaia_scan_events",
    "planet_mass_from_astrometry",
    "thiele_innes_constants",
]


class ProjectedAstrometry(NamedTuple):
    """Projected 2D astrometric sky offsets (dRA*, dDec)."""

    d_ra: u.Quantity
    d_dec: u.Quantity

    def along_scan(self, scan_angle: QuantityLike) -> u.Quantity:
        """Calculate 1D along-scan displacement from scan position angle."""
        return along_scan_displacement(self.d_ra, self.d_dec, scan_angle)

    @property
    def separation(self) -> u.Quantity:
        """Total projected angular separation from the host star barycenter."""
        return np.sqrt(self.d_ra**2 + self.d_dec**2)

    @property
    def position_angle(self) -> u.Quantity:
        """Position angle on the sky measured North to East."""
        d_ra_val = self.d_ra.to_value("uarcsec")
        d_dec_val = self.d_dec.to_value("uarcsec")
        pa = np.arctan2(d_ra_val, d_dec_val) % (2 * np.pi)
        return pa * u.rad


class BinaryAstrometricContamination(NamedTuple):
    """Estimated visual binary astrometric contamination on Gaia observations."""

    excess_noise: u.Quantity
    ruwe: float
    max_centroid_shift: u.Quantity
    contamination_flag: str


class ThieleInnesConstants(NamedTuple):
    """Thiele-Innes orbital constants (A, B, F, G)."""

    A: u.Quantity
    B: u.Quantity
    F: u.Quantity
    G: u.Quantity

    def project(self, orb_x: QuantityLike, orb_y: QuantityLike) -> ProjectedAstrometry:
        """Project normalized orbital anomalies (X, Y) onto 2D sky offsets.

        Follows the Thiele-Innes linear projection (Ranalli et al. 2018, Eqs. 3-4):
            dRA* = B * X + G * Y   [Eq. 3]
            dDec = A * X + F * Y   [Eq. 4]
        """
        d_ra = self.B * orb_x + self.G * orb_y
        d_dec = self.A * orb_x + self.F * orb_y
        return ProjectedAstrometry(d_ra=d_ra, d_dec=d_dec)


class GaiaScanEvents(NamedTuple):
    """Predicted Gaia scan observation epochs and scan direction position angles."""

    times: u.Quantity
    scan_angles: u.Quantity

    @property
    def n_transits(self) -> int:
        """Number of scan transit events."""
        return len(self.times)

    @property
    def time_baseline(self) -> u.Quantity:
        """Survey baseline duration (difference between last and first scan epoch)."""
        if len(self.times) == 0:
            return 0.0 * u.day
        return self.times[-1] - self.times[0]

    def project(
        self,
        d_ra_or_proj: QuantityLike | ProjectedAstrometry,
        d_dec: QuantityLike | None = None,
    ) -> u.Quantity:
        """Calculate 1D along-scan displacements using these scan event angles."""
        if isinstance(d_ra_or_proj, ProjectedAstrometry):
            return along_scan_displacement(d_ra_or_proj.d_ra, d_ra_or_proj.d_dec, self.scan_angles)
        if d_dec is None:
            raise ValueError(
                "Both d_ra and d_dec must be provided when not passing ProjectedAstrometry."
            )
        return along_scan_displacement(d_ra_or_proj, d_dec, self.scan_angles)


@cites("Perryman2014")
def astrometric_semi_amplitude(
    m_planet: QuantityLike,
    m_star: QuantityLike = 1.0,
    distance: QuantityLike | None = None,
    parallax: QuantityLike | None = None,
    semi_major_axis: QuantityLike | None = None,
    period: QuantityLike | None = None,
    flux_ratio: float = 0.0,
) -> u.Quantity:
    """Calculate the astrometric wobble semi-amplitude of the host star (alpha_*).

    Follows the astrometric reflex amplitude formula (Perryman et al. 2014, Eq. 1):
        alpha_* = (m_p / (M_* + m_p)) * (a / d)   [Eq. 1]

    When ``flux_ratio > 0`` (luminous companion), accounts for photocentric reduction
    (van de Kamp 1967, Eq. 1; Perryman et al. 2014, Section 2):
        a_photo = a * |m2 / (m1 + m2) - f2 / (f1 + f2)|

    Parameters
    ----------
    m_planet : QuantityLike
        Companion mass. Assumed to be in Jupiter masses (M_jup) if no unit is given.
    m_star : QuantityLike, optional
        Host star mass. Assumed to be in Solar masses (M_sun) if no unit is given. Default is 1.0.
    distance : QuantityLike, optional
        Distance to the system. Assumed to be in parsecs (pc) if no unit is given.
    parallax : QuantityLike, optional
        Stellar parallax. Assumed to be in milliarcseconds (mas) if no unit is given.
    semi_major_axis : QuantityLike, optional
        Orbital semi-major axis. Assumed to be in AU if no unit is given.
    period : QuantityLike, optional
        Orbital period. Assumed to be in days if no unit is given. Used with Kepler's
        third law (with total system mass M_* + m_p) if ``semi_major_axis`` is not specified.
    flux_ratio : float, optional
        Companion-to-host flux ratio F_2 / F_1. Default is 0.0 (dark companion / exoplanet).
        When > 0 (luminous stellar companion), accounts for photocentric reduction
        (van de Kamp 1967; Perryman et al. 2014).

    Returns
    -------
    alpha_star : u.Quantity
        Astrometric semi-amplitude of the stellar/photocentric reflex motion in microarcseconds (uarcsec).

    Examples
    --------
    >>> from exohelp.planet.astrometry import astrometric_semi_amplitude
    >>> alpha = astrometric_semi_amplitude(m_planet=1.0, m_star=1.0, distance=10.0, semi_major_axis=1.0)
    >>> f"{alpha.to('uarcsec').value:.2f}"
    '95.37'
    """
    m_planet = u.Quantity(m_planet, "M_jup")
    m_star = u.Quantity(m_star, "M_sun")
    m_total = m_star + m_planet.to("M_sun")

    if distance is None and parallax is None:
        raise ValueError("Either 'distance' or 'parallax' must be provided.")
    if distance is None:
        parallax = u.Quantity(parallax, "mas")
        distance = parallax.to("pc", equivalencies=u.parallax())
    else:
        distance = u.Quantity(distance, "pc")

    if semi_major_axis is None:
        if period is None:
            raise ValueError("Either 'semi_major_axis' or 'period' must be provided.")
        semi_major_axis = keplers_third_law(period=period, mass=m_total)
    else:
        semi_major_axis = u.Quantity(semi_major_axis, "AU")

    a_star = semi_major_axis * (m_planet.to("M_sun") / m_total)

    if flux_ratio > 0.0:
        # Photocenter reduction for luminous companion (van de Kamp 1967; Perryman 2014):
        # a_photo = a * |m2 / (m1 + m2) - f2 / (f1 + f2)|
        mass_fraction = (m_planet.to("M_sun") / m_total).value
        flux_fraction = flux_ratio / (1.0 + flux_ratio)
        photo_reduction = abs(mass_fraction - flux_fraction) / mass_fraction
        a_star = a_star * photo_reduction

    alpha_rad = (a_star / distance).to_value(u.dimensionless_unscaled)
    return (alpha_rad * u.rad).to("uarcsec")


@cites("Perryman2014")
def planet_mass_from_astrometry(
    alpha: QuantityLike,
    m_star: QuantityLike = 1.0,
    distance: QuantityLike | None = None,
    parallax: QuantityLike | None = None,
    semi_major_axis: QuantityLike | None = None,
    period: QuantityLike | None = None,
) -> u.Quantity:
    """Calculate the companion true dynamical mass from the astrometric wobble amplitude.

    Solves the exact relation a_* = a * m_p / (M_* + m_p) where a_* = alpha * d
    (Perryman et al. 2014, Eq. 1). When period is provided, solves Kepler's third law
    simultaneously for the true dynamical mass. Because the astrometric semi-major axis
    is invariant under projection onto the sky plane, alpha directly measures the true mass.

    Parameters
    ----------
    alpha : QuantityLike
        Astrometric semi-amplitude alpha_*. Assumed to be in microarcseconds (uarcsec)
        if no unit is given.
    m_star : QuantityLike, optional
        Host star mass. Assumed to be in Solar masses (M_sun). Default is 1.0.
    distance : QuantityLike, optional
        Distance to the system. Assumed to be in parsecs (pc).
    parallax : QuantityLike, optional
        Stellar parallax. Assumed to be in milliarcseconds (mas).
    semi_major_axis : QuantityLike, optional
        Orbital semi-major axis in AU.
    period : QuantityLike, optional
        Orbital period in days.

    Returns
    -------
    m_planet : u.Quantity
        Companion dynamical true mass in Jupiter masses (M_jup).
    """
    alpha = u.Quantity(alpha, "uarcsec")
    m_star = u.Quantity(m_star, "M_sun")

    if distance is None and parallax is None:
        raise ValueError("Either 'distance' or 'parallax' must be provided.")
    if distance is None:
        parallax = u.Quantity(parallax, "mas")
        distance = parallax.to("pc", equivalencies=u.parallax())
    else:
        distance = u.Quantity(distance, "pc")

    a_star = (alpha.to_value("rad") * distance).to("AU")

    if semi_major_axis is not None:
        a = u.Quantity(semi_major_axis, "AU")
        # Exact: a_* (M_* + m_p) = a * m_p => m_p = M_* * a_* / (a - a_*)
        diff = a - a_star
        is_invalid = diff.to_value("AU") <= 0.0
        if np.any(is_invalid):
            safe_diff = np.where(is_invalid, 1.0 * u.AU, diff)
            m_p_msun = m_star * (a_star / safe_diff)
            return Masked(m_p_msun.to("M_jup"), mask=is_invalid)
        m_p_msun = m_star * (a_star / diff)
        return m_p_msun.to("M_jup")

    if period is None:
        raise ValueError("Either 'semi_major_axis' or 'period' must be provided.")

    period = u.Quantity(period, "day")
    k0 = keplers_third_law(period=period, mass=1.0 * u.M_sun).to_value("AU")
    ms_val = m_star.to_value("M_sun")
    as_val = a_star.to_value("AU")
    ratio = as_val / k0
    m = ratio * (ms_val ** (2 / 3))
    for _ in range(20):
        f = m - ratio * ((ms_val + m) ** (2 / 3))
        f_prime = 1.0 - ratio * (2 / 3) * ((ms_val + m) ** (-1 / 3))
        step = f / f_prime
        m = m - step
        if np.all(np.abs(step) < 1e-12):
            break
    return (m * u.M_sun).to("M_jup")


def _thiele_innes_coefficients(
    alpha: float | np.ndarray,
    omega_rad: float,
    node_rad: float | np.ndarray,
    inc_rad: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate Thiele-Innes coefficients (A, B, F, G) for scalars or broadcastable arrays."""
    cos_om = np.cos(omega_rad)
    sin_om = np.sin(omega_rad)
    cos_nodes = np.cos(node_rad)
    sin_nodes = np.sin(node_rad)
    cos_inc = np.cos(inc_rad)

    val_a = alpha * (cos_om * cos_nodes - sin_om * sin_nodes * cos_inc)
    val_b = alpha * (cos_om * sin_nodes + sin_om * cos_nodes * cos_inc)
    val_f = alpha * (-sin_om * cos_nodes - cos_om * sin_nodes * cos_inc)
    val_g = alpha * (-sin_om * sin_nodes + cos_om * cos_nodes * cos_inc)
    return val_a, val_b, val_f, val_g


@cites("Perryman2014", "Ranalli2018")
def thiele_innes_constants(
    a_star: QuantityLike,
    inclination: QuantityLike = 90.0,
    omega: QuantityLike = 0.0,
    node: QuantityLike = 0.0,
) -> ThieleInnesConstants:
    """Calculate the Thiele-Innes orbital constants (A, B, F, G).

    Follows the Thiele-Innes orbital formulation (Ranalli et al. 2018, Eqs. 5-8;
    Perryman et al. 2014, Section 2):
        A = alpha_* * (cos(omega) * cos(Omega) - sin(omega) * sin(Omega) * cos(i))   [Eq. 5]
        B = alpha_* * (cos(omega) * sin(Omega) + sin(omega) * cos(Omega) * cos(i))   [Eq. 6]
        F = alpha_* * (-sin(omega) * cos(Omega) - cos(omega) * sin(Omega) * cos(i))  [Eq. 7]
        G = alpha_* * (-sin(omega) * sin(Omega) + cos(omega) * cos(Omega) * cos(i))  [Eq. 8]

    Parameters
    ----------
    a_star : QuantityLike
        Astrometric semi-major axis (alpha_*). Assumed to be in microarcseconds (uarcsec)
        if no unit is given.
    inclination : QuantityLike, optional
        Orbital inclination (i). Assumed to be in degrees. Default is 90.0 deg.
    omega : QuantityLike, optional
        Argument of periastron of the host star (omega_*). Assumed to be in degrees.
        Default is 0.0 deg. (Consistent with RV convention where omega_* = omega_p + 180 deg).
    node : QuantityLike, optional
        Longitude of ascending node (Omega). Assumed to be in degrees. Default is 0.0 deg.

    Returns
    -------
    ThieleInnesConstants
        Named tuple containing (A, B, F, G) as Quantity objects in uarcsec.
    """
    a_star = u.Quantity(a_star, "uarcsec")
    inc_rad = u.Quantity(inclination, "deg").to("rad").value
    w_rad = u.Quantity(omega, "deg").to("rad").value
    node_rad = u.Quantity(node, "deg").to("rad").value

    val_a, val_b, val_f, val_g = _thiele_innes_coefficients(
        alpha=a_star.to_value("uarcsec"),
        omega_rad=w_rad,
        node_rad=node_rad,
        inc_rad=inc_rad,
    )

    return ThieleInnesConstants(
        A=val_a * u.uarcsec,
        B=val_b * u.uarcsec,
        F=val_f * u.uarcsec,
        G=val_g * u.uarcsec,
    )


@cites("Perryman2014", "Ranalli2018")
def astrometric_orbit(
    times: QuantityLike,
    period: QuantityLike,
    m_planet: QuantityLike,
    m_star: QuantityLike = 1.0,
    distance: QuantityLike | None = None,
    parallax: QuantityLike | None = None,
    semi_major_axis: QuantityLike | None = None,
    inclination: QuantityLike = 90.0,
    eccentricity: QuantityLike = 0.0,
    omega: QuantityLike = 0.0,
    node: QuantityLike = 0.0,
    t0: QuantityLike = 0.0,
) -> ProjectedAstrometry:
    """Calculate 2D projected sky displacements (dRA*, dDec) of the host star.

    Follows the Thiele-Innes orbital formulation (Ranalli et al. 2018, Eqs. 3-4, 9-12;
    Perryman et al. 2014):
        X(t) = cos(E(t)) - e                    [Eq. 9]
        Y(t) = sqrt(1 - e^2) * sin(E(t))        [Eq. 10]
        E(t) = M(t) + e * sin(E(t))             [Eq. 11]
        M(t) = (2 * pi / P) * (t - T_p)         [Eq. 12]
    projected through natural elements A, B, F, G (Ranalli et al. 2018, Eqs. 3-4).

    Parameters
    ----------
    times : QuantityLike
        Epoch times. Assumed to be in days if no unit is given.
    period : QuantityLike
        Orbital period. Assumed to be in days if no unit is given.
    m_planet : QuantityLike
        Companion mass. Assumed to be in Jupiter masses (M_jup).
    m_star : QuantityLike, optional
        Host star mass in M_sun. Default is 1.0.
    distance : QuantityLike, optional
        Distance in pc.
    parallax : QuantityLike, optional
        Parallax in mas.
    semi_major_axis : QuantityLike, optional
        Semi-major axis in AU.
    inclination : QuantityLike, optional
        Orbital inclination in degrees. Default is 90.0.
    eccentricity : QuantityLike, optional
        Orbital eccentricity (0 <= e < 1). Default is 0.0.
    omega : QuantityLike, optional
        Argument of periastron of the host star (omega_*). Assumed to be in degrees.
        Default is 0.0 deg. (Consistent with RV convention where omega_* = omega_p + 180 deg).
    node : QuantityLike, optional
        Longitude of ascending node in degrees. Default is 0.0.
    t0 : QuantityLike, optional
        Time of periastron passage. Assumed to be in days. Default is 0.0.

    Returns
    -------
    ProjectedAstrometry
        Named tuple with (d_ra, d_dec) in microarcseconds (uarcsec).
    """
    times = u.Quantity(times, "day")
    period = u.Quantity(period, "day")
    t0 = u.Quantity(t0, "day")
    e = float(
        eccentricity.to_value(u.dimensionless_unscaled)
        if isinstance(eccentricity, u.Quantity)
        else eccentricity
    )

    alpha_star = astrometric_semi_amplitude(
        m_planet=m_planet,
        m_star=m_star,
        distance=distance,
        parallax=parallax,
        semi_major_axis=semi_major_axis,
        period=period,
    )

    ti = thiele_innes_constants(
        a_star=alpha_star,
        inclination=inclination,
        omega=omega,
        node=node,
    )

    mean_motion = 2 * np.pi / period.to("day").value
    mean_anom = (mean_motion * (times.to("day").value - t0.to("day").value)) % (2 * np.pi)
    ecc_anom = solve_kepler(mean_anom, e)

    orb_x = np.cos(ecc_anom) - e
    orb_y = np.sqrt(1 - e**2) * np.sin(ecc_anom)

    return ti.project(orb_x, orb_y)


@cites("Ranalli2018")
def along_scan_displacement(
    d_ra: QuantityLike,
    d_dec: QuantityLike,
    scan_angle: QuantityLike,
) -> u.Quantity:
    """Calculate 1D along-scan displacement from 2D sky offsets and scan angle.

    Follows the Gaia along-scan projection (Ranalli et al. 2018, Eq. 14):
        Delta eta(t) = Delta alpha * cos(theta(t)) + Delta delta * sin(theta(t))   [Eq. 14]
    expressed with the standard astronomical scan position angle psi measured
    North to East (psi = 90 deg - theta):
        d_AL = d_RA* * sin(psi) + d_Dec * cos(psi)

    Parameters
    ----------
    d_ra : QuantityLike
        Offset in Right Ascension (dRA*). Assumed to be in microarcseconds (uarcsec).
    d_dec : QuantityLike
        Offset in Declination (dDec). Assumed to be in microarcseconds (uarcsec).
    scan_angle : QuantityLike
        Scan position angle (psi) measured North to East. Assumed to be in radians if no unit is given.

    Returns
    -------
    d_al : u.Quantity
        Along-scan projected displacement in microarcseconds (uarcsec).
    """
    d_ra = u.Quantity(d_ra, "uarcsec")
    d_dec = u.Quantity(d_dec, "uarcsec")
    psi = u.Quantity(scan_angle, "rad").to("rad").value

    d_al = d_ra * np.sin(psi) + d_dec * np.cos(psi)
    return d_al.to("uarcsec")
    d_ra = u.Quantity(d_ra, "uarcsec")
    d_dec = u.Quantity(d_dec, "uarcsec")
    psi = u.Quantity(scan_angle, "rad").to("rad").value

    d_al = d_ra * np.sin(psi) + d_dec * np.cos(psi)
    return d_al.to("uarcsec")


@cites("GaiaCollaboration2016")
def gaia_scan_events(
    ra: QuantityLike,
    dec: QuantityLike,
    duration_years: float = 5.5,
    step_seconds: float = 30.0,
    epoch_start: QuantityLike | None = None,
) -> GaiaScanEvents:
    """Predict Gaia nominal scanning law (NSL) observation epochs and scan direction angles.

    Parameters
    ----------
    ra : QuantityLike
        Right Ascension. Assumed to be in degrees if no unit is given.
    dec : QuantityLike
        Declination. Assumed to be in degrees if no unit is given.
    duration_years : float, optional
        Survey duration in years (e.g., 5.5 for nominal DR4). Default is 5.5.
    step_seconds : float, optional
        Time grid resolution in seconds. Default is 30.0.
    epoch_start : QuantityLike, optional
        Mission start reference epoch. Assumed to be in days (e.g., BJD 2456860.5 for J2014.5).
        If provided, scan event timestamps are shifted by this epoch offset. Default is None (starts at 0.0).

    Returns
    -------
    GaiaScanEvents
        Named tuple with ``times`` (Quantity in days) and ``scan_angles`` (Quantity in radians).
    """
    ra_deg = u.Quantity(ra, "deg").to("deg").value
    dec_deg = u.Quantity(dec, "deg").to("deg").value

    target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    ecl = target.geocentricmeanecliptic

    beta = ecl.lat.rad
    lambda_ecl = ecl.lon.rad

    u_targ = np.array(
        [
            np.cos(beta) * np.cos(lambda_ecl),
            np.cos(beta) * np.sin(lambda_ecl),
            np.sin(beta),
        ]
    )

    t_days = np.arange(0.0, duration_years * 365.25, step_seconds / 86400.0)

    omega_prec = 2 * np.pi / 63.12  # 63.12 days precession period
    omega_spin = 2 * np.pi / (6.0 / 24.0)  # 6 hours spin period
    xi = np.radians(45.0)  # 45 deg solar aspect angle
    fov_sep = np.radians(106.5)  # 106.5 deg telescope separation

    lambda_s = 2 * np.pi * (t_days / 365.25)
    nu = omega_prec * t_days

    z_s = np.column_stack(
        [
            np.cos(xi) * np.cos(lambda_s) - np.sin(xi) * np.sin(lambda_s) * np.cos(nu),
            np.cos(xi) * np.sin(lambda_s) + np.sin(xi) * np.cos(lambda_s) * np.cos(nu),
            np.sin(xi) * np.sin(nu),
        ]
    )

    dot_z = np.dot(z_s, u_targ)
    belt_mask = np.abs(dot_z) < np.sin(np.radians(0.35))

    t_belt = t_days[belt_mask]
    if len(t_belt) == 0:
        return GaiaScanEvents(times=np.array([]) * u.day, scan_angles=np.array([]) * u.rad)

    phi_spin = (omega_spin * t_belt) % (2 * np.pi)

    s_hat = np.column_stack(
        [
            np.cos(lambda_s[belt_mask]),
            np.sin(lambda_s[belt_mask]),
            np.zeros(len(t_belt)),
        ]
    )
    x_s = np.cross(s_hat, z_s[belt_mask])
    norm_x = np.linalg.norm(x_s, axis=1, keepdims=True)
    norm_x[norm_x == 0] = 1.0
    x_s /= norm_x
    y_s = np.cross(z_s[belt_mask], x_s)

    target_phase = np.arctan2(np.sum(u_targ * y_s, axis=1), np.sum(u_targ * x_s, axis=1)) % (
        2 * np.pi
    )

    diff_fov1 = np.abs(np.arctan2(np.sin(target_phase - phi_spin), np.cos(target_phase - phi_spin)))
    diff_fov2 = np.abs(
        np.arctan2(
            np.sin(target_phase - (phi_spin + fov_sep)),
            np.cos(target_phase - (phi_spin + fov_sep)),
        )
    )

    is_transit = (diff_fov1 < np.radians(0.35)) | (diff_fov2 < np.radians(0.35))

    scan_times = t_belt[is_transit]
    if len(scan_times) == 0:
        return GaiaScanEvents(times=np.array([]) * u.day, scan_angles=np.array([]) * u.rad)

    dt_group = np.diff(scan_times, prepend=scan_times[0] - 1.0)
    scan_times = scan_times[dt_group > (2.0 / 1440.0)]

    epoch_offset = (
        u.Quantity(epoch_start, "day").to_value("day") if epoch_start is not None else 0.0
    )

    # Compute along-scan position angle psi on the sky (measured North towards East).
    # The scan sweeps along a great circle with normal z_s (satellite spin axis).
    # Rotate z_s at transit epochs from geocentric ecliptic coordinates to ICRS equatorial.
    eps_obliq = np.radians(23.4392911)
    cos_eps, sin_eps = np.cos(eps_obliq), np.sin(eps_obliq)
    r_ecl2eq = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_eps, -sin_eps],
            [0.0, sin_eps, cos_eps],
        ]
    )

    lambda_s_trans = 2 * np.pi * (scan_times / 365.25)
    nu_trans = omega_prec * scan_times
    z_s_ecl = np.column_stack(
        [
            np.cos(xi) * np.cos(lambda_s_trans)
            - np.sin(xi) * np.sin(lambda_s_trans) * np.cos(nu_trans),
            np.cos(xi) * np.sin(lambda_s_trans)
            + np.sin(xi) * np.cos(lambda_s_trans) * np.cos(nu_trans),
            np.sin(xi) * np.sin(nu_trans),
        ]
    )
    z_s_eq = z_s_ecl @ r_ecl2eq.T

    # Equatorial local frame at the target (ICRS)
    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)
    u_eq = np.array(
        [
            np.cos(dec_rad) * np.cos(ra_rad),
            np.cos(dec_rad) * np.sin(ra_rad),
            np.sin(dec_rad),
        ]
    )
    e_north = np.array(
        [
            -np.sin(dec_rad) * np.cos(ra_rad),
            -np.sin(dec_rad) * np.sin(ra_rad),
            np.cos(dec_rad),
        ]
    )
    e_east = np.array(
        [
            -np.sin(ra_rad),
            np.cos(ra_rad),
            0.0,
        ]
    )

    # Instantaneous along-scan unit tangent vector on the celestial sphere
    t_scan = np.cross(z_s_eq, u_eq)
    norm_t = np.linalg.norm(t_scan, axis=1, keepdims=True)
    norm_t[norm_t == 0] = 1.0
    t_scan = t_scan / norm_t

    # Position angle psi measured North to East
    cos_psi = np.sum(t_scan * e_north, axis=1)
    sin_psi = np.sum(t_scan * e_east, axis=1)
    psi = np.arctan2(sin_psi, cos_psi) % (2 * np.pi)

    return GaiaScanEvents(times=(scan_times + epoch_offset) * u.day, scan_angles=psi * u.rad)


@cites("Lindegren2021_edr3_astro", "Fabricius2021_edr3_val", "ElBadry2021_binaries")
def expected_ruwe(
    delta_chi2: QuantityLike,
    n_transits: int = 80,
    baseline_ruwe: float = 1.0,
    on_ccd: bool = True,
) -> float | np.ndarray:
    """Calculate the expected Gaia Renormalized Unit Weight Error (RUWE).

    Under a 5-parameter single-star astrometric fit, an unmodeled orbital motion
    inflates the total chi^2 by Delta chi^2. Following Lindegren et al. (2021, Section 5.1 & 5.3):
        UWE = [ chi^2 / (n - n_p) ]^(1/2)   [Lindegren et al. 2021, Sect. 5.1]
    where n is the number of good along-scan CCD observations (astrometric_n_good_obs_al)
    and n_p = 5. Combined with a baseline RUWE (e.g., from calibration floor or visual
    binary PSF blending; Fabricius et al. 2021, El-Badry et al. 2021):
        RUWE = sqrt(max(baseline_ruwe^2 + Delta chi^2 / dof, 0.0)).

    In the official Gaia DPAC astrometric pipeline, the 5-parameter solution is fit
    at the individual CCD level (up to 9 Astrometric Field CCDs AF1-AF9 per transit;
    Lindegren et al. 2021, Table 4), meaning dof = 9 * n_transits - 5. When evaluating
    simplified transit-level models (where each transit is treated as a single measurement),
    dof = n_transits - 5.

    Parameters
    ----------
    delta_chi2 : QuantityLike
        Astrometric signal Delta chi^2 value(s).
    n_transits : int, optional
        Number of along-scan observations/transits. Default is 80 (representative for Gaia DR3/DR4).
    baseline_ruwe : float, optional
        Baseline RUWE from astrometric error floor or visual binary contamination.
        Default is 1.0 (clean single star).
    on_ccd : bool, optional
        Whether to calculate RUWE at the official Gaia CCD level (assumes ~9 Astrometric Field
        CCDs per transit, matching ``astrometric_n_good_obs_al`` in the Gaia archive; default is True).
        If False, calculates RUWE at the simplified transit level.

    Returns
    -------
    ruwe : float or np.ndarray
        Expected RUWE value(s). (Values > 1.4 typically indicate non-single star motion).
    """
    d = np.asarray(delta_chi2, dtype=float)
    n_meas = (9 * n_transits) if on_ccd else n_transits
    dof = max(n_meas - 5, 1)
    val = np.sqrt(np.maximum(baseline_ruwe**2 + d / dof, 0.0))
    if d.ndim == 0:
        return float(val)
    return val


@cites("Lindegren2021_edr3_astro", "Fabricius2021_edr3_val", "ElBadry2021_binaries")
def expected_astrometric_excess_noise(
    delta_chi2: QuantityLike,
    n_transits: int = 80,
    sigma_al: QuantityLike = 34.0,
    baseline_excess_noise: QuantityLike = 0.0,
    on_ccd: bool = True,
) -> u.Quantity:
    """Calculate the expected Gaia astrometric excess noise from an unmodeled orbital signal and baseline binary jitter.

    Excess noise epsilon is added in quadrature to measurement errors such that the
    reduced chi^2 is 1 (Lindegren et al. 2012, Eq. 36; Lindegren et al. 2021, Section 5.1 & 5.3):
        epsilon = sqrt(epsilon_baseline^2 + sigma_meas^2 * max(RUWE_planet^2 - 1, 0.0)).

    Parameters
    ----------
    delta_chi2 : QuantityLike
        Astrometric signal Delta chi^2 value(s).
    n_transits : int, optional
        Number of along-scan observations/transits. Default is 80.
    sigma_al : QuantityLike, optional
        Along-scan single-transit astrometric uncertainty. Assumed in microarcseconds (uarcsec).
        Default is 34.0 uarcsec.
    baseline_excess_noise : QuantityLike, optional
        Baseline astrometric excess noise from visual binary contamination.
        Default is 0.0 uarcsec.
    on_ccd : bool, optional
        Whether to calculate excess noise at the official Gaia CCD level (default is True).

    Returns
    -------
    excess_noise : u.Quantity
        Expected astrometric excess noise in microarcseconds (uarcsec).
    """
    sigma = u.Quantity(sigma_al, "uarcsec")
    base_noise = u.Quantity(baseline_excess_noise, "uarcsec")
    ruwe_planet = expected_ruwe(delta_chi2, n_transits=n_transits, baseline_ruwe=1.0, on_ccd=on_ccd)
    # At the CCD level, per-CCD uncertainty is sigma_CCD = sqrt(9) * sigma_AL = 3 * sigma_AL.
    # Because (RUWE^2 - 1) scales as 1/9, the factor 3^2 = 9 cancels out, yielding an invariant physical excess noise.
    sigma_meas = (3.0 * sigma) if on_ccd else sigma
    planet_excess_sq = (sigma_meas.to_value("uarcsec") ** 2) * np.maximum(
        np.asarray(ruwe_planet) ** 2 - 1.0, 0.0
    )
    total_excess_uas = np.sqrt(base_noise.to_value("uarcsec") ** 2 + planet_excess_sq)
    return u.Quantity(total_excess_uas, "uarcsec")


@cites("Lindegren2021_edr3_astro", "Fabricius2021_edr3_val", "ElBadry2021_binaries")
def binary_astrometric_contamination(
    separation: QuantityLike,
    delta_mag: float,
    sigma_al: QuantityLike = 34.0,
    n_transits: int = 80,
    phot_g_mean_mag: float | None = None,
    on_ccd: bool = True,
) -> BinaryAstrometricContamination:
    """Estimate the astrometric excess noise and RUWE inflation induced by a visual binary companion.

    In visual binaries with angular separations rho <= 2-4 arcseconds, scan-angle-dependent
    PSF contamination and window allocation conflicts inflate the Gaia astrometric excess noise
    and RUWE (Fabricius et al. 2021, Section 2.3 & 3.8; El-Badry et al. 2021, Section 2, Eq. 2;
    Section 5, Eqs. 13-15, & Figure 18).
    Maximum photocentric centroid displacement is:
        a_photo = rho * f_sec = rho * q / (1 + q)
    where q = 10^(-0.4 * Delta G).

    Parameters
    ----------
    separation : QuantityLike
        Angular separation between the components. Assumed to be in arcseconds if no unit is given.
    delta_mag : float
        Gaia G-band magnitude difference Delta G = G_secondary - G_primary.
        (Positive if the companion is fainter, negative if evaluating the fainter star with a brighter companion).
    sigma_al : QuantityLike, optional
        Single-transit along-scan astrometric uncertainty. Assumed in microarcseconds (uarcsec).
        Default is 34.0 uarcsec.
    n_transits : int, optional
        Number of along-scan transits. Default is 80.
    phot_g_mean_mag : float, optional
        Gaia G-band magnitude of the target star. If provided, overrides sigma_al with the
        characteristic Gaia single-transit uncertainty for that magnitude.
    on_ccd : bool, optional
        Whether to calculate RUWE at the official Gaia CCD level (assumes ~9 Astrometric Field
        CCDs per transit; default is True).

    Returns
    -------
    BinaryAstrometricContamination
        Named tuple containing:
        - ``excess_noise``: Predicted astrometric excess noise in microarcseconds (uarcsec).
        - ``ruwe``: Predicted baseline RUWE under binary contamination.
        - ``max_centroid_shift``: Maximum instantaneous photocentric displacement in microarcseconds.
        - ``contamination_flag``: Diagnostic status ('clean', 'mild_contamination', or 'blended_window_conflict').

    References
    ----------
    Lindegren et al. (2021), A&A, 649, A2.
    Fabricius et al. (2021), A&A, 649, A5.
    El-Badry et al. (2021), MNRAS, 506, 2269.
    """
    sep_arcsec = float(u.Quantity(separation, "arcsec").to_value("arcsec"))
    sigma = u.Quantity(sigma_al, "uarcsec")

    if phot_g_mean_mag is not None:
        g = float(phot_g_mean_mag)
        sigma_val = max(34.0, 100.0 * (10.0 ** (0.2 * max(0.0, g - 11.0))))
        sigma = sigma_val * u.uarcsec

    q_flux = 10.0 ** (-0.4 * delta_mag)
    f_sec = q_flux / (1.0 + q_flux)

    sep_uas = sep_arcsec * 1.0e6
    max_shift_uas = sep_uas * f_sec

    if sep_arcsec > 5.0:
        excess_uas = 0.0
        flag = "clean"
    elif sep_arcsec > 2.5:
        geom_weight = float(np.exp(-0.5 * ((sep_arcsec - 1.5) / 0.8) ** 2))
        excess_uas = max_shift_uas * geom_weight * 0.05
        flag = "mild_contamination"
    else:
        geom_weight = float(np.exp(-0.5 * ((sep_arcsec - 1.5) / 1.0) ** 2))
        if delta_mag >= 0:
            excess_uas = (max_shift_uas * 0.010 + 90.0) * geom_weight
        else:
            excess_uas = (max_shift_uas * 0.00032 + 520.0) * geom_weight
        flag = "blended_window_conflict"

    n_meas = (9 * n_transits) if on_ccd else n_transits
    dof = max(n_meas - 5, 1)
    sig_val = sigma.to_value("uarcsec")
    sig_denom = max(sig_val, 200.0) if phot_g_mean_mag is None else sig_val
    ruwe = float(np.sqrt(1.0 + (excess_uas / sig_denom) ** 2 * (dof / n_meas)))

    return BinaryAstrometricContamination(
        excess_noise=u.Quantity(excess_uas, "uarcsec"),
        ruwe=ruwe,
        max_centroid_shift=u.Quantity(max_shift_uas, "uarcsec"),
        contamination_flag=flag,
    )


def _mode_to_df(mode: str) -> int:
    mode_clean = mode.lower().strip()
    if mode_clean in ("rv", "rv_conditioned", "rv_fixed", "fixed"):
        return 2
    elif mode_clean in ("joint", "rv_informed", "informed"):
        return 3
    elif mode_clean in ("blind", "unconstrained", "search", "perryman"):
        return 7
    elif mode_clean in ("blind_5yr", "blind_5y", "ranalli_5yr", "ranalli_5y", "5yr"):
        return 11
    elif mode_clean in ("blind_10yr", "blind_10y", "ranalli_10yr", "ranalli_10y", "10yr"):
        return 16
    else:
        raise ValueError(
            f"Unknown detection mode {mode!r}. Expected 'rv' (known RV orbit, 2 free astrometric "
            f"parameters), 'joint' (3 free parameters), 'blind' (7 orbital parameters; Perryman et al. 2014), "
            f"'blind_5yr' (11 effective dof for 5-yr mission; Ranalli et al. 2018, Section 6.2), or "
            f"'blind_10yr' (16 effective dof for 10-yr mission; Ranalli et al. 2018, Section 6.2)."
        )


@cites("Perryman2014", "Ranalli2018")
def delta_chi2_to_sigma(delta_chi2: QuantityLike, mode: str = "rv") -> np.ndarray | float:
    """Convert astrometric Delta chi^2 to equivalent Gaussian significance in standard deviations (sigma).

    Parameters
    ----------
    delta_chi2 : QuantityLike
        Astrometric signal Delta chi^2 value(s).
    mode : {"rv", "blind", "joint", "blind_5yr", "blind_10yr"}, optional
        Detection context determining the astrometric degrees of freedom:
        - ``"rv"`` (default): Orbit pre-constrained by RVs (2 astrometric degrees of freedom:
          inclination and ascending node).
        - ``"joint"``: Joint fit with free mass/amplitude (3 degrees of freedom).
        - ``"blind"``: Unconstrained astrometric blind search (7 nominal degrees of freedom;
          Perryman et al. 2014).
        - ``"blind_5yr"``: Unconstrained blind search effective degrees of freedom for a 5-year mission
          accounting for non-linear period grid fitting (11 degrees of freedom; Ranalli et al. 2018, Section 6.2).
        - ``"blind_10yr"``: Unconstrained blind search effective degrees of freedom for a 10-year mission
          (16 degrees of freedom; Ranalli et al. 2018, Section 6.2).

    Returns
    -------
    sigma : np.ndarray or float
        Equivalent Gaussian significance in standard deviations.

    Examples
    --------
    >>> from exohelp.planet.astrometry import delta_chi2_to_sigma
    >>> round(float(delta_chi2_to_sigma(30.70, mode="rv")), 2)
    5.19
    """
    df = _mode_to_df(mode)
    vals = np.asarray(delta_chi2, dtype=float)

    log_p = stats.chi2.logsf(vals, df)
    # When delta_chi2 / 2 > ~700, float64 underflows to -inf; use asymptotic expansion:
    # ln Q(a, z) ~ -z + (a - 1) * ln(z) - gammaln(a)
    bad = np.isneginf(log_p) | np.isnan(log_p) | ((vals / 2.0) > 650.0)
    if np.any(bad):
        z_bad = vals[bad] / 2.0
        a = df / 2.0
        log_p[bad] = -z_bad + (a - 1.0) * np.log(z_bad) - sc.gammaln(a)

    p = stats.chi2.sf(vals, df)

    sigma = np.zeros_like(vals)
    mask_normal = p > 1e-15
    sigma[mask_normal] = stats.norm.isf(p[mask_normal] / 2.0)

    mask_extreme = ~mask_normal
    if np.any(mask_extreme):
        lp = log_p[mask_extreme] - np.log(2.0)
        s = np.sqrt(-2.0 * lp)
        for _ in range(3):
            s = np.sqrt(np.maximum(-2.0 * (lp + np.log(s * np.sqrt(2.0 * np.pi))), 0.0))
        sigma[mask_extreme] = s

    if vals.ndim == 0:
        return float(sigma)
    return sigma


@cites("Perryman2014", "Ranalli2018")
def astrometric_detection_thresholds(mode: str = "rv") -> QTable:
    """Return critical Delta chi^2 and SNR detection thresholds for the given detection mode.

    Parameters
    ----------
    mode : {"rv", "blind", "joint", "blind_5yr", "blind_10yr"}, optional
        Detection mode:
        - ``"rv"`` (default): Planet orbit pre-constrained by RVs (2 astrometric degrees of freedom).
        - ``"joint"``: Joint RV-informed fit with free mass/amplitude (3 degrees of freedom).
        - ``"blind"``: Unconstrained astrometric blind search (7 nominal degrees of freedom; Perryman et al. 2014).
        - ``"blind_5yr"``: Unconstrained blind search for 5-year mission (11 effective degrees of freedom; Ranalli et al. 2018).
        - ``"blind_10yr"``: Unconstrained blind search for 10-year mission (16 effective degrees of freedom; Ranalli et al. 2018).

    Returns
    -------
    astropy.table.QTable
        Table of critical thresholds with columns:
        ``level``, ``sigma``, ``p_value``, ``delta_chi2``, ``snr``, ``description``.
    """
    df = _mode_to_df(mode)

    levels = [
        "1-sigma",
        "2-sigma",
        "3-sigma",
        "FAP=1e-4",
        "5-sigma (Discovery)",
        "Ranalli Benchmark",
        "Perryman Benchmark",
    ]
    p_values = [
        float(2 * stats.norm.sf(1.0)),
        float(2 * stats.norm.sf(2.0)),
        float(2 * stats.norm.sf(3.0)),
        1e-4,
        float(2 * stats.norm.sf(5.0)),
        float(stats.chi2.sf(80.0, df)),
        float(stats.chi2.sf(100.0, df)),
    ]
    delta_chi2s = [
        float(stats.chi2.isf(p_values[0], df)),
        float(stats.chi2.isf(p_values[1], df)),
        float(stats.chi2.isf(p_values[2], df)),
        float(stats.chi2.isf(1e-4, df)),
        float(stats.chi2.isf(p_values[4], df)),
        80.0,
        100.0,
    ]
    sigmas = [
        1.0,
        2.0,
        3.0,
        float(stats.norm.isf(1e-4 / 2.0)),
        5.0,
        float(delta_chi2_to_sigma(80.0, mode=mode)),
        float(delta_chi2_to_sigma(100.0, mode=mode)),
    ]
    snrs = [float(np.sqrt(d)) for d in delta_chi2s]
    descriptions = [
        "1-sigma noise floor (68.3% confidence)",
        "2-sigma marginal limit (95.4% confidence)",
        "3-sigma candidate threshold (99.73% confidence)",
        "False alarm probability < 10^-4 threshold",
        "5-sigma gold standard discovery threshold",
        "Ranalli et al. (2018) robust Gaia detection benchmark (ΔBIC=20, Δχ²=80)",
        "Perryman et al. (2014) high-fidelity orbital recovery benchmark (Δχ²=100)",
    ]

    table = QTable(
        {
            "level": levels,
            "sigma": np.asarray(sigmas, dtype=float),
            "p_value": np.asarray(p_values, dtype=float),
            "delta_chi2": np.asarray(delta_chi2s, dtype=float),
            "snr": np.asarray(snrs, dtype=float),
            "description": descriptions,
        }
    )
    table.meta["references"] = ["Perryman2014", "Ranalli2018"]
    table.meta["mode"] = mode
    table.meta["degrees_of_freedom"] = df
    table["sigma"].info.description = "Equivalent Gaussian significance (standard deviations)"
    table["sigma"].info.meta = {"short_description": "Significance (sigma)"}
    table["p_value"].info.description = "False alarm probability p-value"
    table["p_value"].info.meta = {"short_description": "p-value"}
    table["delta_chi2"].info.description = "Critical Delta chi^2 threshold"
    table["delta_chi2"].info.meta = {"short_description": "Critical Δχ²"}
    table["snr"].info.description = "Critical astrometric SNR threshold"
    table["snr"].info.meta = {"short_description": "Critical SNR"}
    return table


def _annotate_astrometric_table(table: QTable, mode: str, refs: list[str]) -> None:
    """Attach descriptions, short descriptions, and references to astrometric detectability tables."""
    table.meta["references"] = refs
    table.meta["mode"] = mode

    is_marginalized = "delta_chi2_min" in table.colnames

    annotations: dict[str, tuple[str, str, list[str] | None]] = {
        "inclination": ("Orbital inclination", "Inclination", None),
        "node": ("Longitude of ascending node Ω", "Ascending node", None),
        "true_mass": ("Companion dynamical true mass", "True mass", None),
        "alpha_star": (
            "Astrometric wobble semi-amplitude of host star",
            "Wobble amplitude",
            ["Perryman2014"],
        ),
        "delta_chi2": (
            "Median signal Δχ² marginalized over ascending node Ω (Perryman et al. 2014, Ranalli et al. 2018)"
            if is_marginalized
            else "Astrometric signal chi-squared Δχ² = Σ (d_AL / sigma_AL)² (Perryman et al. 2014, Ranalli et al. 2018)",
            "Median Δχ²" if is_marginalized else "Signal Δχ²",
            ["Perryman2014", "Ranalli2018"],
        ),
        "delta_chi2_min": (
            "Minimum signal Δχ² across ascending node Ω",
            "Min Δχ²",
            None,
        ),
        "delta_chi2_max": (
            "Maximum signal Δχ² across ascending node Ω",
            "Max Δχ²",
            None,
        ),
        "snr": (
            "Median astrometric detection SNR sqrt(Δχ²) marginalized over Ω"
            if is_marginalized
            else "Astrometric detection signal-to-noise ratio sqrt(Δχ²)",
            "Median SNR" if is_marginalized else "Astrometric SNR",
            ["Perryman2014", "Ranalli2018"],
        ),
        "snr_min": ("Minimum detection SNR across ascending node Ω", "Min SNR", None),
        "snr_max": ("Maximum detection SNR across ascending node Ω", "Max SNR", None),
        "equivalent_sigma": (
            f"Median equivalent Gaussian significance in standard deviations (mode='{mode}')"
            if is_marginalized
            else f"Equivalent Gaussian significance in standard deviations (mode='{mode}')",
            "Significance (sigma)",
            ["Perryman2014", "Ranalli2018"],
        ),
        "equivalent_sigma_min": (
            "Minimum equivalent Gaussian significance across ascending node Ω",
            "Min sigma",
            None,
        ),
        "equivalent_sigma_max": (
            "Maximum equivalent Gaussian significance across ascending node Ω",
            "Max sigma",
            None,
        ),
        "ruwe": (
            "Median expected Gaia RUWE" if is_marginalized else "Expected Gaia RUWE",
            "Median RUWE" if is_marginalized else "RUWE",
            None,
        ),
        "ruwe_min": ("Minimum expected Gaia RUWE", "Min RUWE", None),
        "ruwe_max": ("Maximum expected Gaia RUWE", "Max RUWE", None),
        "excess_noise": (
            "Median expected astrometric excess noise"
            if is_marginalized
            else "Expected astrometric excess noise",
            "Median excess noise" if is_marginalized else "Excess noise",
            None,
        ),
        "excess_noise_min": (
            "Minimum expected astrometric excess noise",
            "Min excess noise",
            None,
        ),
        "excess_noise_max": (
            "Maximum expected astrometric excess noise",
            "Max excess noise",
            None,
        ),
    }

    for col, (desc, short, col_refs) in annotations.items():
        if col in table.colnames:
            table[col].info.description = desc
            meta: dict[str, str | list[str]] = {"short_description": short}
            if col_refs is not None:
                meta["references"] = col_refs
            table[col].info.meta = meta


@cites("Perryman2014", "Ranalli2018", "Lindegren2021_edr3_astro")
def gaia_astrometric_detectability(
    period: QuantityLike,
    m_planet: QuantityLike,
    m_star: QuantityLike = 1.0,
    distance: QuantityLike | None = None,
    parallax: QuantityLike | None = None,
    inclination: QuantityLike = 90.0,
    omega: QuantityLike = 0.0,
    node: QuantityLike | None = None,
    eccentricity: QuantityLike = 0.0,
    t0: QuantityLike = 0.0,
    times: QuantityLike | None = None,
    scan_angle: QuantityLike | None = None,
    ra: QuantityLike | None = None,
    dec: QuantityLike | None = None,
    sigma_al: QuantityLike = 34.0,
    is_msini: bool = False,
    duration_years: float = 5.5,
    step_seconds: float = 30.0,
    epoch_start: QuantityLike | None = None,
    n_node_samples: int = 36,
    mode: str = "rv",
    baseline_ruwe: float | None = None,
    baseline_excess_noise: QuantityLike | None = None,
    binary_separation: QuantityLike | None = None,
    binary_delta_mag: float | None = None,
    on_ccd: bool = True,
) -> QTable:
    """Calculate the Gaia astrometric detection significance (Delta chi^2, SNR, RUWE, and excess noise).

    Evaluates astrometric detectability using the Thiele-Innes orbital formalism and
    projected along-scan displacements (Perryman et al. 2014).

    When ``node`` is ``None`` (default), marginalizes over a uniform distribution of
    ascending nodes Omega in [0, 360 deg) and reports the median, min, and max detectability
    envelope. When ``node`` is provided, evaluates exact orientation(s). Supports
    vectorization over arrays of orbital inclinations.

    Allows specifying a baseline RUWE and excess noise from visual binary contamination
    (e.g., PSF blending / window allocation conflicts; Fabricius et al. 2021, El-Badry et al. 2021),
    either directly via ``baseline_ruwe`` / ``baseline_excess_noise``, or automatically
    via ``binary_separation`` and ``binary_delta_mag``.

    Parameters
    ----------
    period : QuantityLike
        Orbital period in days.
    m_planet : QuantityLike
        Companion mass in Jupiter masses (M_jup). If ``is_msini=True``, this is interpreted
        as minimum mass (m_p sin i) and converted to true mass m_p / sin(i).
    m_star : QuantityLike, optional
        Stellar mass in M_sun. Default is 1.0.
    distance : QuantityLike, optional
        Distance in pc.
    parallax : QuantityLike, optional
        Parallax in mas.
    inclination : QuantityLike, optional
        Orbital inclination in degrees. Can be scalar or array-like. Default is 90.0.
    omega : QuantityLike, optional
        Argument of periastron of the host star (omega_*) in degrees. Default is 0.0.
    node : QuantityLike or None, optional
        Longitude of ascending node (Omega) in degrees. If ``None`` (default), marginalizes
        over Omega in [0, 360 deg). If provided as a float or array, evaluates exact point(s).
    eccentricity : QuantityLike, optional
        Orbital eccentricity (0 <= e < 1). Default is 0.0.
    t0 : QuantityLike, optional
        Time of periastron passage in days. Default is 0.0.
    times : QuantityLike, optional
        Gaia scan times in days.
    scan_angle : QuantityLike, optional
        Gaia scan direction angles (psi) in radians.
    ra : QuantityLike, optional
        Target RA in degrees, used to generate scan times if not provided.
    dec : QuantityLike, optional
        Target Dec in degrees, used to generate scan times if not provided.
    sigma_al : QuantityLike, optional
        Single-transit along-scan astrometric uncertainty. Assumed to be in microarcseconds (uarcsec).
        Default is 34.0 uarcsec.
    is_msini : bool, optional
        Whether ``m_planet`` represents m_p sin(i). Default is False.
    duration_years : float, optional
        Mission duration in years if generating scan events from (ra, dec). Default is 5.5.
    step_seconds : float, optional
        Time grid resolution in seconds when generating scan events. Default is 30.0.
    epoch_start : QuantityLike, optional
        Mission start reference epoch in days (e.g., BJD 2456860.5). Default is None.
    n_node_samples : int, optional
        Number of uniform node samples in [0, 360 deg) used when ``node is None``. Default is 36.
    mode : {"rv", "blind", "joint"}, optional
        Detection significance context for computing ``equivalent_sigma``:
        - ``"rv"`` (default): Orbit pre-constrained by RVs (2 astrometric parameters: i, Omega).
        - ``"joint"``: Joint fit with free mass/amplitude (3 parameters).
        - ``"blind"``: Unconstrained astrometric blind search (7 parameters; Perryman et al. 2014).
    baseline_ruwe : float, optional
        Baseline RUWE from visual binary contamination or calibration error floor.
        Default is None, which resolves to 1.0 (clean single star) unless
        ``binary_separation``/``binary_delta_mag`` are given, in which case it resolves
        to the binary-contamination estimate.
    baseline_excess_noise : QuantityLike, optional
        Baseline astrometric excess noise from visual binary contamination.
        Default is None, which resolves the same way as ``baseline_ruwe``.
    binary_separation : QuantityLike, optional
        Angular separation of a visual binary companion. If provided with ``binary_delta_mag``,
        estimates baseline excess noise and RUWE using the empirical binary contamination
        model (Fabricius et al. 2021, El-Badry et al. 2021).
    binary_delta_mag : float, optional
        Magnitude difference Delta G of the visual binary companion.
    on_ccd : bool, optional
        Whether to calculate RUWE and excess noise at the official Gaia CCD level
        (assumes ~9 Astrometric Field CCDs per transit; default is True).
        If False, calculates at the simplified transit level.

    Returns
    -------
    astropy.table.QTable
        Table containing evaluated orbital parameters and detectability metrics.
        If ``node is None`` (marginalized):
        - ``inclination``: Orbital inclination (deg)
        - ``true_mass``: Companion dynamical true mass (M_jup)
        - ``alpha_star``: Astrometric wobble semi-amplitude of host star (uarcsec)
        - ``delta_chi2``: Median astrometric signal Delta chi^2
        - ``delta_chi2_min``: Minimum signal Delta chi^2 across Omega
        - ``delta_chi2_max``: Maximum signal Delta chi^2 across Omega
        - ``snr``: Median astrometric detection SNR = sqrt(Delta chi^2)
        - ``snr_min``: Minimum detection SNR across Omega
        - ``snr_max``: Maximum detection SNR across Omega
        - ``equivalent_sigma``: Median equivalent Gaussian significance (standard deviations)
        - ``equivalent_sigma_min``: Minimum equivalent Gaussian significance
        - ``equivalent_sigma_max``: Maximum equivalent Gaussian significance
        - ``ruwe``: Median expected Gaia RUWE (incorporating baseline binary contamination)
        - ``ruwe_min``: Minimum expected Gaia RUWE
        - ``ruwe_max``: Maximum expected Gaia RUWE
        - ``excess_noise``: Median expected astrometric excess noise (uarcsec)
        - ``excess_noise_min``: Minimum expected astrometric excess noise (uarcsec)
        - ``excess_noise_max``: Maximum expected astrometric excess noise (uarcsec)

        If ``node`` is provided:
        - ``inclination``: Orbital inclination (deg)
        - ``node``: Longitude of ascending node (deg)
        - ``true_mass``: Companion dynamical true mass (M_jup)
        - ``alpha_star``: Astrometric wobble semi-amplitude of host star (uarcsec)
        - ``delta_chi2``: Astrometric signal Delta chi^2
        - ``snr``: Astrometric detection SNR
        - ``equivalent_sigma``: Equivalent Gaussian significance in standard deviations
        - ``ruwe``: Expected Gaia RUWE
        - ``excess_noise``: Expected astrometric excess noise (uarcsec)

    References
    ----------
    Perryman et al. (2014), ApJ, 797, 14.
    Lindegren et al. (2021), A&A, 649, A2.
    Fabricius et al. (2021), A&A, 649, A5.
    El-Badry et al. (2021), MNRAS, 506, 2269.
    """
    if times is None or scan_angle is None:
        if ra is None or dec is None:
            raise ValueError("Either (times, scan_angle) or (ra, dec) must be provided.")
        events = gaia_scan_events(
            ra,
            dec,
            duration_years=duration_years,
            step_seconds=step_seconds,
            epoch_start=epoch_start,
        )
        times = events.times
        scan_angle = events.scan_angles

    m_p = u.Quantity(m_planet, "M_jup")
    sigma = u.Quantity(sigma_al, "uarcsec")
    sigma_val = sigma.to_value("uarcsec")
    inc_arr = np.atleast_1d(np.asarray(u.Quantity(inclination, "deg").to_value("deg"), dtype=float))

    t_days = times.to_value("day")
    p_days = u.Quantity(period, "day").to_value("day")
    t0_days = u.Quantity(t0, "day").to_value("day")
    e_val = float(
        eccentricity.to_value(u.dimensionless_unscaled)
        if isinstance(eccentricity, u.Quantity)
        else eccentricity
    )
    mean_motion = 2 * np.pi / p_days
    mean_anom = (mean_motion * (t_days - t0_days)) % (2 * np.pi)
    ecc_anom = solve_kepler(mean_anom, e_val)
    orb_x = np.cos(ecc_anom) - e_val
    orb_y = np.sqrt(1 - e_val**2) * np.sin(ecc_anom)

    psi_rad = scan_angle.to_value("rad")
    sin_psi = np.sin(psi_rad)
    cos_psi = np.cos(psi_rad)

    rad_omega = np.radians(u.Quantity(omega, "deg").to_value("deg"))
    n_transits = len(times)

    # If binary parameters are provided, compute binary-induced baseline contamination
    # (only fills in baseline_ruwe / baseline_excess_noise the caller left unset).
    if binary_separation is not None and binary_delta_mag is not None:
        bin_contam = binary_astrometric_contamination(
            separation=binary_separation,
            delta_mag=binary_delta_mag,
            sigma_al=sigma,
            n_transits=n_transits,
            on_ccd=on_ccd,
        )
        if baseline_ruwe is None:
            baseline_ruwe = bin_contam.ruwe
        if baseline_excess_noise is None:
            baseline_excess_noise = bin_contam.excess_noise

    baseline_ruwe = 1.0 if baseline_ruwe is None else baseline_ruwe
    baseline_excess_noise = 0.0 if baseline_excess_noise is None else baseline_excess_noise

    if node is None:
        sample_nodes = np.linspace(0.0, 360.0, n_node_samples, endpoint=False)
        rad_nodes = np.radians(sample_nodes)

        true_masses: list[float] = []
        alpha_stars: list[float] = []
        chi2_medians: list[float] = []
        chi2_mins: list[float] = []
        chi2_maxs: list[float] = []
        snr_medians: list[float] = []
        snr_mins: list[float] = []
        snr_maxs: list[float] = []

        for inc_val in inc_arr:
            sin_i = np.sin(np.radians(inc_val))
            if is_msini and np.isclose(sin_i, 0.0):
                true_masses.append(np.nan)
                alpha_stars.append(np.nan)
                chi2_medians.append(np.nan)
                chi2_mins.append(np.nan)
                chi2_maxs.append(np.nan)
                snr_medians.append(np.nan)
                snr_mins.append(np.nan)
                snr_maxs.append(np.nan)
                continue

            true_m = (m_p / sin_i) if is_msini else m_p

            alpha_val = astrometric_semi_amplitude(
                m_planet=true_m,
                m_star=m_star,
                distance=distance,
                parallax=parallax,
                period=period,
            )
            alpha_uarcsec = alpha_val.to_value("uarcsec")

            val_a, val_b, val_f, val_g = _thiele_innes_coefficients(
                alpha=alpha_uarcsec,
                omega_rad=rad_omega,
                node_rad=rad_nodes,
                inc_rad=np.radians(inc_val),
            )

            coeff_x = np.outer(val_b, sin_psi) + np.outer(val_a, cos_psi)
            coeff_y = np.outer(val_g, sin_psi) + np.outer(val_f, cos_psi)
            d_al_matrix = coeff_x * orb_x + coeff_y * orb_y
            chi2_node_arr = np.sum((d_al_matrix / sigma_val) ** 2, axis=1)
            snr_node_arr = np.sqrt(chi2_node_arr)

            true_masses.append(true_m.to_value("M_jup"))
            alpha_stars.append(alpha_uarcsec)
            chi2_medians.append(float(np.median(chi2_node_arr)))
            chi2_mins.append(float(np.min(chi2_node_arr)))
            chi2_maxs.append(float(np.max(chi2_node_arr)))
            snr_medians.append(float(np.median(snr_node_arr)))
            snr_mins.append(float(np.min(snr_node_arr)))
            snr_maxs.append(float(np.max(snr_node_arr)))

        mask = np.isnan(chi2_medians)
        has_masked = np.any(mask)
        clean_chi2 = np.nan_to_num(chi2_medians, nan=0.0)
        clean_mins = np.nan_to_num(chi2_mins, nan=0.0)
        clean_maxs = np.nan_to_num(chi2_maxs, nan=0.0)

        equiv_sigma = delta_chi2_to_sigma(clean_chi2, mode=mode)
        equiv_sigma_min = delta_chi2_to_sigma(clean_mins, mode=mode)
        equiv_sigma_max = delta_chi2_to_sigma(clean_maxs, mode=mode)

        ruwe_medians = expected_ruwe(
            clean_chi2, n_transits=n_transits, baseline_ruwe=baseline_ruwe, on_ccd=on_ccd
        )
        ruwe_mins = expected_ruwe(
            clean_mins, n_transits=n_transits, baseline_ruwe=baseline_ruwe, on_ccd=on_ccd
        )
        ruwe_maxs = expected_ruwe(
            clean_maxs, n_transits=n_transits, baseline_ruwe=baseline_ruwe, on_ccd=on_ccd
        )

        excess_medians = expected_astrometric_excess_noise(
            clean_chi2,
            n_transits=n_transits,
            sigma_al=sigma,
            baseline_excess_noise=baseline_excess_noise,
            on_ccd=on_ccd,
        )
        excess_mins = expected_astrometric_excess_noise(
            clean_mins,
            n_transits=n_transits,
            sigma_al=sigma,
            baseline_excess_noise=baseline_excess_noise,
            on_ccd=on_ccd,
        )
        excess_maxs = expected_astrometric_excess_noise(
            clean_maxs,
            n_transits=n_transits,
            sigma_al=sigma,
            baseline_excess_noise=baseline_excess_noise,
            on_ccd=on_ccd,
        )

        def _wrap_qty(vals: list[float] | np.ndarray, unit_str: str) -> u.Quantity:
            clean_vals = np.nan_to_num(vals, nan=0.0)
            q = u.Quantity(clean_vals, unit_str)
            return Masked(q, mask=mask) if has_masked else q

        def _wrap_float(vals: list[float] | np.ndarray) -> np.ndarray:
            arr = np.asarray(vals, dtype=float)
            return np.ma.masked_where(mask, arr) if has_masked else arr

        table = QTable(
            {
                "inclination": inc_arr * u.deg,
                "true_mass": _wrap_qty(true_masses, "M_jup"),
                "alpha_star": _wrap_qty(alpha_stars, "uarcsec"),
                "delta_chi2": _wrap_float(chi2_medians),
                "delta_chi2_min": _wrap_float(chi2_mins),
                "delta_chi2_max": _wrap_float(chi2_maxs),
                "snr": _wrap_float(snr_medians),
                "snr_min": _wrap_float(snr_mins),
                "snr_max": _wrap_float(snr_maxs),
                "equivalent_sigma": _wrap_float(equiv_sigma),
                "equivalent_sigma_min": _wrap_float(equiv_sigma_min),
                "equivalent_sigma_max": _wrap_float(equiv_sigma_max),
                "ruwe": _wrap_float(ruwe_medians),
                "ruwe_min": _wrap_float(ruwe_mins),
                "ruwe_max": _wrap_float(ruwe_maxs),
                "excess_noise": Masked(excess_medians, mask=mask) if has_masked else excess_medians,
                "excess_noise_min": Masked(excess_mins, mask=mask) if has_masked else excess_mins,
                "excess_noise_max": Masked(excess_maxs, mask=mask) if has_masked else excess_maxs,
            }
        )

        refs = ["Perryman2014", "Ranalli2018", "Lindegren2021_edr3_astro"]
        if baseline_ruwe > 1.0 or binary_separation is not None:
            refs.extend(["Fabricius2021_edr3_val", "ElBadry2021_binaries"])
        _annotate_astrometric_table(table, mode=mode, refs=refs)
        return table

    node_arr = np.atleast_1d(np.asarray(u.Quantity(node, "deg").to_value("deg"), dtype=float))

    if (
        inc_arr.ndim == 1
        and node_arr.ndim == 1
        and len(inc_arr) > 1
        and len(node_arr) > 1
        and len(inc_arr) != len(node_arr)
    ):
        inc_grid, node_grid = np.meshgrid(inc_arr, node_arr, indexing="ij")
        inc_flat = inc_grid.ravel()
        node_flat = node_grid.ravel()
    else:
        inc_bcast, node_bcast = np.broadcast_arrays(inc_arr, node_arr)
        inc_flat = inc_bcast.ravel()
        node_flat = node_bcast.ravel()

    true_masses_exact: list[float] = []
    alpha_stars_exact: list[float] = []

    for inc_val in inc_flat:
        sin_i = np.sin(np.radians(inc_val))
        if is_msini and np.isclose(sin_i, 0.0):
            true_masses_exact.append(np.nan)
            alpha_stars_exact.append(np.nan)
            continue
        true_m = (m_p / sin_i) if is_msini else m_p
        alpha_val = astrometric_semi_amplitude(
            m_planet=true_m,
            m_star=m_star,
            distance=distance,
            parallax=parallax,
            period=period,
        )
        true_masses_exact.append(true_m.to_value("M_jup"))
        alpha_stars_exact.append(alpha_val.to_value("uarcsec"))

    mask_exact = np.isnan(alpha_stars_exact)
    has_masked_exact = np.any(mask_exact)

    rad_inc_flat = np.radians(inc_flat)
    rad_node_flat = np.radians(node_flat)
    alpha_arr = np.nan_to_num(alpha_stars_exact, nan=0.0)

    val_a, val_b, val_f, val_g = _thiele_innes_coefficients(
        alpha=alpha_arr,
        omega_rad=rad_omega,
        node_rad=rad_node_flat,
        inc_rad=rad_inc_flat,
    )

    coeff_x = np.outer(val_b, sin_psi) + np.outer(val_a, cos_psi)
    coeff_y = np.outer(val_g, sin_psi) + np.outer(val_f, cos_psi)
    d_al_matrix = coeff_x * orb_x + coeff_y * orb_y
    chi2_exact_arr = np.sum((d_al_matrix / sigma_val) ** 2, axis=1)
    snr_exact_arr = np.sqrt(chi2_exact_arr)
    equiv_sigma_exact = delta_chi2_to_sigma(chi2_exact_arr, mode=mode)
    ruwe_exact = expected_ruwe(
        chi2_exact_arr,
        n_transits=n_transits,
        baseline_ruwe=baseline_ruwe,
        on_ccd=on_ccd,
    )
    excess_exact = expected_astrometric_excess_noise(
        chi2_exact_arr,
        n_transits=n_transits,
        sigma_al=sigma,
        baseline_excess_noise=baseline_excess_noise,
        on_ccd=on_ccd,
    )

    def _wrap_qty_exact(vals: list[float] | np.ndarray, unit_str: str) -> u.Quantity:
        clean_vals = np.nan_to_num(vals, nan=0.0)
        q = u.Quantity(clean_vals, unit_str)
        return Masked(q, mask=mask_exact) if has_masked_exact else q

    def _wrap_float_exact(vals: list[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(vals, dtype=float)
        return np.ma.masked_where(mask_exact, arr) if has_masked_exact else arr

    table = QTable(
        {
            "inclination": inc_flat * u.deg,
            "node": node_flat * u.deg,
            "true_mass": _wrap_qty_exact(true_masses_exact, "M_jup"),
            "alpha_star": _wrap_qty_exact(alpha_stars_exact, "uarcsec"),
            "delta_chi2": _wrap_float_exact(chi2_exact_arr),
            "snr": _wrap_float_exact(snr_exact_arr),
            "equivalent_sigma": _wrap_float_exact(equiv_sigma_exact),
            "ruwe": _wrap_float_exact(ruwe_exact),
            "excess_noise": Masked(excess_exact, mask=mask_exact)
            if has_masked_exact
            else excess_exact,
        }
    )

    refs = ["Perryman2014", "Ranalli2018", "Lindegren2021_edr3_astro"]
    if baseline_ruwe > 1.0 or binary_separation is not None:
        refs.extend(["Fabricius2021_edr3_val", "ElBadry2021_binaries"])
    _annotate_astrometric_table(table, mode=mode, refs=refs)
    return table
