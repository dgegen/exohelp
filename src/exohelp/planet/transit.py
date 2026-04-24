import astropy.units as u
import numpy as np
from astropy.table import QTable

from ..kepler import keplers_third_law
from ..type import QuantityLike

__all__ = [
    "a_over_r_star",
    "geometric_occultation_probability",
    "geometric_transit_probability",
    "impact_parameter",
    "orbital_inclination",
    "secondary_eclipse_timing_offset",
    "transit_depth",
    "transit_duration_flat",
    "transit_duration_ingress",
    "transit_duration_total",
    "transit_quantities",
]


def impact_parameter(
    semi_major_axis: QuantityLike,
    r_star: QuantityLike = 1.0,
    cos_inclination: QuantityLike = 0.0,
    eccentricity: float | np.ndarray = 0.0,
    omega: QuantityLike = 90.0,
) -> np.ndarray:
    """Compute the transit impact parameter b = (a/R★) cos i * (1 - e²) / (1 + e sin ω).

    Parameters
    ----------
    semi_major_axis : QuantityLike
        Orbital semi-major axis. Assumed to be in AU if no unit is given.
    r_star : QuantityLike
        Stellar radius. Assumed to be in Solar radii if no unit is given.
    cos_inclination : float or array-like
        Cosine of the orbital inclination. Default is 0 (edge-on).
    eccentricity : float or array-like
        Orbital eccentricity. Default is 0 (circular).
    omega : QuantityLike
        Argument of periastron in degrees. Default is 90°.

    Returns
    -------
    b : ndarray
        Impact parameter (dimensionless).

    References
    ----------
    Winn, J. N. (2010), in Exoplanets, edited by S. Seager. Published by University of Arizona Press, Tucson, AZ, 2010, 526 pp. ISBN 978-0-8165-2945-2., p.55-77
    https://ui.adsabs.harvard.edu/abs/2010exop.book...55W/abstract
    """
    semi_major_axis = u.Quantity(semi_major_axis, "AU")
    r_star = u.Quantity(r_star, "R_sun")
    omega_rad = omega.to(u.rad).value if isinstance(omega, u.Quantity) else np.deg2rad(omega)
    ecc_factor = (1 - eccentricity**2) / (1 + eccentricity * np.sin(omega_rad))
    return (semi_major_axis / r_star).decompose().value * np.asarray(cos_inclination) * ecc_factor


def orbital_inclination(
    semi_major_axis: QuantityLike,
    r_star: QuantityLike = 1.0,
    b: float | np.ndarray = 0.0,
    eccentricity: float | np.ndarray = 0.0,
    omega: QuantityLike = 90.0,
) -> u.Quantity:
    """Compute the orbital inclination from the impact parameter.

        cos i = b * (R★/a) * (1 + e sin ω) / (1 - e²)

    Parameters
    ----------
    semi_major_axis : QuantityLike
        Orbital semi-major axis. Assumed to be in AU if no unit is given.
    r_star : QuantityLike
        Stellar radius. Assumed to be in Solar radii if no unit is given.
    b : float or array-like
        Sky-projected impact parameter at mid-transit. Default is 0 (edge-on).
    eccentricity : float or array-like
        Orbital eccentricity. Default is 0 (circular).
    omega : QuantityLike
        Argument of periastron in degrees. Default is 90°.

    Returns
    -------
    inclination : Quantity
        Orbital inclination in degrees.

    References
    ----------
    Winn, J. N. (2010), in Exoplanets, edited by S. Seager. Published by University of Arizona Press, Tucson, AZ, 2010, 526 pp. ISBN 978-0-8165-2945-2., p.55-77
    https://ui.adsabs.harvard.edu/abs/2010exop.book...55W/abstract
    """
    semi_major_axis = u.Quantity(semi_major_axis, "AU")
    r_star = u.Quantity(r_star, "R_sun")
    omega_rad = omega.to(u.rad).value if isinstance(omega, u.Quantity) else np.deg2rad(omega)
    ecc_factor = (1 + eccentricity * np.sin(omega_rad)) / (1 - eccentricity**2)
    cos_i = b * (r_star / semi_major_axis).decompose().value * ecc_factor
    return (np.arccos(cos_i) * u.rad).to(u.deg)  # type: ignore[return-value]


def _chord_duration(
    period: u.Quantity,
    k: float,
    r_star_over_a: float,
    b: float | np.ndarray,
    eccentricity: float | np.ndarray = 0.0,
    omega_rad: float | np.ndarray = np.pi / 2,
) -> u.Quantity:
    """Duration of a chord across the stellar disk with half-width (1 ± k).

    T = (P / π) arcsin( (R★/a) √( ((1 ± k)² - b²) / (1 - (b R★/a)²) ) )
        * √(1 - e²) / (1 + e sin ω)
    """
    cos_i = b * r_star_over_a
    argument = r_star_over_a * np.sqrt(((1 + k) ** 2 - b**2) / (1 - cos_i**2))
    ecc_factor = np.sqrt(1 - eccentricity**2) / (1 + eccentricity * np.sin(omega_rad))
    return (np.arcsin(argument) / np.pi) * period.to("hour") * u.Quantity(ecc_factor)  # type: ignore[return-value]


def transit_duration_total(
    period: QuantityLike,
    r_planet: QuantityLike = 1.0,
    r_star: QuantityLike = 1.0,
    m_star: QuantityLike = 1.0,
    b: float | np.ndarray = 0.0,
    eccentricity: float | np.ndarray = 0.0,
    omega: QuantityLike = 90.0,
) -> u.Quantity:
    """Compute the total transit duration T₁₄ (first to fourth contact).

    Parameters
    ----------
    period : QuantityLike
        Orbital period. Assumed to be in days if no unit is given.
    r_planet : QuantityLike
        Planet radius. Assumed to be in Earth radii if no unit is given.
    r_star : QuantityLike
        Stellar radius. Assumed to be in Solar radii if no unit is given.
    m_star : QuantityLike
        Stellar mass. Assumed to be in Solar masses if no unit is given.
    b : float or array-like
        Sky-projected impact parameter at mid-transit. Default is 0 (central transit).
        If derived from inclination and eccentricity, use
        b = (a/R★) cos i * (1 - e²) / (1 + e sin ω).
    eccentricity : float or array-like
        Orbital eccentricity. Default is 0 (circular).
    omega : QuantityLike
        Argument of periastron in degrees. Default is 90°.

    Returns
    -------
    duration : Quantity
        Transit duration in hours.

    Notes
    -----
    Derived from the chord length across the stellar disk:

        T₁₄ = (P / π) arcsin( (R★/a) √( ((1 + k)² - b²) / (1 - cos²i) ) )
              * √(1 - e²) / (1 + e sin ω)

    where k = Rp/R★ and b = (a/R★) cos i.

    References
    ----------
    Winn, J. N. (2010), in Exoplanets, edited by S. Seager. Published by University of Arizona Press, Tucson, AZ, 2010, 526 pp. ISBN 978-0-8165-2945-2., p.55-77
    https://ui.adsabs.harvard.edu/abs/2010exop.book...55W/abstract

    Examples
    --------
    >>> transit_duration_total(period=365.25).round(1).to('hour').value
    13.1
    """
    period = u.Quantity(period, "day")
    r_planet = u.Quantity(r_planet, "R_earth")
    r_star = u.Quantity(r_star, "R_sun")
    m_star = u.Quantity(m_star, "M_sun")
    omega_rad = omega.to(u.rad).value if isinstance(omega, u.Quantity) else np.deg2rad(omega)

    a = keplers_third_law(period=period, mass=m_star)
    k = (r_planet.to("R_sun") / r_star).decompose().value
    r_star_over_a = (r_star.to("AU") / a).decompose().value
    return _chord_duration(period, k, r_star_over_a, b, eccentricity, omega_rad)


def transit_duration_flat(
    period: QuantityLike,
    r_planet: QuantityLike = 1.0,
    r_star: QuantityLike = 1.0,
    m_star: QuantityLike = 1.0,
    b: float | np.ndarray = 0.0,
    eccentricity: float | np.ndarray = 0.0,
    omega: QuantityLike = 90.0,
) -> u.Quantity:
    """Compute the flat-bottom transit duration T₂₃ (second to third contact).

    Returns 0 for grazing transits where b > 1 - Rp/R★.

    Parameters
    ----------
    period : QuantityLike
        Orbital period. Assumed to be in days if no unit is given.
    r_planet : QuantityLike
        Planet radius. Assumed to be in Earth radii if no unit is given.
    r_star : QuantityLike
        Stellar radius. Assumed to be in Solar radii if no unit is given.
    m_star : QuantityLike
        Stellar mass. Assumed to be in Solar masses if no unit is given.
    b : float or array-like
        Sky-projected impact parameter at mid-transit. Default is 0 (central transit).
        If derived from inclination and eccentricity, use
        b = (a/R★) cos i * (1 - e²) / (1 + e sin ω).
    eccentricity : float or array-like
        Orbital eccentricity. Default is 0 (circular).
    omega : QuantityLike
        Argument of periastron in degrees. Default is 90°.

    Returns
    -------
    duration : Quantity
        Flat-bottom duration in hours.

    Notes
    -----
    Uses the inner chord with half-width (1 - k):

        T₂₃ = (P / π) arcsin( (R★/a) √( ((1 - k)² - b²) / (1 - cos²i) ) )
              * √(1 - e²) / (1 + e sin ω)

    References
    ----------
    Winn, J. N. (2010), in Exoplanets, edited by S. Seager. Published by University of Arizona Press, Tucson, AZ, 2010, 526 pp. ISBN 978-0-8165-2945-2., p.55-77
    https://ui.adsabs.harvard.edu/abs/2010exop.book...55W/abstract
    """
    period = u.Quantity(period, "day")
    r_planet = u.Quantity(r_planet, "R_earth")
    r_star = u.Quantity(r_star, "R_sun")
    m_star = u.Quantity(m_star, "M_sun")
    omega_rad = omega.to(u.rad).value if isinstance(omega, u.Quantity) else np.deg2rad(omega)

    a = keplers_third_law(period=period, mass=m_star)
    k = (r_planet.to("R_sun") / r_star).decompose().value
    r_star_over_a = (r_star.to("AU") / a).decompose().value

    b_arr = np.asarray(b, dtype=float)
    grazing = b_arr > 1 - k
    b_safe = np.where(grazing, 0.0, b_arr)
    duration = _chord_duration(period, -k, r_star_over_a, b_safe, eccentricity, omega_rad)
    return np.where(grazing, 0.0, duration.to("hour").value) * u.hour  # type: ignore[return-value]


def transit_duration_ingress(
    period: QuantityLike,
    r_planet: QuantityLike = 1.0,
    r_star: QuantityLike = 1.0,
    m_star: QuantityLike = 1.0,
    b: float | np.ndarray = 0.0,
    eccentricity: float | np.ndarray = 0.0,
    omega: QuantityLike = 90.0,
) -> u.Quantity:
    """Compute the ingress (or egress) duration T₁₂ = (T₁₄ - T₂₃) / 2.

    Ingress and egress are always equal: the eccentricity factor √(1-e²)/(1+e sin ω)
    is identical in T₁₄ and T₂₃ and cancels in the difference.

    Parameters
    ----------
    period : QuantityLike
        Orbital period. Assumed to be in days if no unit is given.
    r_planet : QuantityLike
        Planet radius. Assumed to be in Earth radii if no unit is given.
    r_star : QuantityLike
        Stellar radius. Assumed to be in Solar radii if no unit is given.
    m_star : QuantityLike
        Stellar mass. Assumed to be in Solar masses if no unit is given.
    b : float or array-like
        Sky-projected impact parameter at mid-transit. Default is 0 (central transit).
        If derived from inclination and eccentricity, use
        b = (a/R★) cos i * (1 - e²) / (1 + e sin ω).
    eccentricity : float or array-like
        Orbital eccentricity. Default is 0 (circular).
    omega : QuantityLike
        Argument of periastron in degrees. Default is 90°.

    Returns
    -------
    duration : Quantity
        Ingress duration in hours.

    References
    ----------
    Winn, J. N. (2010), in Exoplanets, edited by S. Seager. Published by University of Arizona Press, Tucson, AZ, 2010, 526 pp. ISBN 978-0-8165-2945-2., p.55-77
    https://ui.adsabs.harvard.edu/abs/2010exop.book...55W/abstract
    """
    t_total = transit_duration_total(period, r_planet, r_star, m_star, b, eccentricity, omega)
    t_flat = transit_duration_flat(period, r_planet, r_star, m_star, b, eccentricity, omega)
    return (t_total - t_flat) / 2.0  # type: ignore[return-value]


def transit_depth(radius_ratio: float | np.ndarray, b: float | np.ndarray = 0.0) -> np.ndarray:
    """Compute the transit depth (fractional flux loss) as a function of impact parameter.

    Handles full, grazing, and no-transit geometries analytically.

    Parameters
    ----------
    radius_ratio : float or array-like
        Planet-to-star radius ratio Rp/R★.
    b : float or array-like
        Sky-projected impact parameter at mid-transit. Default is 0 (central transit).
        If derived from inclination and eccentricity, use
        b = (a/R★) cos i * (1 - e²) / (1 + e sin ω).

    Returns
    -------
    depth : ndarray
        Fractional flux loss (dimensionless, between 0 and 1).

    References
    ----------
    Mandel, K. & Agol, E. (2002), ApJL, 580, L171.
    https://doi.org/10.1086/345520

    Seager, S. & Mallén-Ornelas, G. (2003), ApJ, 585, 1038.
    https://doi.org/10.1086/346105
    """
    k = np.asarray(radius_ratio, dtype=float)
    b = np.asarray(b, dtype=float)
    scalar = k.ndim == 0 and b.ndim == 0

    k, b = np.broadcast_arrays(k, b)
    depth = np.zeros_like(k)

    full = b <= 1 - k
    grazing = (~full) & (b < 1 + k)

    depth[full] = k[full] ** 2

    kg, bg = k[grazing], b[grazing]
    k0 = np.arccos((kg**2 + bg**2 - 1) / (2 * bg * kg))
    k1 = np.arccos((1 - kg**2 + bg**2) / (2 * bg))
    depth[grazing] = (
        1 / np.pi * (kg**2 * k0 + k1 - np.sqrt((4 * bg**2 - (1 + bg**2 - kg**2) ** 2) / 4))
    )

    return depth.item() if scalar else depth


def geometric_transit_probability(
    period: QuantityLike,
    r_star: QuantityLike = 1.0,
    m_star: QuantityLike = 1.0,
    eccentricity: float | np.ndarray = 0.0,
    omega: QuantityLike = 90.0,
) -> np.ndarray:
    """Compute the geometric transit probability P_tr = (R★ / a) * (1 + e sin ω) / (1 - e²).

    Parameters
    ----------
    period : QuantityLike
        Orbital period. Assumed to be in days if no unit is given.
    r_star : QuantityLike
        Stellar radius. Assumed to be in Solar radii if no unit is given.
    m_star : QuantityLike
        Stellar mass. Assumed to be in Solar masses if no unit is given.
    eccentricity : float or array-like
        Orbital eccentricity. Default is 0 (circular).
    omega : QuantityLike
        Argument of periastron in degrees. Default is 90°. When ω is unknown,
        set ``eccentricity`` only and use the orbit-averaged approximation
        1/(1 - e²) by passing ``omega=0``.

    Returns
    -------
    probability : ndarray
        Geometric transit probability (dimensionless, between 0 and 1).

    Notes
    -----
    The eccentricity correction (1 + e sin ω)/(1 - e²) equals 1 for circular
    orbits. Averaged uniformly over ω, it reduces to 1/(1 - e²).

    References
    ----------
    Winn, J. N. (2010), in Exoplanets, edited by S. Seager. Published by University of Arizona Press, Tucson, AZ, 2010, 526 pp. ISBN 978-0-8165-2945-2., p.55-77
    https://ui.adsabs.harvard.edu/abs/2010exop.book...55W/abstract
    """
    period = u.Quantity(period, "day")
    r_star = u.Quantity(r_star, "R_sun")
    m_star = u.Quantity(m_star, "M_sun")
    omega_rad = omega.to(u.rad).value if isinstance(omega, u.Quantity) else np.deg2rad(omega)

    a = keplers_third_law(period=period, mass=m_star)
    ecc_factor = (1 + eccentricity * np.sin(omega_rad)) / (1 - eccentricity**2)
    return (r_star.to("AU") / a).decompose().value * ecc_factor


def geometric_occultation_probability(
    period: QuantityLike,
    r_star: QuantityLike = 1.0,
    m_star: QuantityLike = 1.0,
    eccentricity: float | np.ndarray = 0.0,
    omega: QuantityLike = 90.0,
) -> np.ndarray:
    """Compute the geometric occultation (secondary eclipse) probability.

        P_occ = (R★ / a) * (1 - e sin omega) / (1 - e²)

    The occultation occurs on the opposite side of the orbit from the transit,
    so the eccentricity correction flips sign relative to
    `geometric_transit_probability`.

    Parameters
    ----------
    period : QuantityLike
        Orbital period. Assumed to be in days if no unit is given.
    r_star : QuantityLike
        Stellar radius. Assumed to be in Solar radii if no unit is given.
    m_star : QuantityLike
        Stellar mass. Assumed to be in Solar masses if no unit is given.
    eccentricity : float or array-like
        Orbital eccentricity. Default is 0 (circular).
    omega : QuantityLike
        Argument of periastron in degrees. Default is 90°.

    Returns
    -------
    probability : ndarray
        Geometric occultation probability (dimensionless, between 0 and 1).

    Notes
    -----
    For a circular orbit P_occ = P_tr = R★/a. For eccentric orbits the two
    probabilities differ; a planet with ω = 90° (periastron at transit) has
    an enhanced transit probability but a reduced occultation probability.

    References
    ----------
    Winn, J. N. (2010), in Exoplanets, edited by S. Seager. Published by University of Arizona Press, Tucson, AZ, 2010, 526 pp. ISBN 978-0-8165-2945-2., p.55-77
    https://ui.adsabs.harvard.edu/abs/2010exop.book...55W/abstract
    """
    period = u.Quantity(period, "day")
    r_star = u.Quantity(r_star, "R_sun")
    m_star = u.Quantity(m_star, "M_sun")
    omega_rad = omega.to(u.rad).value if isinstance(omega, u.Quantity) else np.deg2rad(omega)

    a = keplers_third_law(period=period, mass=m_star)
    ecc_factor = (1 - eccentricity * np.sin(omega_rad)) / (1 - eccentricity**2)
    return (r_star.to("AU") / a).decompose().value * ecc_factor


def a_over_r_star(
    period: QuantityLike,
    r_star: QuantityLike = 1.0,
    m_star: QuantityLike = 1.0,
) -> np.ndarray:
    """Compute the ratio of the orbital semi-major axis to the stellar radius, a/R★.

    This is a fundamental transit observable, directly measurable from the
    lightcurve shape and used as a fitting parameter in transit models.

    Parameters
    ----------
    period : QuantityLike
        Orbital period. Assumed to be in days if no unit is given.
    r_star : QuantityLike
        Stellar radius. Assumed to be in Solar radii if no unit is given.
    m_star : QuantityLike
        Stellar mass. Assumed to be in Solar masses if no unit is given.

    Returns
    -------
    a_over_r_star : ndarray
        Dimensionless ratio a/R★.

    References
    ----------
    Winn, J. N. (2010), in Exoplanets, edited by S. Seager. Published by University of Arizona Press, Tucson, AZ, 2010, 526 pp. ISBN 978-0-8165-2945-2., p.55-77
    https://ui.adsabs.harvard.edu/abs/2010exop.book...55W/abstract
    """
    period = u.Quantity(period, "day")
    r_star = u.Quantity(r_star, "R_sun")
    m_star = u.Quantity(m_star, "M_sun")

    a = keplers_third_law(period=period, mass=m_star)
    return (a / r_star.to("AU")).decompose().value


def secondary_eclipse_timing_offset(
    period: QuantityLike,
    eccentricity: float | np.ndarray,
    omega: QuantityLike = 90.0,
) -> u.Quantity:
    """Compute the timing offset of the secondary eclipse from phase 0.5.

        Delta_t = P * e * cos(omega) / (pi * sqrt(1 - e^2))

    For a circular orbit the secondary eclipse falls exactly at phase 0.5.
    For an eccentric orbit it is shifted earlier (omega < 90°) or later
    (omega > 90°), providing a direct observable constraint on e*cos(omega).

    Parameters
    ----------
    period : QuantityLike
        Orbital period. Assumed to be in days if no unit is given.
    eccentricity : float or array-like
        Orbital eccentricity.
    omega : QuantityLike
        Argument of periastron in degrees. Default is 90°.

    Returns
    -------
    offset : Quantity
        Timing offset in hours. Positive means the eclipse is late
        (occurs after phase 0.5).

    References
    ----------
    Winn, J. N. (2010), in Exoplanets, edited by S. Seager. Published by University of Arizona Press, Tucson, AZ, 2010, 526 pp. ISBN 978-0-8165-2945-2., p.55-77
    https://ui.adsabs.harvard.edu/abs/2010exop.book...55W/abstract
    """
    period = u.Quantity(period, "day")
    omega_rad = omega.to(u.rad).value if isinstance(omega, u.Quantity) else np.deg2rad(omega)

    return (
        period.to("hour")
        * eccentricity
        * np.cos(omega_rad)
        / (np.pi * np.sqrt(1 - eccentricity**2))
    )


def transit_quantities(
    period: QuantityLike,
    r_planet: QuantityLike = 1.0,
    r_star: QuantityLike = 1.0,
    m_star: QuantityLike = 1.0,
    b: float | np.ndarray = 0.0,
    eccentricity: float | np.ndarray = 0.0,
    omega: QuantityLike = 90.0,
) -> QTable:
    """Compute all transit geometry quantities from standard fit parameters.

    Parameters
    ----------
    period : QuantityLike
        Orbital period. Assumed to be in days if no unit is given.
    r_planet : QuantityLike
        Planet radius. Assumed to be in Earth radii if no unit is given.
    r_star : QuantityLike
        Stellar radius. Assumed to be in Solar radii if no unit is given.
    m_star : QuantityLike
        Stellar mass. Assumed to be in Solar masses if no unit is given.
    b : float or array-like
        Sky-projected impact parameter at mid-transit. Default is 0 (central transit).
        If derived from inclination and eccentricity, use
        b = (a/R★) cos i * (1 - e²) / (1 + e sin ω).
    eccentricity : float or array-like
        Orbital eccentricity. Default is 0 (circular).
    omega : QuantityLike
        Argument of periastron in degrees. Default is 90°.

    Returns
    -------
    table : QTable
        Table of derived transit quantities with units and descriptions.

    References
    ----------
    Winn, J. N. (2010), in Exoplanets, edited by S. Seager. Published by University of Arizona Press, Tucson, AZ, 2010, 526 pp. ISBN 978-0-8165-2945-2., p.55-77
    https://ui.adsabs.harvard.edu/abs/2010exop.book...55W/abstract

    Examples
    --------
    >>> from exohelp.planet.transit import transit_quantities
    >>> table = transit_quantities(period=365.25, r_planet=1.0, r_star=1.0, m_star=1.0, b=0.0, eccentricity=0.0, omega=90.0)
    """
    period = u.Quantity(period, "day")
    r_planet = u.Quantity(r_planet, "R_earth")
    r_star = u.Quantity(r_star, "R_sun")
    m_star = u.Quantity(m_star, "M_sun")

    a = keplers_third_law(period=period, mass=m_star)
    k = (r_planet.to("R_sun") / r_star).decompose().value

    _d = u.dimensionless_unscaled
    cols = [
        np.atleast_1d(k) * _d,
        np.atleast_1d(a.to("AU").value) * u.AU,
        np.atleast_1d(a_over_r_star(period, r_star, m_star)) * _d,
        np.atleast_1d(orbital_inclination(a, r_star, b, eccentricity, omega).to("deg").value)
        * u.deg,
        np.atleast_1d(transit_depth(k, b)) * _d,
        np.atleast_1d(
            transit_duration_total(period, r_planet, r_star, m_star, b, eccentricity, omega)
            .to("hour")
            .value
        )
        * u.hour,
        np.atleast_1d(
            transit_duration_flat(period, r_planet, r_star, m_star, b, eccentricity, omega)
            .to("hour")
            .value
        )
        * u.hour,
        np.atleast_1d(
            transit_duration_ingress(period, r_planet, r_star, m_star, b, eccentricity, omega)
            .to("hour")
            .value
        )
        * u.hour,
        np.atleast_1d(geometric_transit_probability(period, r_star, m_star, eccentricity, omega))
        * _d,
        np.atleast_1d(
            geometric_occultation_probability(period, r_star, m_star, eccentricity, omega)
        )
        * _d,
        np.atleast_1d(secondary_eclipse_timing_offset(period, eccentricity, omega).to("hour").value)
        * u.hour,
    ]
    names = [
        "k",
        "a",
        "a_over_r_star",
        "inclination",
        "transit_depth",
        "transit_duration_total",
        "transit_duration_flat",
        "transit_duration_ingress",
        "transit_probability",
        "occultation_probability",
        "eclipse_timing_offset",
    ]
    descriptions = [
        "Planet-to-star radius ratio Rp/R★",
        "Orbital semi-major axis (Kepler's third law)",
        "Scaled semi-major axis a/R★",
        "Orbital inclination",
        "Fractional flux loss at mid-transit (Mandel & Agol 2002)",
        "Total transit duration T₁₄, first to fourth contact (Winn 2010)",
        "Flat-bottom duration T₂₃, second to third contact (Winn 2010)",
        "Ingress (= egress) duration T₁₂ = (T₁₄ - T₂₃) / 2",
        "Geometric transit probability (R★/a) (1 + e sin ω) / (1 - e²)",
        "Geometric occultation probability (R★/a) (1 - e sin ω) / (1 - e²)",
        "Secondary eclipse offset from phase 0.5 (Winn 2010)",
    ]
    short_descriptions = [
        "Radius ratio",
        "Semi-major axis",
        "Scaled semi-major axis",
        "Orbital inclination",
        "Transit depth",
        "Total transit duration",
        "Flat-bottom duration",
        "Ingress duration",
        "Transit probability",
        "Occultation probability",
        "Eclipse timing offset",
    ]

    table = QTable(cols, names=names)
    for name, desc, short_desc in zip(names, descriptions, short_descriptions):
        table[name].info.description = desc  # type: ignore[union-attr]
        if table[name].info.meta is None:  # type: ignore[union-attr]
            table[name].info.meta = {}  # type: ignore[union-attr]
        table[name].info.meta["short_description"] = short_desc  # type: ignore[union-attr]
    return table
