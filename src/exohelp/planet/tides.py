import logging

import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.table import Table
from scipy.integrate import solve_ivp

__all__ = [
    "roche_limit",
    "tau_a",
    "tau_circ",
    "tau_e",
    "tidal_evolution",
]


def tau_a(a, e, m_star, r_star, m_planet, r_planet, q_planet_prime, q_star_prime):
    """Compute the fractional rate of change of the semi-major axis due to tidal dissipation.

    Implements Equation (1) from Jackson et al. (2009), including the e² correction
    from tides raised on the star:

        1/a · da/dt = -[
            (63/2) · √(G m_*³) r_p⁵ / (Q_p' m_p) · e²
            + (9/2) · √(G/m_*) r_*⁵ m_p / Q_s' · (1 + 57/4 · e²)
        ] · a^(-13/2)

    Parameters
    ----------
    a : Quantity
        Semi-major axis.
    e : float
        Orbital eccentricity.
    m_star : Quantity
        Stellar mass.
    r_star : Quantity
        Stellar radius.
    m_planet : Quantity
        Planet mass.
    r_planet : Quantity
        Planet radius.
    q_planet_prime : float
        Modified tidal quality factor for the planet.
    q_star_prime : float
        Modified tidal quality factor for the star.

    Returns
    -------
    rate : Quantity
        Fractional rate of change 1/a · da/dt, in units of 1/Gyr.

    References
    ----------
    Jackson, B., Greenberg, R., & Barnes, R. (2009), ApJ, 698, 1357.
    https://ui.adsabs.harvard.edu/abs/2009ApJ...698.1357J
    """
    # Term 1: Planetary Tide contribution
    term_p = (
        (63 / 2) * (np.sqrt(const.G * m_star**3) * r_planet**5) / (q_planet_prime * m_planet) * e**2
    )

    # Term 2: Stellar Tide contribution (includes e^2 correction missing in Jackson 2008)
    term_s = (
        (9 / 2)
        * (np.sqrt(const.G / m_star) * r_star**5 * m_planet)
        / q_star_prime
        * (1 + (57 / 4) * e**2)
    )

    da_dt = -(term_p + term_s) * a ** (-13 / 2)
    return da_dt.decompose().to("1 / Gyr")


def tau_e(a, e, m_star, r_star, m_planet, r_planet, q_planet_prime, q_star_prime):
    """Compute the fractional rate of change of eccentricity due to tidal dissipation.

    Implements Equation (2) from Jackson et al. (2009), using the corrected stellar
    tide coefficient (171/16):

        1/e · de/dt = -[
            (63/4) · √(G m_*³) r_p⁵ / (Q_p' m_p)
            + (171/16) · √(G/m_*) r_*⁵ m_p / Q_s'
        ] · a^(-13/2)

    Parameters
    ----------
    a : Quantity
        Semi-major axis.
    e : float
        Orbital eccentricity.
    m_star : Quantity
        Stellar mass.
    r_star : Quantity
        Stellar radius.
    m_planet : Quantity
        Planet mass.
    r_planet : Quantity
        Planet radius.
    q_planet_prime : float
        Modified tidal quality factor for the planet.
    q_star_prime : float
        Modified tidal quality factor for the star.

    Returns
    -------
    rate : Quantity
        Fractional rate of change 1/e · de/dt, in units of 1/Gyr.

    References
    ----------
    Jackson, B., Greenberg, R., & Barnes, R. (2009), ApJ, 698, 1357.
    https://ui.adsabs.harvard.edu/abs/2009ApJ...698.1357J
    """
    # Term 1: Planetary Tide contribution
    term_p = (63 / 4) * (np.sqrt(const.G * m_star**3) * r_planet**5) / (q_planet_prime * m_planet)

    # Term 2: Stellar Tide contribution (using the 171/16 correction)
    term_s = (171 / 16) * (np.sqrt(const.G / m_star) * r_star**5 * m_planet) / q_star_prime

    de_dt = -(term_p + term_s) * a ** (-13 / 2)
    return de_dt.decompose().to("1 / Gyr")


def tau_circ(a, m_star, m_planet, r_planet, q_planet_prime):
    """Compute the circularization timescale assuming constant semi-major axis.

    Implements Equation (4) from Jackson et al. (2008), neglecting stellar tides:

        τ_circ = Q_p' m_p / (63/4 · √(G m_*³) r_p⁵) · a^(13/2)

    Parameters
    ----------
    a : QuantityLike
        Semi-major axis. Assumed in AU if no unit is given.
    m_star : QuantityLike
        Stellar mass. Assumed in solar masses if no unit is given.
    m_planet : QuantityLike
        Planet mass. Assumed in Earth masses if no unit is given.
    r_planet : QuantityLike
        Planet radius. Assumed in Earth radii if no unit is given.
    q_planet_prime : float
        Modified tidal quality factor for the planet.

    Returns
    -------
    tau : Quantity
        Circularization timescale in Gyr.

    References
    ----------
    Jackson, B., Greenberg, R., & Barnes, R. (2008), ApJ, 678, 1396.
    https://ui.adsabs.harvard.edu/abs/2008ApJ...678.1396J

    Examples
    --------
    >>> import astropy.units as u
    >>> tau_circ(0.18 * u.AU, 1.045 * u.M_sun, 17.0 * u.M_earth, 2.66 * u.R_earth, q_planet_prime=1e4)
    <Quantity ... Gyr>
    """
    m_star = u.Quantity(m_star, "M_sun")
    m_planet = u.Quantity(m_planet, "M_earth")
    r_planet = u.Quantity(r_planet, "R_earth")

    inv_tau = (
        (63 / 4)
        * (np.sqrt(const.G * m_star**3) * r_planet**5)
        / (q_planet_prime * m_planet)
        * a ** (-13 / 2)
    )
    return (1 / inv_tau).decompose().to("Gyr")


def _da_dt(a, e, m_star, r_star, m_planet, r_planet, q_planet_prime, q_star_prime):
    """Derivative of semi-major axis with respect to time (AU/Gyr)."""
    return u.Quantity(a, "AU") * tau_a(
        a, e, m_star, r_star, m_planet, r_planet, q_planet_prime, q_star_prime
    )


def _de_dt(a, e, m_star, r_star, m_planet, r_planet, q_planet_prime, q_star_prime):
    """Derivative of eccentricity with respect to time (1/Gyr)."""
    return e * tau_e(a, e, m_star, r_star, m_planet, r_planet, q_planet_prime, q_star_prime)


def _tidal_system(t_gyr, y, m_star, r_star, m_planet, r_planet, q_planet_prime, q_star_prime):
    # solve_ivp works with unitless floats. Restore units for calculation.
    a_val = u.Quantity(y[0], "AU")
    e_val = y[1]

    # RK45 may probe negative a during trial steps; suppress the resulting
    # numpy "invalid value in power" warning — the NaNs are handled by the solver.
    with np.errstate(invalid="ignore"):
        da_dt_val = _da_dt(
            a_val, e_val, m_star, r_star, m_planet, r_planet, q_planet_prime, q_star_prime
        )
        de_dt_val = _de_dt(
            a_val, e_val, m_star, r_star, m_planet, r_planet, q_planet_prime, q_star_prime
        )

    return [da_dt_val.value, de_dt_val.value]


def roche_limit(m_star, m_planet, r_planet):
    """Compute the fluid Roche limit for a planet orbiting a star.

    Parameters
    ----------
    m_star : QuantityLike
        Stellar mass. Assumed in solar masses if no unit is given.
    m_planet : QuantityLike
        Planet mass. Assumed in Earth masses if no unit is given.
    r_planet : QuantityLike
        Planet radius. Assumed in Earth radii if no unit is given.

    Returns
    -------
    a_roche : Quantity
        Roche limit in AU.

    References
    ----------
    Jackson, B., et al. (2016).
    """
    r_planet = u.Quantity(r_planet, "R_earth") if not isinstance(r_planet, u.Quantity) else r_planet
    m_star = u.Quantity(m_star, "M_sun") if not isinstance(m_star, u.Quantity) else m_star
    m_planet = u.Quantity(m_planet, "M_earth") if not isinstance(m_planet, u.Quantity) else m_planet

    return ((r_planet / 0.462) * (m_star / m_planet) ** (1 / 3)).decompose().to("AU")


def _reached_roche_limit(t, y, m_star, r_star, m_planet, r_planet, q_planet_prime, q_star_prime):
    a_curr = u.Quantity(y[0], "AU")
    a_roche = roche_limit(m_star, m_planet, r_planet)
    return (a_curr - a_roche.decompose().to("AU")).value


_reached_roche_limit.terminal = True


def _reached_stellar_surface(
    t, y, m_star, r_star, m_planet, r_planet, q_planet_prime, q_star_prime
):
    a_curr = u.Quantity(y[0], "AU")
    return (a_curr - r_star.to("AU")).value


_reached_stellar_surface.terminal = True


def tidal_evolution(
    a_init,
    e_init,
    m_star,
    r_star,
    m_planet,
    r_planet,
    q_planet_prime,
    q_star_prime,
    time_span_gyr=1000.0,
    max_step=1.0,
):
    """Integrate the tidal evolution equations forward in time.

    Integrates the coupled ODEs for semi-major axis and eccentricity using
    RK45. Integration stops early if the planet reaches the Roche limit or
    the stellar surface.

    Parameters
    ----------
    a_init : Quantity
        Initial semi-major axis.
    e_init : float
        Initial eccentricity.
    m_star : Quantity
        Stellar mass.
    r_star : Quantity
        Stellar radius.
    m_planet : Quantity
        Planet mass.
    r_planet : Quantity
        Planet radius.
    q_planet_prime : float
        Modified tidal quality factor for the planet.
    q_star_prime : float
        Modified tidal quality factor for the star.
    t_start_gyr : float, optional
        Start time for integration in Gyr.
        Choose a negative value to integrate backward in time.
    max_step : float, optional
        Maximum step size for the ODE solver in Gyr.

    Returns
    -------
    solution : OdeResult
        Result from `scipy.integrate.solve_ivp`. The `.t` attribute holds
        time in Gyr and `.y` holds ``[a (AU), e]``.

    References
    ----------
    Jackson, B., Greenberg, R., & Barnes, R. (2008), ApJ, 678, 1396.
    https://ui.adsabs.harvard.edu/abs/2008ApJ...678.1396J

    Jackson, B., Greenberg, R., & Barnes, R. (2009), ApJ, 698, 1357.
    https://ui.adsabs.harvard.edu/abs/2009ApJ...698.1357J

    Examples
    --------
    >>> import astropy.units as u
    >>> from exohelp.planet.tides import tidal_evolution
    >>> solution = tidal_evolution(
    ...     a_init=0.1759 * u.AU, e_init=0.39,
    ...     m_star=1.045 * u.M_sun, r_star=1.235 * u.R_sun,
    ...     m_planet=17.6 * u.M_earth, r_planet=2.65 * u.R_earth,
    ...     q_planet_prime=500,
    ...     q_star_prime=1e4,
    ... )
    >>> ecc_idx = np.argmin(np.abs(solution['ecc'] - 0.2))
    >>> round(float(solution['time'][ecc_idx]), 0)  # Time to reach e=0.2
    69.0
    """
    y0 = [a_init.to("AU").value, e_init]
    t_span = (0, time_span_gyr)

    solution = solve_ivp(
        _tidal_system,
        t_span,
        y0,
        args=(m_star, r_star, m_planet, r_planet, q_planet_prime, q_star_prime),
        events=[_reached_roche_limit, _reached_stellar_surface],
        dense_output=True,
        method="RK45",
        rtol=1e-9,
        max_step=max_step,
    )

    solution_table = Table(
        {
            "time": solution.t,
            "a": solution.y[0],
            "ecc": solution.y[1],
        },
        units={"time": u.Gyr, "a": u.AU, "ecc": None},  # type: ignore
    )
    solution_table.meta = {
        "m_star": m_star,
        "r_star": r_star,
        "m_planet": m_planet,
        "r_planet": r_planet,
        "q_planet_prime": q_planet_prime,
        "q_star_prime": q_star_prime,
        "roche_limit": roche_limit(m_star, m_planet, r_planet),
    }
    solution_table.meta.update({k: v for k, v in solution.items() if k not in ["t", "y"]})

    if not solution.success:
        logging.warning("Warning: Integration did not complete successfully.")

    return solution_table
