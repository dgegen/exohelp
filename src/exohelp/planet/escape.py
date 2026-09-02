"""Atmospheric escape, high-energy (XUV) stellar evolution, and photoevaporation calculations.

This module implements the hydrodynamic energy-limited and radiation/recombination-limited
photoevaporation frameworks, tidal Roche-lobe corrections, stellar high-energy (XUV) evolution
tracks, and analytic scaling fits from coupled thermal-evolution models.

Physical Scope & Validity Regimes
----------------------------------
- **Energy-Limited Regime**: Valid when radiative recombination and line-cooling losses are
  subdominant compared to expansion work (:math:`F_{\\rm XUV} \\lesssim 10^4{\\rm\\ erg\\ cm^{-2}\\ s^{-1}}`).
- **Radiation/Recombination-Limited Regime**: For high fluxes (:math:`F_{\\rm XUV} \\gtrsim 10^4{\\rm\\ erg\\ cm^{-2}\\ s^{-1}}`),
  photoevaporation transitions to :math:`\\dot{M} \\propto F_{\\rm XUV}^{0.5 - 0.6}` due to
  Lyman-:math:`\\alpha` line cooling and radiative recombination (Murray-Clay et al. 2009).
- **Tidal Roche-Lobe Correction**: Accounts for the reduction of the effective gravitational
  potential barrier by the host star's tidal field (Erkaev et al. 2007; Lopez & Fortney 2013).
- **Stellar High-Energy Tracks**: Parametric saturation phase followed by power-law decay
  (Ribas et al. 2005; Jackson et al. 2012; Johnstone et al. 2021).
- **Analytic Scaling Relations**: Fits for threshold stripping flux and envelope loss fraction
  from coupled interior-atmosphere evolution grids (Lopez & Fortney 2013).

*Note on Unimplemented Mechanisms*:
Full 1D steady-state hydrodynamic relaxation wind solvers (Murray-Clay et al. 2009),
stellar wind ram-pressure confinement/breezes, and core-powered mass loss (CPML; Ginzburg et al.
2018; Gupta & Schlichting 2019) require separate numerical formulations.

References
----------
- Watson, A. J., Donahue, T. M., & Walker, J. C. G. (1981), Icarus, 48, 150.
- Ribas, I., et al. (2005), ApJ, 622, 680.
- Valencia, D., O'Connell, R. J., & Sasselov, D. (2006), Icarus, 181, 545.
- Erkaev, N. V., et al. (2007), A&A, 472, 329.
- Fortney, J. J., Marley, M. S., & Barnes, J. W. (2007), ApJ, 659, 1661.
- Murray-Clay, R. A., Chiang, E. I., & Murray, N. (2009), ApJ, 693, 23.
- Jackson, A. P., et al. (2012), MNRAS, 424, 11.
- Lopez, E. D., & Fortney, J. J. (2013), ApJ, 776, 2.
- Owen, J. E., & Wu, Y. (2013), ApJ, 775, 105.
- Owen, J. E., & Wu, Y. (2017), ApJ, 847, 29.
- Salz, M., et al. (2016), A&A, 586, A75.
- Ginzburg, S., Schlichting, H. E., & Sari, R. (2018), MNRAS, 476, 759.
- Johnstone, C. P., et al. (2021), A&A, 649, A96.
- Caldiroli, A., et al. (2021), A&A, 655, A30.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import logging

import astropy.constants as const
import astropy.units as u
import numpy as np
from astropy.table import QTable
from scipy.integrate import solve_ivp

from ..citations import cites
from ..type import QuantityLike
from ..units import S_earth

__all__ = [
    "StellarXUVTrack",
    "default_rocky_core_radius",
    "energy_limited_mass_loss_rate",
    "escape_velocity",
    "jeans_parameter",
    "lopez_fortney_fraction_lost",
    "lopez_fortney_threshold_flux",
    "photoevaporation_evolution",
    "photoevaporation_static",
    "recombination_limited_mass_loss_rate",
    "roche_lobe_correction_factor",
    "salz_efficiency",
]


def escape_velocity(mass: QuantityLike, radius: QuantityLike) -> u.Quantity:
    """Compute the Newtonian surface escape velocity of a body.

    .. math::
        v_{\\rm esc} = \\sqrt{\\frac{2 G M}{R}}

    Parameters
    ----------
    mass : QuantityLike
        Mass of the body. Assumed to be in Earth masses if unit is omitted.
    radius : QuantityLike
        Radius of the body. Assumed to be in Earth radii if unit is omitted.

    Returns
    -------
    v_esc : u.Quantity
        Surface escape velocity in km/s.

    Examples
    --------
    >>> from exohelp.planet.escape import escape_velocity
    >>> round(float(escape_velocity(1.0, 1.0).value), 2)  # Earth
    11.18
    """
    m = u.Quantity(mass, u.M_earth)
    r = u.Quantity(radius, u.R_earth)
    v = np.sqrt(2.0 * const.G * m / r)
    return v.to(u.km / u.s)


@cites("Valencia2006", "FortneyMarleyBarnes2007", "Ginzburg2018")
def default_rocky_core_radius(m_core: QuantityLike) -> u.Quantity:
    """Compute approximate Earth-like rocky core radius.

    .. math::
        R_{\\rm core} \\approx R_\\oplus \\left(\\frac{M_{\\rm core}}{M_\\oplus}\\right)^{0.27}

    Parameters
    ----------
    m_core : QuantityLike
        Core mass. Assumed in Earth masses if unit is omitted.

    Returns
    -------
    r_core : u.Quantity
        Rocky core radius in Earth radii (R_earth).

    References
    ----------
    - Valencia, D., O'Connell, R. J., & Sasselov, D. (2006), Icarus, 181, 545.
    - Fortney, J. J., Marley, M. S., & Barnes, J. W. (2007), ApJ, 659, 1661.
    - Ginzburg, S., Schlichting, H. E., & Sari, R. (2018), MNRAS, 476, 759.

    Examples
    --------
    >>> from exohelp.planet.escape import default_rocky_core_radius
    >>> round(float(default_rocky_core_radius(1.0).value), 2)
    1.0
    >>> round(float(default_rocky_core_radius(5.0).value), 2)
    1.54
    """
    mc = u.Quantity(m_core, u.M_earth).value
    return (mc**0.27) * u.R_earth


def jeans_parameter(
    mass: QuantityLike,
    radius: QuantityLike,
    temperature: QuantityLike,
    mean_molecular_weight: float | u.Quantity = 1.0,
) -> float | np.ndarray:
    """Compute the dimensionless classical Jeans thermal escape parameter.

    .. math::
        \\lambda = \\frac{G M_p \\mu m_{\\rm H}}{k_{\\rm B} T R_p}

    Thermal escape is negligible when :math:`\\lambda \\gg 15 - 20` (Chamberlain & Hunten 1987).

    Parameters
    ----------
    mass : QuantityLike
        Planet mass. Assumed in Earth masses if unit is omitted.
    radius : QuantityLike
        Planet radius. Assumed in Earth radii if unit is omitted.
    temperature : QuantityLike
        Atmospheric / equilibrium temperature. Assumed in Kelvin if unit is omitted.
    mean_molecular_weight : float or Quantity, optional
        Mean molecular weight in atomic mass units (e.g. 1.0 for atomic H,
        2.0 for H2, 18.0 for H2O, 28.97 for air), or as a mass Quantity. Default is 1.0.

    Returns
    -------
    lambda_jeans : float or np.ndarray
        Dimensionless Jeans parameter.

    References
    ----------
    - Jeans, J. H. (1925), *The Dynamical Theory of Gases*, Cambridge Univ. Press.
    - Chamberlain, J. W., & Hunten, D. M. (1987), *Theory of Planetary Atmospheres*, Academic Press.

    Examples
    --------
    >>> from exohelp.planet.escape import jeans_parameter
    >>> round(float(jeans_parameter(1.0, 1.0, 288.0, mean_molecular_weight=28.97)), 1)
    761.6
    """
    m = u.Quantity(mass, u.M_earth)
    r = u.Quantity(radius, u.R_earth)
    t = u.Quantity(temperature, u.K)

    if isinstance(mean_molecular_weight, u.Quantity):
        mu_mass = mean_molecular_weight
    else:
        mu_mass = mean_molecular_weight * const.m_p

    val = (const.G * m * mu_mass) / (const.k_B * t * r)
    res = val.decompose().value
    return float(res) if np.ndim(res) == 0 else res


@cites("Erkaev2007", "LopezFortney2013", note="Erkaev Eqs. 16-18; Lopez & Fortney Eqs. 2-3")
def roche_lobe_correction_factor(
    semi_major_axis: QuantityLike,
    m_planet: QuantityLike,
    m_star: QuantityLike,
    r_xuv: QuantityLike,
) -> float | np.ndarray:
    """Compute the tidal Roche-lobe reduction factor K_tide.

    Accounts for the reduction of the effective gravitational potential barrier
    at the absorption radius due to the host star's tidal field:

    .. math::
        K_{\\rm tide} = 1 - \\frac{3}{2\\xi} + \\frac{1}{2\\xi^3}, \\quad
        \\xi = \\frac{R_{\\rm Roche}}{R_{\\rm xuv}}, \\quad
        R_{\\rm Roche} = a \\left(\\frac{M_p}{3 M_*}\\right)^{1/3}

    When :math:`\\xi \\le 1`, the planet fills or exceeds its Roche lobe, and :math:`K_{\\rm tide} = 0`.

    Parameters
    ----------
    semi_major_axis : QuantityLike
        Semi-major axis. Assumed in AU if unit is omitted.
    m_planet : QuantityLike
        Planet mass. Assumed in Earth masses if unit is omitted.
    m_star : QuantityLike
        Host star mass. Assumed in Solar masses if unit is omitted.
    r_xuv : QuantityLike
        Effective absorption radius for high-energy flux. Assumed in Earth radii if unit is omitted.

    Returns
    -------
    k_tide : float or np.ndarray
        Dimensionless Roche-lobe correction factor (clamped between 0 and 1).

    References
    ----------
    - Erkaev, N. V., et al. (2007), A&A, 472, 329, Equations (16)-(18).
    - Lopez, E. D., & Fortney, J. J. (2013), ApJ, 776, 2, Equations (2)-(3).

    Examples
    --------
    >>> from exohelp.planet.escape import roche_lobe_correction_factor
    >>> round(float(roche_lobe_correction_factor(0.1, 10.0, 1.0, 3.0)), 3)
    0.911
    """
    a = u.Quantity(semi_major_axis, u.AU)
    mp = u.Quantity(m_planet, u.M_earth)
    ms = u.Quantity(m_star, u.M_sun)
    rxuv = u.Quantity(r_xuv, u.R_earth)

    r_roche = a * (mp / (3.0 * ms)) ** (1.0 / 3.0)
    xi = (r_roche / rxuv).decompose().value

    if np.ndim(xi) == 0:
        if xi > 1.0:
            return float(1.0 - 1.5 / xi + 0.5 / (xi**3))
        return 0.0

    k = np.where(xi > 1.0, 1.0 - 1.5 / xi + 0.5 / (xi**3), 0.0)
    return np.clip(k, 0.0, 1.0)


@dataclass(frozen=True)
class StellarXUVTrack:
    """Parametric high-energy (XUV) stellar luminosity evolution model.

    Evolution follows a saturated phase up to `t_sat`, followed by power-law decay:

    .. math::
        L_{\\rm xuv}(t) = \\begin{cases}
            f_{\\rm sat} L_{\\rm bol}, & t \\le t_{\\rm sat} \\\\
            f_{\\rm sat} L_{\\rm bol} \\left(\\frac{t}{t_{\\rm sat}}\\right)^{-\\beta},
            & t > t_{\\rm sat}
        \\end{cases}

    Parameters
    ----------
    f_sat : float
        Saturated XUV luminosity fraction :math:`L_{\\rm xuv} / L_{\\rm bol}`.
    t_sat : QuantityLike
        Saturation duration. Assumed in Myr if unit is omitted.
    beta : float
        Decay power-law exponent (typically 1.1 - 1.4 for solar-type stars).
    l_bol : QuantityLike, optional
        Stellar bolometric luminosity. Assumed in Solar luminosities if unit is omitted.
        Default is 1.0 L_sun.
    m_star : QuantityLike, optional
        Stellar mass. Assumed in Solar masses if unit is omitted. Default is 1.0 M_sun.
    name : str, optional
        Descriptive name of the activity track.
    description : str, optional
        Additional notes or citations.

    References
    ----------
    - Ribas, I., et al. (2005), ApJ, 622, 680.
    - Jackson, A. P., et al. (2012), MNRAS, 424, 11, Equation (1).
    - Johnstone, C. P., et al. (2021), A&A, 649, A96.
    """

    f_sat: float
    t_sat: u.Quantity
    beta: float
    l_bol: u.Quantity = 1.0 * u.L_sun
    m_star: u.Quantity = 1.0 * u.M_sun
    name: str = ""
    description: str = ""

    def __init__(
        self,
        f_sat: float,
        t_sat: QuantityLike,
        beta: float,
        l_bol: QuantityLike = 1.0 * u.L_sun,
        m_star: QuantityLike = 1.0 * u.M_sun,
        name: str = "",
        description: str = "",
    ):
        object.__setattr__(self, "f_sat", float(f_sat))
        object.__setattr__(self, "t_sat", u.Quantity(t_sat, u.Myr))
        object.__setattr__(self, "beta", float(beta))
        object.__setattr__(self, "l_bol", u.Quantity(l_bol, u.L_sun))
        object.__setattr__(self, "m_star", u.Quantity(m_star, u.M_sun))
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", description)

    @classmethod
    @cites("Ribas2005", "Johnstone2021")
    def nominal_solar(
        cls,
        l_bol: QuantityLike = 1.0 * u.L_sun,
        m_star: QuantityLike = 1.0 * u.M_sun,
    ) -> StellarXUVTrack:
        """Nominal G-dwarf track (Ribas et al. 2005; Johnstone et al. 2021)."""
        return cls(
            f_sat=10 ** (-3.5),
            t_sat=100.0 * u.Myr,
            beta=1.24,
            l_bol=l_bol,
            m_star=m_star,
            name="Nominal G-Dwarf",
            description="Typical solar-type G dwarf (Ribas+2005; Johnstone+2021)",
        )

    @classmethod
    def low_activity(
        cls,
        l_bol: QuantityLike = 1.0 * u.L_sun,
        m_star: QuantityLike = 1.0 * u.M_sun,
    ) -> StellarXUVTrack:
        """Slow rotator / low activity early evolution track."""
        return cls(
            f_sat=10 ** (-3.8),
            t_sat=50.0 * u.Myr,
            beta=1.40,
            l_bol=l_bol,
            m_star=m_star,
            name="Low Activity",
            description="Slowly rotating young star, rapid magnetic spin-down",
        )

    @classmethod
    def moderate_activity(
        cls,
        l_bol: QuantityLike = 1.0 * u.L_sun,
        m_star: QuantityLike = 1.0 * u.M_sun,
    ) -> StellarXUVTrack:
        """Moderate-high activity track."""
        return cls(
            f_sat=10 ** (-3.2),
            t_sat=200.0 * u.Myr,
            beta=1.20,
            l_bol=l_bol,
            m_star=m_star,
            name="Moderate-High Activity",
            description="Moderately active early stellar evolution",
        )

    @classmethod
    def high_activity(
        cls,
        l_bol: QuantityLike = 1.0 * u.L_sun,
        m_star: QuantityLike = 1.0 * u.M_sun,
    ) -> StellarXUVTrack:
        """Rapid rotator / upper-envelope active track."""
        return cls(
            f_sat=10 ** (-3.0),
            t_sat=300.0 * u.Myr,
            beta=1.15,
            l_bol=l_bol,
            m_star=m_star,
            name="High Activity (Upper Envelope)",
            description="Rapid rotator with extended saturated emission",
        )

    def luminosity_xuv(self, time: QuantityLike) -> u.Quantity:
        """Calculate the XUV luminosity L_xuv at a given stellar age or array of ages.

        Parameters
        ----------
        time : QuantityLike
            System age. Assumed in Myr if unit is omitted.

        Returns
        -------
        l_xuv : u.Quantity
            High-energy luminosity in erg/s.
        """
        t = u.Quantity(time, u.Myr)
        t_sat = self.t_sat.to(u.Myr)

        ratio = (t / t_sat).decompose().value
        decay = np.where(ratio <= 1.0, 1.0, ratio ** (-self.beta))
        l_xuv = self.f_sat * self.l_bol * decay
        return l_xuv.to(u.erg / u.s)

    def flux_xuv(
        self,
        time: QuantityLike,
        semi_major_axis: QuantityLike,
        eccentricity: float | np.ndarray = 0.0,
    ) -> u.Quantity:
        """Calculate the orbit-averaged XUV flux F_xuv received at semi-major axis a.

        .. math::
            F_{\\rm xuv}(t) = \\frac{L_{\\rm xuv}(t)}{4 \\pi a^2 \\sqrt{1 - e^2}}

        Parameters
        ----------
        time : QuantityLike
            System age. Assumed in Myr if unit is omitted.
        semi_major_axis : QuantityLike
            Orbital semi-major axis. Assumed in AU if unit is omitted.
        eccentricity : float or np.ndarray, optional
            Orbital eccentricity. Default is 0.0.

        Returns
        -------
        f_xuv : u.Quantity
            Orbit-averaged XUV flux in erg / (s * cm^2).
        """
        a = u.Quantity(semi_major_axis, u.AU)
        ecc_factor = 1.0 / np.sqrt(1.0 - np.asarray(eccentricity) ** 2)
        l_xuv = self.luminosity_xuv(time)
        f_xuv = (l_xuv / (4.0 * np.pi * a**2)) * ecc_factor
        return f_xuv.to(u.erg / (u.s * u.cm**2))


@cites("Salz2016", "Caldiroli2021", note="Salz Eq. 5 and Table 2")
def salz_efficiency(v_esc: QuantityLike) -> float | np.ndarray:
    """Compute empirical photoevaporative efficiency eta from surface escape velocity.

    Implements the empirical log-linear relation from Salz et al. (2016):

    .. math::
        \\log_{10} \\eta = -0.98 - 0.29 \\times \\left(\\frac{\\Phi_{\\rm g}}{10^{13}{\\rm\\ erg\\ g^{-1}}}\\right),
        \\quad \\Phi_{\\rm g} = \\frac{v_{\\rm esc}^2}{2}

    Parameters
    ----------
    v_esc : QuantityLike
        Surface escape velocity. Assumed in km/s if unit is omitted.

    Returns
    -------
    eta : float or np.ndarray
        Dimensionless efficiency clamped between 0.01 and 0.35.

    References
    ----------
    - Salz, M., et al. (2016), A&A, 586, A75, Equation (5) and Table 2.
    - Caldiroli, A., et al. (2021), A&A, 655, A30.
    """
    v = u.Quantity(v_esc, u.km / u.s).to(u.cm / u.s)
    phi_g = 0.5 * (v.value**2)
    phi_13 = phi_g / 1.0e13
    log_eta = -0.98 - 0.29 * phi_13
    eta = 10.0**log_eta
    if np.ndim(eta) == 0:
        return float(np.clip(eta, 0.01, 0.35))
    return np.clip(eta, 0.01, 0.35)


@cites("Watson1981", "Erkaev2007", "MurrayClay2009", "LopezFortney2013", "OwenWu2013")
def energy_limited_mass_loss_rate(
    flux_xuv: QuantityLike,
    r_planet: QuantityLike,
    m_planet: QuantityLike,
    eta: float | QuantityLike = 0.10,
    r_xuv: QuantityLike | None = None,
    r_xuv_factor: float = 1.0,
    k_tide: float = 1.0,
) -> u.Quantity:
    """Compute the hydrodynamic energy-limited mass-loss rate dM/dt.

    .. math::
        \\dot{M} = \\frac{\\eta \\pi R_{\\rm xuv}^3 F_{\\rm xuv}}{G M_p K_{\\rm tide}}

    Physical Regime & Validity
    --------------------------
    This formulation assumes that photoevaporation is **energy-limited**—i.e., that a constant
    fraction :math:`\\eta` of the incident XUV flux goes into :math:`P\\,dV` expansion work.
    This holds for moderate incident fluxes (:math:`F_{\\rm XUV} \\lesssim 10^4{\\rm\\ erg\\ cm^{-2}\\ s^{-1}}`).
    At higher fluxes, the escape transitions to the **radiation/recombination-limited** regime
    (:math:`\\dot{M} \\propto F_{\\rm XUV}^{0.5 - 0.6}`), where radiative recombination and
    Lyman-:math:`\\alpha` line cooling dominate over mechanical expansion (Murray-Clay et al. 2009;
    Owen & Wu 2013).

    Parameters
    ----------
    flux_xuv : QuantityLike
        Incident high-energy XUV flux. Assumed in erg / (s * cm^2) if unit is omitted.
    r_planet : QuantityLike
        Planet optical radius. Assumed in Earth radii if unit is omitted.
    m_planet : QuantityLike
        Planet mass. Assumed in Earth masses if unit is omitted.
    eta : float or QuantityLike, optional
        Photoevaporation efficiency (typically 0.05 - 0.15). Default is 0.10.
    r_xuv : QuantityLike, optional
        Effective XUV absorption radius. If provided, overrides `r_xuv_factor`.
    r_xuv_factor : float, optional
        Ratio :math:`R_{\\rm xuv} / R_p` (typically 1.0 - 1.2). Default is 1.0.
    k_tide : float, optional
        Roche-lobe tidal correction factor :math:`K_{\\rm tide} \\le 1.0`. Default is 1.0.

    Returns
    -------
    dM_dt : u.Quantity
        Atmospheric mass-loss rate in g/s.

    References
    ----------
    - Watson, A. J., Donahue, T. M., & Walker, J. C. G. (1981), Icarus, 48, 150.
    - Erkaev, N. V., et al. (2007), A&A, 472, 329.
    - Murray-Clay, R. A., Chiang, E. I., & Murray, N. (2009), ApJ, 693, 23, Equation (19).
    - Lopez, E. D., & Fortney, J. J. (2013), ApJ, 776, 2, Equation (1).
    - Owen, J. E., & Wu, Y. (2013), ApJ, 775, 105, Equation (1).

    Examples
    --------
    >>> from exohelp.planet.escape import energy_limited_mass_loss_rate
    >>> mdot = energy_limited_mass_loss_rate(100.0, 2.5, 15.0, eta=0.10)
    >>> round(float(mdot.to('kg/s').value), 1)
    21301.7
    """
    f_xuv = u.Quantity(flux_xuv, u.erg / (u.s * u.cm**2))
    rp = u.Quantity(r_planet, u.R_earth)
    mp = u.Quantity(m_planet, u.M_earth)

    rxuv = u.Quantity(r_xuv, u.R_earth) if r_xuv is not None else r_xuv_factor * rp

    # Ensure k_tide is strictly positive to prevent division by zero
    k_eff = max(float(k_tide), 1.0e-4)

    numerator = eta * np.pi * (rxuv**3) * f_xuv
    denominator = const.G * mp * k_eff

    rate = numerator / denominator
    return rate.to(u.g / u.s)


@cites("MurrayClay2009", "OwenWu2013")
def recombination_limited_mass_loss_rate(
    flux_xuv: QuantityLike,
    exponent: float = 0.6,
    f_ref: QuantityLike = 5.0e5 * u.erg / (u.s * u.cm**2),
    mdot_ref: QuantityLike = 4.0e12 * u.g / u.s,
) -> u.Quantity:
    """Compute the radiation/recombination-limited photoevaporative mass-loss rate.

    .. math::
        \\dot{M}_{\\rm rr-lim} \\approx \\dot{M}_{\\rm ref} \\left(\\frac{F_{\\rm XUV}}{F_{\\rm ref}}\\right)^{\\alpha}

    where :math:`\\alpha \\approx 0.5 - 0.6`. At high incident XUV fluxes
    (:math:`F_{\\rm XUV} \\gtrsim 10^4{\\rm\\ erg\\ cm^{-2}\\ s^{-1}}`), photoevaporative escape
    transitions from the energy-limited regime to the radiation/recombination-limited regime
    due to efficient radiative recombination and Lyman-:math:`\\alpha` line cooling (Murray-Clay et al. 2009;
    Owen & Wu 2013).

    Parameters
    ----------
    flux_xuv : QuantityLike
        Incident high-energy XUV flux. Assumed in erg / (s * cm^2) if unit is omitted.
    exponent : float, optional
        Power-law scaling exponent :math:`\\alpha` (typically 0.5 to 0.6). Default is 0.6.
    f_ref : QuantityLike, optional
        Reference incident flux. Default is 5.0e5 erg / (s * cm^2).
    mdot_ref : QuantityLike, optional
        Reference mass-loss rate at `f_ref`. Default is 4.0e12 g/s.

    Returns
    -------
    dM_dt : u.Quantity
        Mass-loss rate in g/s.

    References
    ----------
    - Murray-Clay, R. A., Chiang, E. I., & Murray, N. (2009), ApJ, 693, 23.
    - Owen, J. E., & Wu, Y. (2013), ApJ, 775, 105.

    Examples
    --------
    >>> from exohelp.planet.escape import recombination_limited_mass_loss_rate
    >>> mdot = recombination_limited_mass_loss_rate(5.0e5)
    >>> float(mdot.to('g/s').value)
    4000000000000.0
    """
    f = u.Quantity(flux_xuv, u.erg / (u.s * u.cm**2))
    f0 = u.Quantity(f_ref, u.erg / (u.s * u.cm**2))
    md0 = u.Quantity(mdot_ref, u.g / u.s)

    ratio = (f / f0).decompose().value
    rate = md0 * (ratio**exponent)
    return rate.to(u.g / u.s)


@cites("LopezFortney2013", note="Eqs. 5 & 6")
def lopez_fortney_threshold_flux(
    m_core: QuantityLike,
    eta: float = 0.10,
    age: QuantityLike | None = None,
) -> u.Quantity:
    """Compute the threshold incident flux for complete sub-Neptune envelope stripping.

    Implements the analytic scaling power-law from Lopez & Fortney (2013):

    .. math::
        F_{\\rm th} = 0.5\\, F_\\oplus \\left(\\frac{M_{\\rm core}}{M_\\oplus}\\right)^{2.4}
        \\left(\\frac{\\eta}{0.1}\\right)^{-0.7}

    For young systems (:math:`t < 1{\\rm\\ Gyr}`), an optional early decay term from Equation (5)
    of Lopez & Fortney (2013) is included:

    .. math::
        F_{\\rm th}(t) \\approx F_{\\rm th,\\infty} + 3.4\\,F_\\oplus \\exp\\left(-\\frac{t - 140{\\rm\\ Myr}}{80{\\rm\\ Myr}}\\right)

    where :math:`F_\\oplus = 1361{\\rm\\ W\\ m^{-2}}` is the solar constant at 1 AU. Planets receiving
    fluxes :math:`F_{\\rm bol} \\gtrsim F_{\\rm th}` are expected to be stripped to bare rocky cores.

    Parameters
    ----------
    m_core : QuantityLike
        Rocky core mass. Assumed in Earth masses if unit is omitted.
    eta : float, optional
        Photoevaporative efficiency. Default is 0.10.
    age : QuantityLike, optional
        System age. If provided and :math:`t < 1{\\rm\\ Gyr}`, includes the early threshold flux
        elevation from Lopez & Fortney (2013, Eq. 5). Default is None (mature system asymptotic limit).

    Returns
    -------
    f_th : u.Quantity
        Threshold bolometric insolation flux in units of Earth insolation (S_earth) or W/m^2.

    References
    ----------
    Lopez, E. D., & Fortney, J. J. (2013), ApJ, 776, 2, Equations (5) and (6).

    Examples
    --------
    >>> from exohelp.planet.escape import lopez_fortney_threshold_flux
    >>> round(float(lopez_fortney_threshold_flux(5.0).value), 1)
    23.8
    >>> round(float(lopez_fortney_threshold_flux(5.0, age=140.0).value), 1)
    27.2
    """
    mc = u.Quantity(m_core, u.M_earth).value
    f_th_searth = 0.5 * (mc**2.4) * ((eta / 0.10) ** (-0.7))

    if age is not None:
        t_myr = u.Quantity(age, u.Myr).value
        if t_myr < 1000.0:
            f_th_searth += 3.4 * np.exp(-(t_myr - 140.0) / 80.0)

    return f_th_searth * S_earth


@cites("LopezFortney2013", note="Eq. 7")
def lopez_fortney_fraction_lost(
    insolation: QuantityLike,
    m_core: QuantityLike,
    eta: float = 0.10,
    age: QuantityLike | None = None,
) -> float | np.ndarray:
    """Compute the analytic fraction of primordial envelope lost to photoevaporation.

    Implements Equation (7) of Lopez & Fortney (2013):

    .. math::
        f_{\\rm lost} = 0.5 \\left(\\frac{F}{F_{\\rm th}}\\right)^{1.1}

    clamped between 0.0 and 1.0.

    Parameters
    ----------
    insolation : QuantityLike
        Incident bolometric stellar flux. Can be given in units of Earth flux (S_earth),
        or physical flux units (e.g. W/m^2, erg/(s cm^2)), or raw float (assumed in S_earth).
    m_core : QuantityLike
        Rocky core mass. Assumed in Earth masses if unit is omitted.
    eta : float, optional
        Photoevaporative efficiency. Default is 0.10.
    age : QuantityLike, optional
        System age. If provided, passed to `lopez_fortney_threshold_flux`.

    Returns
    -------
    f_lost : float or np.ndarray
        Fraction of initial H/He envelope lost (0 to 1).

    References
    ----------
    Lopez, E. D., & Fortney, J. J. (2013), ApJ, 776, 2, Equation (7).

    Examples
    --------
    >>> from exohelp.planet.escape import lopez_fortney_fraction_lost
    >>> round(float(lopez_fortney_fraction_lost(10.0, 5.0)), 3)
    0.193
    """
    f_inc = (
        insolation.to(S_earth)
        if isinstance(insolation, u.Quantity)
        else u.Quantity(insolation, S_earth)
    )
    f_th = lopez_fortney_threshold_flux(m_core, eta=eta, age=age).to(S_earth)
    ratio = (f_inc / f_th).decompose().value
    f_lost = 0.5 * (ratio**1.1)

    if np.ndim(f_lost) == 0:
        return float(np.clip(f_lost, 0.0, 1.0))
    return np.clip(f_lost, 0.0, 1.0)


@cites("Watson1981", "Erkaev2007", "MurrayClay2009", "LopezFortney2013", "OwenWu2013")
def photoevaporation_static(
    m_planet: QuantityLike,
    r_planet: QuantityLike,
    semi_major_axis: QuantityLike,
    track: StellarXUVTrack,
    age: QuantityLike = 5.0 * u.Gyr,
    eccentricity: float = 0.0,
    m_star: QuantityLike | None = None,
    eta: float | Sequence[float] = 0.10,
    r_xuv_factor: float = 1.0,
    apply_tidal_correction: bool = True,
    t_start: QuantityLike = 5.0 * u.Myr,
    n_steps: int = 5000,
) -> QTable:
    """Compute integrated lifetime photoevaporative mass loss under static planet dimensions.

    Integrates :math:`\\Delta M = \\int_{t_{\\rm start}}^{t_{\\rm age}} \\dot{M}(t) dt`
    assuming fixed radius and mass over time. Ideal for fast parameter surveys and comparisons.

    Parameters
    ----------
    m_planet : QuantityLike
        Planet mass. Assumed in Earth masses if unit is omitted.
    r_planet : QuantityLike
        Planet radius. Assumed in Earth radii if unit is omitted.
    semi_major_axis : QuantityLike
        Orbital semi-major axis. Assumed in AU if unit is omitted.
    track : StellarXUVTrack
        Stellar high-energy evolution track.
    age : QuantityLike, optional
        System age at the end of integration. Default is 5.0 Gyr.
    eccentricity : float, optional
        Orbital eccentricity. Default is 0.0.
    m_star : QuantityLike, optional
        Host stellar mass. Defaults to `track.m_star`.
    eta : float or Sequence[float], optional
        Photoevaporation efficiency value or list of values. Default is 0.10.
    r_xuv_factor : float, optional
        Effective absorption radius ratio :math:`R_{\\rm xuv} / R_p`. Default is 1.0.
    apply_tidal_correction : bool, optional
        Whether to apply the Erkaev et al. (2007) tidal reduction factor. Default is True.
    t_start : QuantityLike, optional
        Start time of high-energy integration. Default is 5.0 Myr.
    n_steps : int, optional
        Number of logarithmic time steps for integration. Default is 5000.

    Returns
    -------
    table : QTable
        Astropy QTable with columns:
        - `eta`: Efficiency
        - `mass_lost`: Lost mass in Earth masses (:math:`M_\\oplus`)
        - `mass_fraction_lost`: Fraction :math:`\\Delta M / M_p`
        - `k_tide`: Tidal correction factor
    """
    mp = u.Quantity(m_planet, u.M_earth)
    rp = u.Quantity(r_planet, u.R_earth)
    a = u.Quantity(semi_major_axis, u.AU)
    t_end = u.Quantity(age, u.Gyr)
    t_0 = u.Quantity(t_start, u.Myr)
    ms = u.Quantity(m_star, u.M_sun) if m_star is not None else track.m_star

    rxuv = r_xuv_factor * rp

    k_tide = 1.0
    if apply_tidal_correction:
        k_tide = roche_lobe_correction_factor(a, mp, ms, rxuv)

    # Time grid in seconds
    t_grid_s = np.geomspace(t_0.to(u.s).value, t_end.to(u.s).value, n_steps)
    t_grid_myr = (t_grid_s * u.s).to(u.Myr)

    f_xuv = track.flux_xuv(t_grid_myr, a, eccentricity=eccentricity)

    eta_list = [eta] if isinstance(eta, (int, float)) else list(eta)
    rows = []

    for eff in eta_list:
        mdot = energy_limited_mass_loss_rate(
            flux_xuv=f_xuv,
            r_planet=rp,
            m_planet=mp,
            eta=eff,
            r_xuv=rxuv,
            k_tide=k_tide,
        )
        delta_m_g = np.trapezoid(mdot.to(u.g / u.s).value, t_grid_s) * u.g
        delta_m_earth = delta_m_g.to(u.M_earth)
        fraction_lost = (delta_m_earth / mp).decompose().value

        rows.append(
            {
                "eta": float(eff),
                "mass_lost": delta_m_earth,
                "mass_fraction_lost": float(fraction_lost),
                "k_tide": float(k_tide),
            }
        )

    tbl = QTable(rows)
    tbl.meta = {
        "m_planet": mp,
        "r_planet": rp,
        "semi_major_axis": a,
        "eccentricity": eccentricity,
        "m_star": ms,
        "age": t_end,
        "track": track.name or "StellarXUVTrack",
        "references": [
            "Watson1981",
            "Erkaev2007",
            "MurrayClay2009",
            "LopezFortney2013",
            "OwenWu2013",
        ],
    }
    return tbl


@cites(
    "Watson1981",
    "Erkaev2007",
    "MurrayClay2009",
    "LopezFortney2013",
    "OwenWu2013",
    "Valencia2006",
    "FortneyMarleyBarnes2007",
    "Ginzburg2018",
)
def photoevaporation_evolution(
    m_planet_init: QuantityLike,
    r_planet_init: QuantityLike,
    semi_major_axis: QuantityLike,
    track: StellarXUVTrack,
    m_env_init: QuantityLike | None = None,
    m_core: QuantityLike | None = None,
    age: QuantityLike = 5.0 * u.Gyr,
    eccentricity: float = 0.0,
    m_star: QuantityLike | None = None,
    eta: float | Callable[[u.Quantity], float] = 0.10,
    r_xuv_factor: float = 1.0,
    apply_tidal_correction: bool = True,
    radius_func: Callable[[u.Quantity, u.Quantity, u.Quantity], u.Quantity] | None = None,
    t_start: QuantityLike = 5.0 * u.Myr,
    max_step: QuantityLike | None = None,
) -> QTable:
    """Integrate dynamic atmospheric photoevaporation and envelope mass loss over time.

    Solves the coupled initial value problem:

    .. math::
        \\frac{d M_{\\rm env}}{dt} = -\\dot{M}\\left(M_p(t), R_p(t), F_{\\rm xuv}(t)\\right)

    If `radius_func` is not provided, the planet radius contracts linearly with envelope mass loss
    down to the bare rocky core radius :math:`R_{\\rm core} = (M_{\\rm core}/M_\\oplus)^{0.27} R_\\oplus`
    (Valencia et al. 2006; Fortney et al. 2007).

    Integration stops early if the planet loses its entire envelope (:math:`M_{\\rm env} \\le 0`).

    Parameters
    ----------
    m_planet_init : QuantityLike
        Initial total planet mass at `t_start`. Assumed in Earth masses if unit is omitted.
    r_planet_init : QuantityLike
        Initial planet radius. Assumed in Earth radii if unit is omitted.
    semi_major_axis : QuantityLike
        Orbital semi-major axis. Assumed in AU if unit is omitted.
    track : StellarXUVTrack
        Stellar high-energy evolution track.
    m_env_init : QuantityLike, optional
        Initial envelope mass. If not provided, assumed to be 5% of total mass or
        `m_planet_init - m_core`.
    m_core : QuantityLike, optional
        Underlying core mass. If not provided, computed as `m_planet_init - m_env_init`.
    age : QuantityLike, optional
        Final integration age. Default is 5.0 Gyr.
    eccentricity : float, optional
        Orbital eccentricity. Default is 0.0.
    m_star : QuantityLike, optional
        Host stellar mass. Defaults to `track.m_star`.
    eta : float or callable, optional
        Photoevaporation efficiency, or callable `eta(v_esc)`. Default is 0.10.
    r_xuv_factor : float, optional
        Effective absorption radius ratio :math:`R_{\\rm xuv} / R_p`. Default is 1.0.
    apply_tidal_correction : bool, optional
        Whether to apply the Erkaev et al. (2007) tidal reduction factor. Default is True.
    radius_func : callable, optional
        Callable `radius_func(t, m_total, m_env)` returning `u.Quantity` radius.
        If None, contracts toward `default_rocky_core_radius(m_core)`.
    t_start : QuantityLike, optional
        Start time of integration. Default is 5.0 Myr.
    max_step : QuantityLike, optional
        Maximum step size for `solve_ivp`. Default is 10.0 Myr.

    Returns
    -------
    table : QTable
        Time series table containing:
        - `time` in Gyr
        - `m_planet` in M_earth
        - `m_env` in M_earth
        - `m_lost` in M_earth
        - `radius` in R_earth
        - `f_xuv` in erg / (s * cm^2)
        - `dM_dt` in M_earth / Gyr
    """
    m_tot0 = u.Quantity(m_planet_init, u.M_earth)
    r_p0 = u.Quantity(r_planet_init, u.R_earth)
    a = u.Quantity(semi_major_axis, u.AU)
    t_start_q = u.Quantity(t_start, u.Myr)
    t_end_q = u.Quantity(age, u.Gyr)
    ms = u.Quantity(m_star, u.M_sun) if m_star is not None else track.m_star

    if m_env_init is not None and m_core is not None:
        m_env0 = u.Quantity(m_env_init, u.M_earth)
        m_core_q = u.Quantity(m_core, u.M_earth)
    elif m_env_init is not None:
        m_env0 = u.Quantity(m_env_init, u.M_earth)
        m_core_q = m_tot0 - m_env0
    elif m_core is not None:
        m_core_q = u.Quantity(m_core, u.M_earth)
        m_env0 = m_tot0 - m_core_q
    else:
        # Default 5% envelope
        m_env0 = 0.05 * m_tot0
        m_core_q = m_tot0 - m_env0

    if m_core_q < 0.0 * u.M_earth:
        raise ValueError(f"Core mass cannot be negative: {m_core_q}")
    if m_env0 < 0.0 * u.M_earth:
        raise ValueError(f"Initial envelope mass cannot be negative: {m_env0}")

    r_core_default = default_rocky_core_radius(m_core_q)

    # Times in Gyr
    t_start_gyr = t_start_q.to(u.Gyr).value
    t_end_gyr = t_end_q.to(u.Gyr).value

    def get_radius(
        t_curr: u.Quantity, m_tot_curr: u.Quantity, m_env_curr: u.Quantity
    ) -> u.Quantity:
        if radius_func is not None:
            return radius_func(t_curr, m_tot_curr, m_env_curr)
        if m_env0.value > 0.0:
            env_frac = max(float((m_env_curr / m_env0).decompose().value), 0.0)
            return r_core_default + (r_p0 - r_core_default) * env_frac
        return r_core_default

    def ode_system(t_gyr: float, y: list[float]) -> list[float]:
        m_env_val = max(y[0], 0.0)
        if m_env_val <= 0.0:
            return [0.0]

        m_env_curr = m_env_val * u.M_earth
        m_tot_curr = m_core_q + m_env_curr
        t_curr = t_gyr * u.Gyr

        r_curr = get_radius(t_curr, m_tot_curr, m_env_curr)
        rxuv = r_xuv_factor * r_curr

        k_tide = 1.0
        if apply_tidal_correction:
            k_tide = roche_lobe_correction_factor(a, m_tot_curr, ms, rxuv)

        f_xuv = track.flux_xuv(t_curr, a, eccentricity=eccentricity)

        if callable(eta):
            v_esc = escape_velocity(m_tot_curr, r_curr)
            eff = eta(v_esc)
        else:
            eff = eta

        mdot = energy_limited_mass_loss_rate(
            flux_xuv=f_xuv,
            r_planet=r_curr,
            m_planet=m_tot_curr,
            eta=eff,
            r_xuv=rxuv,
            k_tide=k_tide,
        )

        dm_dt_earth_per_gyr = mdot.to(u.M_earth / u.Gyr).value
        return [-dm_dt_earth_per_gyr]

    def envelope_depleted(t_gyr: float, y: list[float]) -> float:
        return y[0]

    envelope_depleted.terminal = True
    envelope_depleted.direction = -1

    max_step_val = (
        u.Quantity(max_step, u.Gyr).value
        if max_step is not None
        else (0.01 if (t_end_gyr - t_start_gyr) > 0.1 else (t_end_gyr - t_start_gyr) / 50.0)
    )

    sol = solve_ivp(
        ode_system,
        (t_start_gyr, t_end_gyr),
        [m_env0.value],
        events=[envelope_depleted],
        dense_output=True,
        rtol=1e-7,
        atol=1e-8,
        max_step=max_step_val,
    )

    t_eval = sol.t
    m_env_arr = np.maximum(sol.y[0], 0.0) * u.M_earth
    m_planet_arr = m_core_q + m_env_arr
    m_lost_arr = m_env0 - m_env_arr

    # Compute radius, f_xuv, and dM_dt for table
    t_q = t_eval * u.Gyr
    r_arr = u.Quantity(
        [get_radius(t_i, mp_i, me_i) for t_i, mp_i, me_i in zip(t_q, m_planet_arr, m_env_arr)]
    )

    f_xuv_arr = track.flux_xuv(t_q, a, eccentricity=eccentricity)

    rates = []
    for f_i, r_i, mp_i, me_i in zip(f_xuv_arr, r_arr, m_planet_arr, m_env_arr):
        if me_i.value <= 0.0:
            rates.append(0.0 * (u.M_earth / u.Gyr))
            continue
        rxuv_i = r_xuv_factor * r_i
        k_t_i = roche_lobe_correction_factor(a, mp_i, ms, rxuv_i) if apply_tidal_correction else 1.0
        eff_i = eta(escape_velocity(mp_i, r_i)) if callable(eta) else eta
        mdot_i = energy_limited_mass_loss_rate(
            flux_xuv=f_i,
            r_planet=r_i,
            m_planet=mp_i,
            eta=eff_i,
            r_xuv=rxuv_i,
            k_tide=k_t_i,
        )
        rates.append(mdot_i.to(u.M_earth / u.Gyr))

    table = QTable(
        {
            "time": t_q,
            "m_planet": m_planet_arr,
            "m_env": m_env_arr,
            "m_lost": m_lost_arr,
            "radius": r_arr,
            "f_xuv": f_xuv_arr,
            "dM_dt": u.Quantity(rates),
        }
    )

    table.meta = {
        "m_core": m_core_q,
        "r_core_default": r_core_default,
        "m_env_init": m_env0,
        "m_planet_init": m_tot0,
        "r_planet_init": r_p0,
        "semi_major_axis": a,
        "eccentricity": eccentricity,
        "m_star": ms,
        "track": track.name or "StellarXUVTrack",
        "solver_success": sol.success,
        "status": sol.status,
        "message": sol.message,
        "references": [
            "Watson1981",
            "Erkaev2007",
            "MurrayClay2009",
            "LopezFortney2013",
            "OwenWu2013",
            "Valencia2006",
            "FortneyMarleyBarnes2007",
            "Ginzburg2018",
        ],
    }

    if not sol.success and sol.status != 1:  # Status 1 is event termination
        logging.warning("Integration did not complete normally: %s", sol.message)

    return table
