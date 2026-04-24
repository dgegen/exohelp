import re

import astropy.constants as const
import astropy.units as u
import numpy as np
from astropy.table import QTable

from ..body import bulk_density, log_surface_gravity, surface_gravity
from ..kepler import keplers_third_law
from ..type import QuantityLike
from .properties import (
    equilibrium_temperature,
    hill_sphere_radius,
    insolation_flux,
    periastron_distance,
)
from .rv import planet_mass_from_rv
from .rv import rv_semi_amplitude as _rv_semi_amplitude
from .spectroscopy import (
    emission_spectroscopy_metric,
    scale_height as _scale_height,
    transmission_signal_size,
    transmission_spectroscopy_metric,
)
from .transit import orbital_inclination, transit_quantities

__all__ = ["DERIVED_LATEX_PATTERNS", "derived_planet_quantities"]


DERIVED_LATEX_PATTERNS = [
    (r"^k$", r"k_{planet_index}"),
    (r"^k_p1$", r"k_{planet_index}"),
    (r"^a$", r"a_{planet_index}"),
    (r"^a_over_r_star$", r"(a_{planet_index}/R_\star)"),
    (r"^inclination$", r"i_{planet_index}"),
    (r"^insolation$", r"S_{planet_index}"),
    (r"^surface_gravity$", r"g_{planet_index}"),
    (r"^log_surface_gravity$", r"\log g_{planet_index}"),
    (r"^scale_height$", r"H_{planet_index}"),
    (r"^hill_sphere$", r"r_{\mathrm{H}, planet_index}"),
    (r"^transit_depth$", r"\delta_{planet_index}"),
    (r"^transit_duration_total$", r"T_{14, planet_index}"),
    (r"^transit_duration_flat$", r"T_{23, planet_index}"),
    (r"^transit_duration_ingress$", r"\tau_{\mathrm{ing}, planet_index}"),
    (r"^transit_probability$", r"P_{\mathrm{tr}, planet_index}"),
    (r"^occultation_probability$", r"P_{\mathrm{occ}, planet_index}"),
    (r"^transmission_signal_1H$", r"\mathrm{Amp}_{\mathrm{transit}, planet_index}"),
    (r"^teq$", r"T_{\mathrm{eq}, planet_index}"),
    (r"^teff_star$", r"T_{\mathrm{eff}, \star}"),
    (r"^logg_star$", r"\log g_\star"),
    (r"^j_mag$", r"J_{\mathrm{mag}}"),
    (r"^rv_semi_amplitude$", r"K_{planet_index}"),
    (r"^m_planet$", r"M_{planet_index}"),
    (r"^bulk_density$", r"\rho_{planet_index}"),
    (r"^periastron_distance$", r"q_{planet_index}"),
    (r"^eclipse_timing_offset$", r"\Delta t_{\mathrm{ecl}, planet_index}"),
    (r"^tsm$", r"\mathrm{TSM}_{planet_index}"),
    (r"^esm$", r"\mathrm{ESM}_{planet_index}"),
]


def _normalize_planet_index(planet_index: str | int | None) -> str | None:
    if planet_index is None:
        return None
    idx = str(planet_index).strip()
    return idx if idx else None


def _column_with_planet_index(name: str, planet_index: str | None) -> str:
    if planet_index is None:
        return name
    return f"{name}_{planet_index}"


def _latex_from_template(template: str, planet_index: str | None) -> str:
    if planet_index is None:
        return (
            template.replace("_{planet_index}", "")
            .replace(", planet_index", "")
            .replace(" planet_index", "")
        )
    return template.replace("planet_index", planet_index)


def _latex_symbol_for_column(column_name: str, planet_index: str | None) -> str:
    base_name = column_name
    if planet_index is not None and base_name.endswith(f"_{planet_index}"):
        base_name = base_name[: -(len(planet_index) + 1)]

    for pattern, template in DERIVED_LATEX_PATTERNS:
        if re.match(pattern, base_name):
            return _latex_from_template(template, planet_index)
    return base_name.replace("_", r"\_")


def derived_planet_quantities(
    period: QuantityLike,
    r_planet: QuantityLike,
    r_star: QuantityLike = 1.0,
    m_star: QuantityLike = 1.0,
    b: float | np.ndarray = 0.0,
    eccentricity: float | np.ndarray = 0.0,
    omega: QuantityLike = 90.0,
    *,
    teff_star: QuantityLike | None = None,
    luminosity: QuantityLike | None = None,
    rv_semi_amplitude: QuantityLike | None = None,
    m_planet: QuantityLike | None = None,
    j_mag: QuantityLike | None = None,
    k_mag: QuantityLike | None = None,
    bond_albedo: float | np.ndarray = 0.0,
    planet_index: str | int | None = None,
) -> QTable:
    """Derive all computable planet quantities from transit fit parameters and optional extras.

    Starts from the standard transit observables and adds physical and observational
    quantities as optional inputs become available. Columns are only added when all
    their prerequisites are present.

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
        Impact parameter. Default is 0 (central transit).
    eccentricity : float or array-like
        Orbital eccentricity. Default is 0 (circular).
    omega : QuantityLike
        Argument of periastron in degrees. Default is 90°.
    teff_star : QuantityLike, optional
        Stellar effective temperature in K. Enables: equilibrium temperature,
        insolation flux, scale height, transmission/emission spectroscopy metrics.
    luminosity : QuantityLike, optional
        Stellar luminosity in Solar luminosities. If omitted and ``teff_star`` is
        given, luminosity is derived from ``teff_star`` and ``r_star`` via
        L = 4π R² sigma T⁴. Enables: insolation flux.
    rv_semi_amplitude : QuantityLike, optional
        Observed RV semi-amplitude K in m/s. Enables: planet mass, bulk density,
        surface gravity, Hill sphere, scale height, spectroscopy metrics.
        Mutually exclusive with ``m_planet``.
    m_planet : QuantityLike, optional
        Planet mass in Earth masses. Enables: predicted RV semi-amplitude,
        bulk density, surface gravity, Hill sphere, scale height, spectroscopy metrics.
        Mutually exclusive with ``rv_semi_amplitude``.
    j_mag : QuantityLike, optional
        Host-star J-band magnitude. Enables: TSM (requires ``m_planet`` / ``rv_semi_amplitude``
        and ``teff_star``).
    k_mag : QuantityLike, optional
        Host-star K-band magnitude. Enables: ESM (requires ``m_planet`` / ``rv_semi_amplitude``
        and ``teff_star``).
    bond_albedo : float or array-like
        Bond albedo used for equilibrium temperature. Default is 0.
    planet_index : str, int, or None
        Optional planet identifier appended to all output column names, e.g. ``b`` or ``1``.
        If provided, a column like ``transit_depth`` becomes ``transit_depth_b``.

    Returns
    -------
    table : QTable
        All computable derived quantities with units and descriptions.
        Always contains the transit geometry columns from `transit_quantities`.

    Examples
    --------
    >>> from exohelp.planet.summary import derived_planet_quantities
    >>> t = derived_planet_quantities(3.0, 2.0, r_star=0.5, m_star=0.5, teff_star=3800,
    ...                               m_planet=10.0, j_mag=8.0, k_mag=7.5)
    >>> t.colnames  # doctest: +ELLIPSIS
    [...]
    """
    period = u.Quantity(period, "day")
    r_planet = u.Quantity(r_planet, "R_earth")
    r_star = u.Quantity(r_star, "R_sun")
    m_star = u.Quantity(m_star, "M_sun")

    planet_index = _normalize_planet_index(planet_index)

    a = keplers_third_law(period=period, mass=m_star)
    incl = orbital_inclination(a, r_star, b, eccentricity, omega)

    table = transit_quantities(period, r_planet, r_star, m_star, b, eccentricity, omega)
    if planet_index is not None:
        old_names = list(table.colnames)
        new_names = [_column_with_planet_index(name, planet_index) for name in old_names]
        table.rename_columns(old_names, new_names)

    _d = u.dimensionless_unscaled

    def _add(name: str, value, description: str) -> None:
        column_name = _column_with_planet_index(name, planet_index)
        table[column_name] = np.atleast_1d(value)
        table[column_name].info.description = description  # type: ignore[union-attr]

    # --- insolation flux and equilibrium temperature ---
    _lum = None
    if luminosity is not None:
        _lum = u.Quantity(luminosity, "L_sun")
    elif teff_star is not None:
        _teff = u.Quantity(teff_star, "K")
        _lum = (4 * np.pi * r_star.to("m") ** 2 * const.sigma_sb * _teff**4).to("L_sun")  # type: ignore[attr-defined]

    if _lum is not None:
        _add(
            "insolation",
            insolation_flux(_lum, a),
            "Insolation flux relative to Earth's S/S⊕ = (L★/L⊙)(AU/a)²",
        )

    _teq = None
    if teff_star is not None:
        _teff = u.Quantity(teff_star, "K")
        _teq = equilibrium_temperature(
            _teff, semi_major_axis=a, r_star=r_star, bond_albedo=bond_albedo
        )
        _add(
            "teq",
            _teq.to("K"),
            f"Equilibrium temperature T_eq = T★ √(R★/2a) (1-A)^(1/4), A={bond_albedo}",
        )

    if eccentricity > 0:
        _add(
            "periastron_distance",
            periastron_distance(a, eccentricity).to("AU"),
            "Periastron distance q = a(1-e)",
        )

    # --- planet mass (from RV or explicit) ---
    _m_planet = None
    if rv_semi_amplitude is not None:
        _K = (  # noqa: N806
            rv_semi_amplitude
            if isinstance(rv_semi_amplitude, u.Quantity)
            else u.Quantity(rv_semi_amplitude, "m/s")
        )
        _m_planet = planet_mass_from_rv(_K, period, eccentricity, m_star, incl)
        _add("rv_semi_amplitude", _K.to("m/s"), "Observed RV semi-amplitude K")
        _add(
            "m_planet",
            _m_planet.to("M_earth"),
            "Planet mass from RV semi-amplitude (Lovis & Fischer 2010)",
        )
    elif m_planet is not None:
        _m_planet = u.Quantity(m_planet, "M_earth")
        _add("m_planet", _m_planet.to("M_earth"), "Planet mass")
        _K = _rv_semi_amplitude(_m_planet, period, eccentricity, m_star, incl)  # noqa: N806
        _add(
            "rv_semi_amplitude", _K.to("m/s"), "Predicted RV semi-amplitude (Lovis & Fischer 2010)"
        )

    if _m_planet is not None:
        _add("bulk_density", bulk_density(_m_planet, r_planet), "Bulk density")
        _g = surface_gravity(_m_planet, r_planet)
        _add("surface_gravity", _g.to("m/s^2"), "Surface gravity g = G M_p / R_p²")
        _add(
            "log_surface_gravity",
            np.asarray(log_surface_gravity(_m_planet, r_planet)) * _d,
            "Log surface gravity log10(g/[cm s^-2])",
        )
        _add(
            "hill_sphere",
            hill_sphere_radius(a, _m_planet, m_star, eccentricity).to("AU"),
            "Hill sphere radius r_H = a(1-e)(m_p/3M★)^(1/3) (Hamilton & Burns 1992)",
        )

        if _teq is not None:
            _H = _scale_height(_teq, _g)  # noqa: N806
            _add(
                "scale_height",
                _H.to("km"),
                "Atmospheric scale height H = k_B T_eq / (μ m_H g), μ=2.3",
            )
            _sig = transmission_signal_size(_H, r_planet, r_star)
            _add(
                "transmission_signal_1H",
                np.asarray(_sig) * _d,
                "Single-scale-height transmission signal ΔD = 2H R_p / R★² (ppm)",
            )

            if j_mag is not None:
                _jmag = j_mag if isinstance(j_mag, u.Quantity) else u.Quantity(j_mag, "mag")
                _tsm = transmission_spectroscopy_metric(r_planet, _m_planet, _teq, r_star, _jmag)
                _add(
                    "tsm",
                    np.asarray(_tsm) * _d,
                    "Transmission Spectroscopy Metric (Kempton et al. 2018)",
                )

            if k_mag is not None:
                _kmag = k_mag if isinstance(k_mag, u.Quantity) else u.Quantity(k_mag, "mag")
                _esm = emission_spectroscopy_metric(r_planet, _teq, r_star, _kmag, _teff)
                _add(
                    "esm",
                    np.asarray(_esm) * _d,
                    "Emission Spectroscopy Metric at 7.5 µm (Kempton et al. 2018)",
                )

    if table.meta is None:
        table.meta = {}
    table.meta["latex_names"] = {
        name: _latex_symbol_for_column(name, planet_index) for name in table.colnames
    }
    return table
