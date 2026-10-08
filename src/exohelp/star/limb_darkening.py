"""
limb_darkening.py

Quadratic limb-darkening coefficients and priors for CHEOPS and TESS, interpolated from
the tabulated theoretical coefficients of Claret.

Sources:
- Claret (2017), Astronomy & Astrophysics, Vol. 600, A30 (TESS)
  https://ui.adsabs.harvard.edu/abs/2017A&A...600A..30C/abstract
- Claret (2021), Research Notes of the AAS, Vol. 5, 13 (CHEOPS)
  https://ui.adsabs.harvard.edu/abs/2021RNAAS...5...13C/abstract
- Kipping (2013), Monthly Notices of the Royal Astronomical Society, Vol. 435, pp. 2152-2160
  https://ui.adsabs.harvard.edu/abs/2013MNRAS.435.2152K/abstract

The tables are fetched once from VizieR and cached on disk. Coefficients are the
least-squares (LSM) fits of the quadratic law ``I(mu)/I(1) = 1 - u1 (1 - mu) - u2 (1 - mu)^2``.

Grid structure
--------------
- ATLAS (plane-parallel): 3500 K <= Teff <= 50 000 K, 0 <= log g <= 5. Metallicity
  ([M/H] = -5 ... +1) is tabulated only at microturbulence 2 km/s; the other
  microturbulences (0, 1, 4, 8 km/s) exist only at solar metallicity.
- PHOENIX-COND, r-method: 2300 K <= Teff <= 12 000 K, log g >= 2.5 (TESS) or >= 3 (CHEOPS),
  solar metallicity and microturbulence 2 km/s only. For TESS this is Claret (2017) table 16
  (table 15, the quasi-spherical fit, is not used). The CHEOPS table tabulates mu_cri, i.e.
  it is also an r-method fit. Both grids lack the nodes (2800 K, 6.0) and (3000 K, 5.0).

Interpolation is multilinear over (Teff, log g, [M/H]) on the table's own nodes, using only
the grid cell that contains the point. Points outside the grid, or in a cell with a missing
corner, are masked rather than extrapolated. Microturbulence is never interpolated; it must
be one of the tabulated values.

Choosing a model
----------------
The two grids agree to within ~0.03 in (q1, q2) around 4500-5000 K but diverge towards both
ends: ATLAS - PHOENIX reaches ~+0.05-0.10 in q2 at 3500 K and ~-0.05-0.10 in q2 at 6000-8000 K
(CHEOPS and TESS alike), i.e. comparable to the 0.1 prior width commonly adopted. Following
common practice:

- Teff < 3500 K: PHOENIX, the only grid that covers it.
- 3500 K <= Teff < 4000 K: PHOENIX is preferred; its spherical models and molecular opacities
  describe cool dwarfs better, which outweighs the lack of metallicity dependence.
- Teff >= 4000 K (FGK): ATLAS, the standard grid there and the only one with metallicity.

`recommended_model` implements this split (threshold `PHOENIX_TEFF_THRESHOLD`), and
``model="auto"`` in `sample_limb_darkening` / `limb_darkening_prior` applies it to the
star's central Teff. The threshold is a convention, not a sharp physical boundary: state the
grid used in the paper, and for stars near it (or wherever the coefficients drive a result)
consider the ATLAS - PHOENIX difference as an estimate of the model-atmosphere systematic.

Examples
--------
>>> from exohelp.star.limb_darkening import limb_darkening_prior
>>> limb_darkening_prior(5700, 100, 4.4, 0.1, 0.0, 0.1, band="CHEOPS")  # doctest: +SKIP
{'q1': (0.452..., 0.011...), 'q2': (0.319..., 0.011...)}
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.table import QTable
from scipy.interpolate import RegularGridInterpolator

from ..citations import REFERENCES, cites
from ..stats import truncated_normal
from ..type import QuantityLike

__all__ = [
    "claret_quadratic_coefficients",
    "limb_darkening_prior",
    "load_claret_table",
    "q_to_u",
    "recommended_model",
    "sample_limb_darkening",
    "u_to_q",
]

BANDS = ("CHEOPS", "TESS")
MODELS = ("ATLAS", "PHOENIX")

# Below this Teff [K], `recommended_model` (and ``model="auto"``) picks PHOENIX over ATLAS.
PHOENIX_TEFF_THRESHOLD = 4000.0

# Canonical column names of the tables returned by `load_claret_table`.
_COLUMNS = ["teff", "logg", "feh", "vturb", "u1", "u2"]


@dataclass(frozen=True)
class _ClaretTable:
    catalog: str
    reference: str
    description: str
    # VizieR column name -> canonical column name
    columns: dict[str, str]
    # VizieR column name -> required value, for catalogs that merge several tables
    filters: dict[str, str] = field(default_factory=dict)
    # Fixed microturbulence [km/s] for tables without a microturbulence column
    vturb: float | None = None


_CLARET_TABLES = {
    ("CHEOPS", "ATLAS"): _ClaretTable(
        catalog="J/other/RNAAS/5.13/table8",
        reference="Claret2021",
        description="Claret (2021), table 8: ATLAS, CHEOPS, LSM",
        columns={"Teff": "teff", "logg": "logg", "ZR": "feh", "Vel": "vturb", "a": "u1", "b": "u2"},
    ),
    ("CHEOPS", "PHOENIX"): _ClaretTable(
        catalog="J/other/RNAAS/5.13/table2",
        reference="Claret2021",
        description="Claret (2021), table 2: PHOENIX-COND, CHEOPS, LSM",
        columns={"Teff": "teff", "logg": "logg", "ZR": "feh", "a": "u1", "b": "u2"},
        vturb=2.0,
    ),
    ("TESS", "ATLAS"): _ClaretTable(
        catalog="J/A+A/600/A30/table25",
        reference="Claret2017",
        description="Claret (2017), table 25: ATLAS, TESS, LSM",
        columns={
            "Teff": "teff",
            "logg": "logg",
            "Z": "feh",
            "xi": "vturb",
            "aLSM": "u1",
            "bLSM": "u2",
        },
    ),
    # VizieR merges Claret (2017) tables 4, 5, 15, 16 into `tableab`: Type "q" is the
    # quasi-spherical fit, "r" the r-method; Mod "PC" is PHOENIX-COND, "PD" PHOENIX-DRIFT.
    # Table 16 (r-method, COND) is used.
    ("TESS", "PHOENIX"): _ClaretTable(
        catalog="J/A+A/600/A30/tableab",
        reference="Claret2017",
        description="Claret (2017), table 16: PHOENIX-COND r-method, TESS, LSM",
        columns={"Teff": "teff", "logg": "logg", "Z": "feh", "aLSM": "u1", "bLSM": "u2"},
        filters={"Type": "r", "Mod": "PC"},
        vturb=2.0,
    ),
}


def _claret_table(band: str, model: str) -> _ClaretTable:
    key = (band.upper(), model.upper())
    if key not in _CLARET_TABLES:
        raise ValueError(
            f"Unsupported band/model {band!r}/{model!r}; choose from {BANDS} x {MODELS}."
        )
    return _CLARET_TABLES[key]


def _default_cache_dir() -> Path:
    from astropy.config import get_cache_dir_path

    return get_cache_dir_path() / "exohelp" / "limb_darkening"


def _download_claret_table(table: _ClaretTable, timeout: float) -> pd.DataFrame:
    from astroquery.vizier import Vizier

    viz = Vizier(columns=["**"], row_limit=-1, timeout=timeout)
    try:
        result = viz.get_catalogs(table.catalog)
    except Exception as e:
        raise RuntimeError(f"Failed to download {table.catalog} from VizieR: {e}") from e
    if len(result) == 0:
        raise RuntimeError(f"VizieR returned no table for {table.catalog}.")
    df = result[0].to_pandas()

    for col, value in table.filters.items():
        df = df[df[col].astype(str).str.strip() == value]
    df = df[list(table.columns)].rename(columns=table.columns)
    if table.vturb is not None:
        df["vturb"] = table.vturb
    return df[_COLUMNS].reset_index(drop=True)


def load_claret_table(
    band: str,
    model: str = "ATLAS",
    cache_dir: str | Path | None = None,
    timeout: float = 300,
) -> pd.DataFrame:
    """Load a Claret quadratic limb-darkening table, downloading it from VizieR on first use.

    Parameters
    ----------
    band : {"CHEOPS", "TESS"}
        Photometric band.
    model : {"ATLAS", "PHOENIX"}
        Model atmospheres the coefficients were computed from.
    cache_dir : str or Path, optional
        Directory holding the cached tables. Defaults to ``exohelp/limb_darkening`` in the
        astropy cache directory.
    timeout : float
        VizieR timeout in seconds.

    Returns
    -------
    pandas.DataFrame
        Columns ``teff`` [K], ``logg`` [dex, cgs], ``feh`` [dex], ``vturb`` [km/s], ``u1``,
        ``u2`` (the tabulated ``a``, ``b``), one row per grid node.
    """
    table = _claret_table(band, model)
    cache_dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    path = cache_dir / f"claret_{band.lower()}_{model.lower()}.csv"

    if path.exists():
        return pd.read_csv(path)[_COLUMNS]

    df = _download_claret_table(table, timeout=timeout)
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


class _ClaretGrid:
    """Multilinear interpolator over the (Teff, log g, [M/H]) nodes of one Claret table slice.

    Missing nodes are allowed: a point is masked whenever any corner of its grid cell
    that carries non-zero interpolation weight is missing, or it lies outside the grid.
    """

    def __init__(self, df: pd.DataFrame):
        df = df.copy()
        for col, decimals in [("teff", 1), ("logg", 3), ("feh", 3)]:
            df[col] = df[col].astype(float).round(decimals)
        if df.duplicated(["teff", "logg", "feh"]).any():
            raise ValueError("Duplicate (teff, logg, feh) nodes; filter the table first.")

        self.nodes = {col: np.unique(df[col]) for col in ("teff", "logg", "feh")}
        # Axes with a single node (e.g. solar-only grids) cannot be interpolated over.
        self.axes = [col for col, nodes in self.nodes.items() if len(nodes) > 1]

        shape = tuple(len(self.nodes[col]) for col in self.axes)
        index = tuple(np.searchsorted(self.nodes[col], df[col]) for col in self.axes)
        values = np.full((*shape, 2), np.nan)
        values[index] = df[["u1", "u2"]].to_numpy()

        missing = np.isnan(values[..., 0]) | np.isnan(values[..., 1])
        points = [self.nodes[col] for col in self.axes]
        kwargs = {"bounds_error": False, "fill_value": np.nan}
        self._values = RegularGridInterpolator(
            points, np.where(missing[..., None], 0.0, values), **kwargs
        )
        self._missing = RegularGridInterpolator(points, missing.astype(float), **kwargs)

    def __call__(self, teff, logg, feh) -> tuple[np.ma.MaskedArray, np.ma.MaskedArray]:
        inputs = {"teff": teff, "logg": logg, "feh": feh}
        for col in ("teff", "logg", "feh"):
            if col not in self.axes and np.any(inputs[col] != self.nodes[col][0]):
                raise ValueError(
                    f"This grid is tabulated only at {col} = {self.nodes[col][0]:g}; "
                    f"got {col} != {self.nodes[col][0]:g}."
                )

        broadcast = np.broadcast_arrays(
            *(np.asarray(inputs[col], dtype=float) for col in self.axes)
        )
        shape = broadcast[0].shape
        xi = np.stack([b.ravel() for b in broadcast], axis=-1)
        u_values = self._values(xi).reshape((*shape, 2))
        # Weight on missing corners; NaN outside the grid.
        missing_weight = self._missing(xi).reshape(shape)
        mask = ~(missing_weight < 1e-12)

        u1 = np.ma.masked_array(u_values[..., 0], mask=mask)
        u2 = np.ma.masked_array(u_values[..., 1], mask=mask)
        return u1, u2


@functools.lru_cache(maxsize=16)
def _get_grid(band: str, model: str, vturb: float, cache_dir: str | None) -> _ClaretGrid:
    df = load_claret_table(band, model, cache_dir=cache_dir)
    available = np.unique(df["vturb"])
    if not np.any(np.isclose(available, vturb)):
        raise ValueError(
            f"Microturbulence {vturb:g} km/s is not tabulated for {band}/{model}; "
            f"available values: {available.tolist()} km/s."
        )
    return _ClaretGrid(df[np.isclose(df["vturb"], vturb)])


def _evaluate(grid: _ClaretGrid, teff, logg, feh):
    return grid(teff, logg, feh)


# One evaluator per paper, so `citation_tracker()` records only the table actually used.
_EVALUATORS = {key: cites(key)(_evaluate) for key in ("Claret2017", "Claret2021")}


def _to_value(x: QuantityLike, unit: u.UnitBase | str) -> np.ndarray:
    if isinstance(x, u.Quantity):
        return np.asarray(x.to_value(unit), dtype=float)
    return np.asarray(x, dtype=float)


def _check_finite(**kwargs) -> None:
    for name, value in kwargs.items():
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be finite; got {value!r}.")


def claret_quadratic_coefficients(
    teff: QuantityLike,
    logg: QuantityLike,
    feh: QuantityLike | None = None,
    *,
    band: str,
    model: str = "ATLAS",
    vturb: float = 2.0,
    cache_dir: str | Path | None = None,
) -> tuple[np.ma.MaskedArray, np.ma.MaskedArray]:
    """Quadratic limb-darkening coefficients (u1, u2) interpolated from the Claret tables.

    Parameters
    ----------
    teff : QuantityLike
        Effective temperature [K].
    logg : QuantityLike
        Surface gravity [dex, cgs].
    feh : QuantityLike, optional
        Metallicity [M/H] [dex]. ``None`` uses solar metallicity. Only the ATLAS grid at
        ``vturb = 2`` km/s is tabulated at non-solar metallicity; anything else raises.
    band : {"CHEOPS", "TESS"}
        Photometric band (Claret 2021 for CHEOPS, Claret 2017 for TESS).
    model : {"ATLAS", "PHOENIX"}
        Model atmospheres. PHOENIX-COND covers cooler stars (down to 2300 K); ATLAS covers
        3500-50 000 K and non-solar metallicities. See "Choosing a model" in the module
        docstring and `recommended_model`; ``"auto"`` is only accepted by
        `sample_limb_darkening` and `limb_darkening_prior`, which select one grid per star.
    vturb : float
        Microturbulent velocity [km/s]; must be a tabulated value (ATLAS: 0, 1, 2, 4, 8
        at solar metallicity, 2 otherwise; PHOENIX: 2).
    cache_dir : str or Path, optional
        See `load_claret_table`.

    Returns
    -------
    u1, u2 : numpy.ma.MaskedArray
        Coefficients, masked outside the tabulated grid.

    Raises
    ------
    ValueError
        If any input is NaN or infinite, or the requested metallicity or microturbulence
        is not tabulated.
    """
    table = _claret_table(band, model)
    teff = _to_value(teff, u.K)
    logg = _to_value(logg, u.dex)
    feh = np.asarray(0.0) if feh is None else _to_value(feh, u.dex)
    _check_finite(teff=teff, logg=logg, feh=feh)

    cache_key = None if cache_dir is None else str(cache_dir)
    grid = _get_grid(band.upper(), model.upper(), float(vturb), cache_key)
    return _EVALUATORS[table.reference](grid, teff, logg, feh)


# Not decorated with `@cites` so that a call only records the band's own paper (see
# `_EVALUATORS`), but `exohelp.cite(claret_quadratic_coefficients)` still lists both.
claret_quadratic_coefficients.references = (  # type: ignore[attr-defined]
    REFERENCES["Claret2017"],
    REFERENCES["Claret2021"],
)


def recommended_model(teff: QuantityLike) -> str:
    """Recommended Claret grid for a star: ``"PHOENIX"`` below `PHOENIX_TEFF_THRESHOLD`, else ATLAS.

    See "Choosing a model" in the module docstring for the rationale.

    Parameters
    ----------
    teff : QuantityLike
        The star's (central) effective temperature [K]; must be a finite scalar.

    Examples
    --------
    >>> recommended_model(3200), recommended_model(5700)
    ('PHOENIX', 'ATLAS')
    """
    teff = _to_value(teff, u.K)
    if teff.ndim != 0:
        raise ValueError("recommended_model takes a single (scalar) Teff.")
    _check_finite(teff=teff)
    return "PHOENIX" if teff < PHOENIX_TEFF_THRESHOLD else "ATLAS"


def _resolve_model(model: str, teff: QuantityLike, feh: QuantityLike | None):
    """Resolve ``model="auto"``; PHOENIX is solar-only, so [M/H] is dropped with a warning."""
    if model.lower() != "auto":
        return model.upper(), feh
    model = recommended_model(teff)
    if model == "PHOENIX" and feh is not None:
        warnings.warn(
            f"model='auto' selected PHOENIX (Teff < {PHOENIX_TEFF_THRESHOLD:g} K), which is "
            "tabulated at solar metallicity only; the given [M/H] is ignored.",
            stacklevel=3,
        )
        feh = None
    return model, feh


@cites("Kipping2013")
def u_to_q(u1: QuantityLike, u2: QuantityLike) -> tuple[np.ndarray, np.ndarray]:
    """Convert quadratic coefficients (u1, u2) to the Kipping (2013) (q1, q2) parametrization.

    ``q1 = (u1 + u2)^2`` and ``q2 = u1 / (2 (u1 + u2))`` (Kipping 2013, Eqs. 17-18).
    Physically valid limb darkening has ``0 <= q1, q2 <= 1``.

    Examples
    --------
    >>> q1, q2 = u_to_q(0.4, 0.2)
    >>> print(f"{q1:.3f} {q2:.3f}")
    0.360 0.333
    """
    u1, u2 = np.asanyarray(u1), np.asanyarray(u2)
    q1 = (u1 + u2) ** 2
    q2 = u1 / (2 * (u1 + u2))
    return q1, q2


@cites("Kipping2013")
def q_to_u(q1: QuantityLike, q2: QuantityLike) -> tuple[np.ndarray, np.ndarray]:
    """Convert Kipping (2013) (q1, q2) to quadratic coefficients (u1, u2) (Eqs. 15-16).

    Examples
    --------
    >>> u1, u2 = q_to_u(0.36, 1 / 3)
    >>> print(f"{u1:.3f} {u2:.3f}")
    0.400 0.200
    """
    q1, q2 = np.asanyarray(q1), np.asanyarray(q2)
    u1 = 2 * np.sqrt(q1) * q2
    u2 = np.sqrt(q1) * (1 - 2 * q2)
    return u1, u2


def sample_limb_darkening(
    teff: QuantityLike,
    teff_err: QuantityLike,
    logg: QuantityLike,
    logg_err: QuantityLike,
    feh: QuantityLike | None = None,
    feh_err: QuantityLike = 0.0,
    *,
    band: str,
    model: str = "ATLAS",
    vturb: float = 2.0,
    n_samples: int = 100_000,
    seed: int | None = None,
    cache_dir: str | Path | None = None,
) -> QTable:
    """Monte Carlo propagation of stellar-parameter uncertainties to limb-darkening coefficients.

    Teff is drawn from a normal truncated at 0 K, log g and [M/H] from normals, and each
    sample is interpolated with `claret_quadratic_coefficients`.

    Parameters
    ----------
    teff, teff_err : QuantityLike
        Effective temperature and its 1-sigma uncertainty [K].
    logg, logg_err : QuantityLike
        Surface gravity and its 1-sigma uncertainty [dex, cgs].
    feh, feh_err : QuantityLike, optional
        Metallicity and its 1-sigma uncertainty [dex]. ``feh=None`` uses solar
        metallicity without scatter (required for PHOENIX).
    model : {"ATLAS", "PHOENIX", "auto"}
        Model atmospheres; ``"auto"`` applies `recommended_model` to the central `teff`
        (all samples use the same grid). If that selects PHOENIX, `feh` is ignored with a
        warning. The grid used is recorded in ``meta["model"]``.
    band, vturb, cache_dir
        See `claret_quadratic_coefficients`.
    n_samples : int
        Number of Monte Carlo samples.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    QTable
        Columns ``teff``, ``logg``, ``feh``, ``u1``, ``u2``, ``q1``, ``q2``. Coefficients are
        masked where a sample falls outside the grid; ``q1``/``q2`` are additionally masked
        outside the physical range [0, 1]. ``meta["masked_fraction"]`` is the fraction of
        samples with masked ``q1``/``q2``.
    """
    model, feh = _resolve_model(model, teff, feh)
    table = _claret_table(band, model)
    rng = np.random.default_rng(seed)

    teff, teff_err = _to_value(teff, u.K), _to_value(teff_err, u.K)
    logg, logg_err = _to_value(logg, u.dex), _to_value(logg_err, u.dex)
    _check_finite(teff=teff, teff_err=teff_err, logg=logg, logg_err=logg_err)

    teff_s = truncated_normal(teff, teff_err, n_samples, lower=0.0, rng=rng)
    logg_s = rng.normal(logg, logg_err, n_samples)
    if feh is None:
        feh_s = None
    else:
        feh, feh_err = _to_value(feh, u.dex), _to_value(feh_err, u.dex)
        _check_finite(feh=feh, feh_err=feh_err)
        feh_s = rng.normal(feh, feh_err, n_samples)

    u1, u2 = claret_quadratic_coefficients(
        teff_s, logg_s, feh_s, band=band, model=model, vturb=vturb, cache_dir=cache_dir
    )
    q1, q2 = u_to_q(u1, u2)
    unphysical = ~((q1 >= 0) & (q1 <= 1) & (q2 >= 0) & (q2 <= 1)).filled(False)
    q1 = np.ma.masked_array(q1, mask=np.ma.getmaskarray(q1) | unphysical)
    q2 = np.ma.masked_array(q2, mask=np.ma.getmaskarray(q2) | unphysical)

    out = QTable(
        [teff_s, logg_s, np.zeros(n_samples) if feh_s is None else feh_s, u1, u2, q1, q2],
        names=["teff", "logg", "feh", "u1", "u2", "q1", "q2"],
    )
    claret = table.reference
    descriptions = {
        "u1": f"Quadratic limb-darkening coefficient u1 ({table.description})",
        "u2": f"Quadratic limb-darkening coefficient u2 ({table.description})",
        "q1": "Kipping (2013) q1 = (u1 + u2)^2",
        "q2": "Kipping (2013) q2 = u1 / (2 (u1 + u2))",
    }
    references = {"u1": [claret], "u2": [claret], "q1": [claret, "Kipping2013"]}
    references["q2"] = references["q1"]
    for name, description in descriptions.items():
        out[name].description = description  # type: ignore[union-attr]
        out[name].info.meta = {"references": references[name]}  # type: ignore[union-attr]

    out.meta.update(
        {
            "band": band.upper(),
            "model": model,
            "vturb": float(vturb),
            "masked_fraction": float(np.mean(np.ma.getmaskarray(q1))),
            "references": [claret, "Kipping2013"],
        }
    )
    return out


def limb_darkening_prior(
    teff: QuantityLike,
    teff_err: QuantityLike,
    logg: QuantityLike,
    logg_err: QuantityLike,
    feh: QuantityLike | None = None,
    feh_err: QuantityLike = 0.0,
    *,
    band: str,
    model: str = "ATLAS",
    vturb: float = 2.0,
    parametrization: str = "q",
    min_sigma: float = 0.0,
    n_samples: int = 100_000,
    seed: int | None = 42,
    cache_dir: str | Path | None = None,
) -> dict[str, tuple[float, float]]:
    """Gaussian prior (mean, sigma) on the quadratic limb-darkening coefficients of a star.

    Built from `sample_limb_darkening`: the mean and standard deviation of the valid samples,
    with sigma floored at `min_sigma`. A warning is issued when samples fall outside the grid
    (or, for ``parametrization="q"``, outside [0, 1]), since the statistics are then computed
    from the in-grid samples only and are biased towards the grid interior.

    Parameters
    ----------
    teff, teff_err, logg, logg_err, feh, feh_err, band, model, vturb, n_samples, seed, cache_dir
        See `sample_limb_darkening`. With ``model="auto"``, `recommended_model` tells which
        grid was used.
    parametrization : {"q", "u"}
        Return a prior on Kipping (2013) (q1, q2) or on (u1, u2).
    min_sigma : float
        Lower bound on each returned sigma. The propagated scatter only reflects the stellar
        parameter uncertainties, not the systematic error of the model atmospheres, so a
        floor (e.g. 0.1) is commonly applied.

    Returns
    -------
    dict
        ``{"q1": (mu, sigma), "q2": (mu, sigma)}`` or the same with ``"u1"``, ``"u2"``.
    """
    if parametrization not in ("q", "u"):
        raise ValueError(f"parametrization must be 'q' or 'u', got {parametrization!r}.")
    samples = sample_limb_darkening(
        teff, teff_err, logg, logg_err, feh, feh_err,
        band=band, model=model, vturb=vturb, n_samples=n_samples, seed=seed, cache_dir=cache_dir,
    )  # fmt: skip
    model = samples.meta["model"]
    names = [f"{parametrization}1", f"{parametrization}2"]
    mask = np.ma.getmaskarray(samples[names[0]]) | np.ma.getmaskarray(samples[names[1]])

    if mask.all():
        raise ValueError(
            f"All samples fall outside the {band}/{model} grid "
            f"(vturb = {vturb:g} km/s) or the physical range."
        )
    if mask.any():
        warnings.warn(
            f"{mask.mean():.1%} of the limb-darkening samples fall outside the {band}/{model} "
            "grid or the physical range; the prior is computed from the remaining samples.",
            stacklevel=2,
        )

    prior = {}
    for name in names:
        values = np.ma.getdata(samples[name])[~mask]
        prior[name] = (float(np.mean(values)), float(max(np.std(values), min_sigma)))
    return prior
