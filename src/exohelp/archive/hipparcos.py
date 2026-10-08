"""Hipparcos cross-matching and Hipparcos-Gaia astrometric combinability.

Assesses whether stars have Hipparcos observations (Perryman et al. 1997) and
whether their astrometry can be combined with Gaia via the Hipparcos-Gaia
Catalog of Accelerations (HGCA; Brandt 2018, 2021).
"""

from __future__ import annotations

import logging
import re
from typing import NamedTuple

from ..citations import cites
from ..planet.astrometry import delta_chi2_to_sigma

logger = logging.getLogger("exohelp.archive.hipparcos")

__all__ = [
    "HGCARecord",
    "HipparcosStatus",
    "check_hipparcos_gaia_combinable",
    "extract_hip_id",
    "in_hipparcos_by_name",
    "query_hgca",
]

_HIP_PREFIX_RE = re.compile(r"^HIP\s*(\d+)$", re.IGNORECASE)
_DIGITS_RE = re.compile(r"^\d+$")


class HGCARecord(NamedTuple):
    """Calibrated proper motions and acceleration from HGCA (Brandt 2021).

    Parameters
    ----------
    hip_number : int
        Hipparcos catalog number.
    gaia_source_id : int | None
        Gaia EDR3/DR3 source identifier.
    chisq : float | None
        Chi-squared value testing constant proper motion (df=2).
    significance_sigma : float | None
        Astrometric acceleration significance in equivalent Gaussian standard deviations.
    pmra_hip : float | None
        Hipparcos epoch proper motion in RA* = pmRA * cos(dec) (mas/yr).
    pmra_hip_error : float | None
        Uncertainty on pmra_hip (mas/yr).
    pmdec_hip : float | None
        Hipparcos epoch proper motion in Dec (mas/yr).
    pmdec_hip_error : float | None
        Uncertainty on pmdec_hip (mas/yr).
    pmra_gaia : float | None
        Gaia epoch proper motion in RA* (mas/yr).
    pmra_gaia_error : float | None
        Uncertainty on pmra_gaia (mas/yr).
    pmdec_gaia : float | None
        Gaia epoch proper motion in Dec (mas/yr).
    pmdec_gaia_error : float | None
        Uncertainty on pmdec_gaia (mas/yr).
    pmra_hg : float | None
        Long-term cross-epoch Hipparcos-Gaia proper motion in RA* (mas/yr).
    pmra_hg_error : float | None
        Uncertainty on pmra_hg (mas/yr).
    pmdec_hg : float | None
        Long-term cross-epoch Hipparcos-Gaia proper motion in Dec (mas/yr).
    pmdec_hg_error : float | None
        Uncertainty on pmdec_hg (mas/yr).
    epoch_ra_gaia : float | None
        Gaia central epoch for RA (Julian Year).
    epoch_dec_gaia : float | None
        Gaia central epoch for Dec (Julian Year).
    """

    hip_number: int
    gaia_source_id: int | None = None
    chisq: float | None = None
    significance_sigma: float | None = None
    pmra_hip: float | None = None
    pmra_hip_error: float | None = None
    pmdec_hip: float | None = None
    pmdec_hip_error: float | None = None
    pmra_gaia: float | None = None
    pmra_gaia_error: float | None = None
    pmdec_gaia: float | None = None
    pmdec_gaia_error: float | None = None
    pmra_hg: float | None = None
    pmra_hg_error: float | None = None
    pmdec_hg: float | None = None
    pmdec_hg_error: float | None = None
    epoch_ra_gaia: float | None = None
    epoch_dec_gaia: float | None = None


class HipparcosStatus(NamedTuple):
    """Summary of Hipparcos observation and Hipparcos-Gaia combinability.

    Parameters
    ----------
    in_hipparcos : bool
        True if the target has an identified Hipparcos catalog entry.
    hip_id : str | None
        Formatted Hipparcos ID (e.g. ``"HIP 27989"``).
    hip_number : int | None
        Numeric Hipparcos catalog number.
    in_hgca : bool
        True if cross-matched and calibrated in HGCA (Brandt 2021).
    combinable : bool
        True if Hipparcos and Gaia can be combined (i.e. cross-matched in HGCA).
    gaia_source_id : int | None
        Gaia EDR3/DR3 source identifier.
    hgca_chisq : float | None
        Chi-squared value of constant proper motion from HGCA.
    pm_anomaly_significance : float | None
        Significance of proper motion anomaly / acceleration in Gaussian sigmas.
    hgca_record : HGCARecord | None
        Full HGCA record if available.
    details : str
        Diagnostic or descriptive message.
    """

    in_hipparcos: bool
    hip_id: str | None
    hip_number: int | None = None
    in_hgca: bool = False
    combinable: bool = False
    gaia_source_id: int | None = None
    hgca_chisq: float | None = None
    pm_anomaly_significance: float | None = None
    hgca_record: HGCARecord | None = None
    details: str = ""


def extract_hip_id(identifier: str | int) -> int | None:
    """Parse numeric Hipparcos ID from an identifier string or integer.

    Parameters
    ----------
    identifier : str or int
        Identifier such as ``"HIP 27989"``, ``"HIP27989"``, ``"27989"``, or ``27989``.

    Returns
    -------
    int or None
        Numeric Hipparcos number, or None if it cannot be parsed.

    Examples
    --------
    >>> from exohelp.archive.hipparcos import extract_hip_id
    >>> extract_hip_id("HIP 27989")
    27989
    >>> extract_hip_id("HIP27989")
    27989
    >>> extract_hip_id(27989)
    27989
    >>> extract_hip_id("HD 81466") is None
    True
    """
    if isinstance(identifier, int):
        return identifier if identifier > 0 else None

    s = str(identifier).strip()
    match = _HIP_PREFIX_RE.match(s)
    if match:
        return int(match.group(1))
    if _DIGITS_RE.match(s):
        return int(s)
    return None


@cites("Perryman1997")
def in_hipparcos_by_name(star_name: str) -> tuple[bool, str | None]:
    """Query SIMBAD to check whether a star was observed by Hipparcos.

    Parameters
    ----------
    star_name : str
        Target star identifier (e.g. ``"Betelgeuse"``, ``"HD 81466"``).

    Returns
    -------
    tuple[bool, str | None]
        ``(True, "HIP <id>")`` if found in Hipparcos, else ``(False, None)``.

    Examples
    --------
    >>> # doctest: +SKIP
    >>> observed, hip_id = in_hipparcos_by_name("Betelgeuse")
    >>> observed
    True
    """
    # Fast path: check if the input string itself is already a HIP identifier
    parsed_id = extract_hip_id(star_name)
    if parsed_id is not None and (
        star_name.strip().upper().startswith("HIP") or isinstance(star_name, int)
    ):
        return True, f"HIP {parsed_id}"

    try:
        from astroquery.simbad import Simbad

        result = Simbad.query_objectids(star_name)
    except Exception as err:
        logger.warning(f"SIMBAD query failed for {star_name}: {err}")
        return False, None

    if result is None or len(result) == 0:
        return False, None

    id_col = next((c for c in ("id", "ID") if c in result.colnames), result.colnames[0])

    all_ids = [str(x).strip() for x in result[id_col]]
    hip_ids = [ident for ident in all_ids if ident.upper().startswith("HIP ")]

    if hip_ids:
        # Standardize format e.g. "HIP 27989"
        num = extract_hip_id(hip_ids[0])
        canonical = f"HIP {num}" if num is not None else hip_ids[0]
        return True, canonical

    return False, None


@cites("Brandt2021")
def query_hgca(
    hip_id: int | str,
    catalog: str = "J/ApJS/254/42/table1",
) -> HGCARecord | None:
    """Query VizieR for calibrated Hipparcos-Gaia astrometry from HGCA (Brandt 2021).

    Parameters
    ----------
    hip_id : int or str
        Hipparcos catalog number or identifier (e.g. ``27989`` or ``"HIP 27989"``).
    catalog : str, optional
        VizieR catalog table path for HGCA. Defaults to ``"J/ApJS/254/42/table1"``
        (Gaia EDR3/DR3 edition).

    Returns
    -------
    HGCARecord or None
        Parsed record containing calibrated proper motions and acceleration,
        or None if not found in HGCA.

    Examples
    --------
    >>> # doctest: +SKIP
    >>> record = query_hgca(27989)
    >>> record.gaia_source_id is not None
    True
    """
    hip_number = extract_hip_id(hip_id)
    if hip_number is None:
        return None

    try:
        from astroquery.vizier import Vizier

        v = Vizier(columns=["*"], catalog=catalog)
        tables = v.query_constraints(catalog=catalog, HIP=hip_number)
    except Exception as err:
        logger.warning(f"VizieR query for HGCA HIP {hip_number} failed: {err}")
        return None

    if not tables or len(tables[0]) == 0:
        return None

    row = tables[0][0]

    def _get_val(col: str) -> float | None:
        if col in row.colnames:
            val = row[col]
            try:
                if val is not None and str(val).strip() not in ("", "--", "nan", "None"):
                    return float(val)
            except (ValueError, TypeError):
                pass
        return None

    def _get_int(col: str) -> int | None:
        if col in row.colnames:
            val = row[col]
            try:
                if val is not None and str(val).strip() not in ("", "--", "nan", "None"):
                    return int(val)
            except (ValueError, TypeError):
                pass
        return None

    chisq = _get_val("chisq")
    sig = float(delta_chi2_to_sigma(chisq, mode="rv")) if chisq is not None else None

    return HGCARecord(
        hip_number=hip_number,
        gaia_source_id=_get_int("gaia_source_id") or _get_int("GaiaEDR3"),
        chisq=chisq,
        significance_sigma=sig,
        pmra_hip=_get_val("pmRA_hip"),
        pmra_hip_error=_get_val("e_pmRA_hip"),
        pmdec_hip=_get_val("pmDE_hip"),
        pmdec_hip_error=_get_val("e_pmDE_hip"),
        pmra_gaia=_get_val("pmRA_gaia"),
        pmra_gaia_error=_get_val("e_pmRA_gaia"),
        pmdec_gaia=_get_val("pmDE_gaia"),
        pmdec_gaia_error=_get_val("e_pmDE_gaia"),
        pmra_hg=_get_val("pmRA_hg"),
        pmra_hg_error=_get_val("e_pmRA_hg"),
        pmdec_hg=_get_val("pmDE_hg"),
        pmdec_hg_error=_get_val("e_pmDE_hg"),
        epoch_ra_gaia=_get_val("epRA_gaia"),
        epoch_dec_gaia=_get_val("epDE_gaia"),
    )


@cites("Perryman1997", "Brandt2021")
def check_hipparcos_gaia_combinable(
    star_name: str,
    check_hgca: bool = True,
    catalog: str = "J/ApJS/254/42/table1",
) -> HipparcosStatus:
    """Assess whether a star was observed by Hipparcos and can be combined with Gaia.

    Evaluates:
    1. Hipparcos observation presence (via SIMBAD aliases or direct HIP number).
    2. Cross-calibration in the Hipparcos-Gaia Catalog of Accelerations (HGCA; Brandt 2021).
       Stars in HGCA have calibrated epoch frames, error inflation, and acceleration statistics.

    Parameters
    ----------
    star_name : str
        Target star identifier (e.g. ``"HD 81466"``, ``"Betelgeuse"``, ``"HIP 27989"``).
    check_hgca : bool, optional
        Whether to query VizieR HGCA to verify cross-calibration, by default True.
    catalog : str, optional
        VizieR catalog table path for HGCA.

    Returns
    -------
    HipparcosStatus
        NamedTuple detailing Hipparcos identification, HGCA entry, and acceleration significance.

    Examples
    --------
    >>> # doctest: +SKIP
    >>> status = check_hipparcos_gaia_combinable("HD 81466")
    >>> status.in_hipparcos
    True
    >>> status.combinable
    True
    """
    observed, hip_id = in_hipparcos_by_name(star_name)

    if not observed or hip_id is None:
        return HipparcosStatus(
            in_hipparcos=False,
            hip_id=None,
            hip_number=None,
            in_hgca=False,
            combinable=False,
            details=f"'{star_name}' has no Hipparcos (HIP) identifier in SIMBAD.",
        )

    hip_num = extract_hip_id(hip_id)

    if not check_hgca:
        return HipparcosStatus(
            in_hipparcos=True,
            hip_id=hip_id,
            hip_number=hip_num,
            in_hgca=False,
            combinable=True,
            details=f"Found in Hipparcos as {hip_id} (HGCA cross-check skipped).",
        )

    hgca_record = query_hgca(hip_num, catalog=catalog) if hip_num is not None else None

    if hgca_record is not None:
        sig_str = (
            f", acceleration significance = {hgca_record.significance_sigma:.1f} sigma"
            if hgca_record.significance_sigma is not None
            else ""
        )
        return HipparcosStatus(
            in_hipparcos=True,
            hip_id=hip_id,
            hip_number=hip_num,
            in_hgca=True,
            combinable=True,
            gaia_source_id=hgca_record.gaia_source_id,
            hgca_chisq=hgca_record.chisq,
            pm_anomaly_significance=hgca_record.significance_sigma,
            hgca_record=hgca_record,
            details=f"Cross-calibrated with Gaia in HGCA (Brandt 2021{sig_str}).",
        )

    return HipparcosStatus(
        in_hipparcos=True,
        hip_id=hip_id,
        hip_number=hip_num,
        in_hgca=False,
        combinable=False,
        details=(
            f"Observed by Hipparcos ({hip_id}), but not cataloged in HGCA "
            "(may lack valid Gaia DR3 cross-match or 5-parameter astrometric solution)."
        ),
    )
