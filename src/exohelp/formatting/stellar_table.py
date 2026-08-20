import re
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd

from .numbers import format_number, format_value_with_uncertainty

DEFAULT_TABLEBIB: dict[str, str] = {
    "Gaia DR3": r"\textit{Gaia} DR3: \citet{GaiaCollaboration2023}",
    r"\textit{Gaia} DR3": r"\textit{Gaia} DR3: \citet{GaiaCollaboration2023}",
    "Lin21": r"Lin21: \citet{lindegren2021}",
    "Tycho-2": r"Tycho-2: \citet{Hoeg2000}",
    "TIC": r"TIC: \citet{Stassun2019}",
    "2MASS": r"2MASS: \citet{Skrutskie2006}",
    "ALLWISE": r"ALLWISE: \citet{Wright2010}",
}

AUTHOR_TO_SURVEY: dict[str, str] = {
    "Skr06": "2MASS",
    "Cut13": "ALLWISE",
    "Høg00": "Tycho-2",
    "Sta19": "TIC",
    "Gai23": "Gaia DR3",
}

SURVEY_TO_AUTHOR: dict[str, str] = {
    "2MASS": "Skr06",
    "ALLWISE": "Cut13",
    "WISE": "Cut13",
    "Tycho-2": "Høg00",
    "Tycho": "Høg00",
    "TIC": "Sta19",
}


def _normalize_bib_key(key: str) -> str:
    """Normalize bibliography key by removing LaTeX formatting and lowercasing."""
    cleaned = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", str(key))
    cleaned = cleaned.replace(r"\_", "_").replace(r"\ ", " ").replace("\\", "")
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def _resolve_bib_entry(
    src_token: str,
    bib_map: dict[str, str],
    normalized_bib_map: dict[str, str],
) -> str | None:
    """Resolve a source token to a bibliography citation entry without false substring matches."""
    if not src_token or src_token.lower() in ("this work", "nan", "none", "--"):
        return None
    if r"\ref" in src_token or r"\cite" in src_token:
        return None

    # 1. Exact match in raw bib_map
    if src_token in bib_map:
        return bib_map[src_token]

    # 2. Check mapped author/survey aliases
    mapped_survey = AUTHOR_TO_SURVEY.get(src_token)
    if mapped_survey and mapped_survey in bib_map:
        return bib_map[mapped_survey]

    mapped_author = SURVEY_TO_AUTHOR.get(src_token)
    if mapped_author and mapped_author in bib_map:
        return bib_map[mapped_author]

    # 3. Exact match against normalized keys
    norm_src = _normalize_bib_key(src_token)
    if norm_src in normalized_bib_map:
        return normalized_bib_map[norm_src]

    if mapped_survey:
        norm_survey = _normalize_bib_key(mapped_survey)
        if norm_survey in normalized_bib_map:
            return normalized_bib_map[norm_survey]

    return None


def format_unit_aa(unit_raw: str) -> str:
    """Standardize Astropy/raw unit string into clean A&A LaTeX notation."""
    if not unit_raw or unit_raw.strip() == "":
        return ""

    u_str = unit_raw.strip().replace(" ", "")
    mapping = {
        "mas": r"\mathrm{mas}",
        "pc": r"\mathrm{pc}",
        "km/s": r"\mathrm{km\,s^{-1}}",
        "m/s": r"\mathrm{m\,s^{-1}}",
        "mas/yr": r"\mathrm{mas\,a^{-1}}",
        "mas/a": r"\mathrm{mas\,a^{-1}}",
        "mag": r"\mathrm{mag}",
        "K": r"\mathrm{K}",
        "deg": r"\mathrm{deg}",
        "dex": r"\mathrm{dex}",
        "cgs": r"\mathrm{cgs}",
        "g/cm3": r"\mathrm{g\,cm^{-3}}",
        "g/cm^3": r"\mathrm{g\,cm^{-3}}",
        "Rsun": r"\mathrm{R}_\odot",
        "R_sun": r"\mathrm{R}_\odot",
        "Msun": r"\mathrm{M}_\odot",
        "M_sun": r"\mathrm{M}_\odot",
        "Lsun": r"\mathrm{L}_\odot",
        "L_sun": r"\mathrm{L}_\odot",
        "d": r"\mathrm{d}",
        "day": r"\mathrm{d}",
        "days": r"\mathrm{d}",
        "Gyr": r"\mathrm{Gyr}",
        "Myr": r"\mathrm{Myr}",
        "yr": r"\mathrm{a}",
    }

    if u_str in mapping:
        return mapping[u_str]
    try:
        formatted = u.Unit(unit_raw).to_string("latex_inline").replace("$", "")
        return formatted
    except Exception:
        return unit_raw


def generate_stellar_table_latex(
    star_name: str,
    id_df: pd.DataFrame | None = None,
    param_df: pd.DataFrame | None = None,
    significant_digits: int = 2,
    adaptive_sigfigs: bool = False,
    units_in_parameter: bool = True,
    unit_brackets: str = "round",
    section_hlines: bool = True,
    noalign_smallskip: bool = False,
    reference_style: str = "survey",
    custom_references: dict[str, str] | None = None,
    include_tablebib: bool = True,
    tablebib_mapping: dict[str, str] | None = None,
    caption: str | None = None,
    label: str = "tab:stellar_properties",
    tablefoot_notes: dict[str, str] | list[str] | str | None = None,
) -> str:
    """Generate a publication-ready single continuous 3-column A&A stellar table."""
    latex = []

    # 1. Float & Caption
    latex.append(r"\begin{table}[!ht]")
    tab_caption = caption if caption is not None else f"Stellar parameters of {star_name}."
    latex.append(f"    \\caption{{{tab_caption}}}")
    latex.append(f"    \\label{{{label}}}")
    latex.append(r"    \centering")
    latex.append(r"    \begin{tabular}{l c r}")
    latex.append(r"    \hline\hline")
    if noalign_smallskip:
        latex.append(r"    \noalign{\smallskip}")
    latex.append(r"    Parameter & Value & Reference \\")
    if noalign_smallskip:
        latex.append(r"    \noalign{\smallskip}")
    if not section_hlines:
        latex.append(r"    \hline")

    # 2. Section Grouping
    has_custom_sections = (id_df is not None and "section" in id_df.columns) or (
        param_df is not None and "section" in param_df.columns
    )

    sections = []
    if has_custom_sections:
        combined_dfs = []
        if id_df is not None and not id_df.empty:
            combined_dfs.append(id_df)
        if param_df is not None and not param_df.empty:
            combined_dfs.append(param_df)
        full_df = pd.concat(combined_dfs, ignore_index=True)
        if "section" in full_df.columns:
            unique_sections = list(dict.fromkeys(full_df["section"].dropna()))
            for sec in unique_sections:
                sec_sub = full_df[full_df["section"] == sec]
                sections.append((sec, sec_sub))
        else:
            sections.append(("Basic identifiers and data", full_df))
    else:
        identifier_keys = [
            "star_name",
            "toi",
            "tic",
            "tyc",
            "gaia_dr3",
            "twomass",
            "allwise",
            "spectral_type",
        ]

        astrometric_keys = [
            "ra",
            "dec",
            "pm_ra",
            "pm_dec",
            "radial_velocity",
            "parallax",
            "distance",
            "u_lsr",
            "v_lsr",
            "w_lsr",
            "galactic_population",
        ]

        # Mission/Survey Order: Gaia -> Tycho -> TESS -> 2MASS -> WISE
        photometric_keys = [
            "g_mag",
            "bp_mag",
            "rp_mag",
            "bt_mag",
            "vt_mag",
            "tess_mag",
            "j_mag",
            "h_mag",
            "ks_mag",
            "w1_mag",
            "w2_mag",
            "w3_mag",
            "w4_mag",
        ]

        def _sort_by_keys(df_subset, key_order):
            if df_subset is None or df_subset.empty:
                return df_subset
            if "name" not in df_subset.columns:
                return df_subset
            order_dict = {k: i for i, k in enumerate(key_order)}
            ranks = df_subset["name"].map(lambda x: order_dict.get(x, 999))
            return df_subset.iloc[np.argsort(ranks, kind="stable")]

        id_subset = _sort_by_keys(id_df, identifier_keys) if id_df is not None else None
        if param_df is not None and not param_df.empty:
            if "name" in param_df.columns:
                astrometric_subset = _sort_by_keys(
                    param_df[param_df["name"].isin(astrometric_keys)], astrometric_keys
                )
                photo_raw = param_df[
                    param_df["name"].isin(photometric_keys)
                    | (
                        ~param_df["name"].isin(astrometric_keys)
                        & param_df.get("unit", pd.Series(dtype=object))
                        .astype(str)
                        .str.contains("mag")
                    )
                ]
                photo_subset = _sort_by_keys(photo_raw, photometric_keys)
            else:
                astrometric_subset = None
                photo_subset = None
        else:
            astrometric_subset = None
            photo_subset = None

        sections = [
            ("Basic identifiers and data", id_subset),
            ("Astrometric properties", astrometric_subset),
            ("Photometric properties", photo_subset),
        ]

        # Remaining parameters (Fundamental parameters / Activity)
        if param_df is not None and not param_df.empty:
            if "name" in param_df.columns:
                used_keys = (
                    (
                        set(id_df["name"])
                        if id_df is not None and not id_df.empty and "name" in id_df.columns
                        else set()
                    )
                    .union(astrometric_keys)
                    .union(photometric_keys)
                )
                remaining = param_df[~param_df["name"].isin(used_keys)]
                if not remaining.empty:
                    sections.append(("Fundamental parameters", remaining))
            elif astrometric_subset is None and photo_subset is None:
                sections.append(("Fundamental parameters", param_df))

    open_b = "[" if unit_brackets in ("square", "[]") else "("
    close_b = "]" if unit_brackets in ("square", "[]") else ")"

    for section_title, subset in sections:
        if subset is None or subset.empty:
            continue

        if section_hlines:
            latex.append(r"    \hline")
            if noalign_smallskip:
                latex.append(r"    \noalign{\smallskip}")
            latex.append(f"    \\multicolumn{{3}}{{c}}{{\\textit{{{section_title}}}}} \\\\")
            if noalign_smallskip:
                latex.append(r"    \noalign{\smallskip}")
            latex.append(r"    \hline")
        else:
            if noalign_smallskip:
                latex.append(r"    \noalign{\smallskip}")
            latex.append(f"    \\multicolumn{{3}}{{l}}{{\\textit{{{section_title}}}}} \\\\")
            if noalign_smallskip:
                latex.append(r"    \noalign{\smallskip}")

        for _, row in subset.iterrows():
            p_sym = str(row.get("symbol", row.get("name", "")))
            p_tex = p_sym if "$" in p_sym else p_sym.replace("_", r"\_")

            v_raw = row.get("value", np.nan)
            e_raw = row.get("uncertainty", np.nan)

            # Support asymmetric uncertainties (tuple or list of 2 elements)
            is_asymmetric = isinstance(e_raw, (tuple, list, np.ndarray)) and len(e_raw) == 2
            if is_asymmetric:
                try:
                    e_plus = float(e_raw[0])
                    e_minus = float(e_raw[1])
                    has_asym_err = True
                except (ValueError, TypeError):
                    has_asym_err = False
            else:
                has_asym_err = False

            val = (
                np.ravel(v_raw)[0]
                if hasattr(v_raw, "__iter__") and not isinstance(v_raw, str) and np.size(v_raw) > 0
                else v_raw
            )
            err = (
                np.ravel(e_raw)[0]
                if hasattr(e_raw, "__iter__") and not is_asymmetric and np.size(e_raw) > 0
                else e_raw
            )

            # Check if numerical
            try:
                v_num = float(val)
                is_num = not np.isnan(v_num)
                try:
                    e_num = float(err) if not has_asym_err else np.nan
                    has_err = not np.isnan(e_num) and e_num != 0
                except (ValueError, TypeError):
                    has_err = False
            except (ValueError, TypeError):
                is_num = False
                has_err = False

            if section_title in ("Basic identifiers and data", "Identifiers"):
                val_str = str(val).replace("_", r"\_")
            elif is_num:
                if has_asym_err:
                    val_str = format_value_with_uncertainty(
                        v_num,
                        lower_uncertainty=abs(e_minus),
                        upper_uncertainty=abs(e_plus),
                        uncertainty_style="asymmetric",
                        adaptive_sigfigs=adaptive_sigfigs,
                        significant_digits=significant_digits,
                    )
                elif has_err:
                    val_str = format_value_with_uncertainty(
                        v_num,
                        e_num,
                        uncertainty_style="pm",
                        adaptive_sigfigs=adaptive_sigfigs,
                        significant_digits=significant_digits,
                    )
                else:
                    val_str = f"${format_number(v_num, significant_digits=significant_digits)}$"
            else:
                val_str = str(val).replace("_", r"\_")

            unit_raw = str(row.get("unit", ""))
            unit_fmt = format_unit_aa(unit_raw)

            source_raw = str(row.get("source", ""))
            if reference_style == "survey":
                source_mapped = AUTHOR_TO_SURVEY.get(source_raw, source_raw)
                # Omit self-referential catalog references in the identifiers section
                if section_title in (
                    "Basic identifiers and data",
                    "Identifiers",
                ) and source_mapped in (
                    "TIC",
                    "Tycho-2",
                    "Tycho",
                    "Gaia DR3",
                    "Gaia EDR3",
                    "2MASS",
                    "ALLWISE",
                    "WISE",
                    p_sym,
                ):
                    source_mapped = ""
            elif reference_style == "author":
                source_mapped = SURVEY_TO_AUTHOR.get(source_raw, source_raw)
            else:
                source_mapped = source_raw

            if custom_references and source_mapped in custom_references:
                source_mapped = custom_references[source_mapped]

            source = (
                source_mapped
                if r"\ref" in source_mapped or r"\cite" in source_mapped
                else source_mapped.replace("_", r"\_")
            )

            if units_in_parameter and unit_fmt:
                if p_tex.startswith("$") and p_tex.endswith("$"):
                    inner = p_tex[1:-1].strip()
                    p_col = rf"${inner}\ {open_b}{unit_fmt}{close_b}$"
                else:
                    p_col = rf"{p_tex}~${open_b}{unit_fmt}{close_b}$"
            else:
                p_col = p_tex

            latex.append(f"    {p_col} & {val_str} & {source} \\\\")

    if noalign_smallskip:
        latex.append(r"    \noalign{\smallskip}")
    latex.append(r"    \hline")
    latex.append(r"    \end{tabular}")

    # 3. Tablefoot
    if tablefoot_notes:
        latex.append(r"    \tablefoot{")
        if isinstance(tablefoot_notes, dict):
            for k, v in tablefoot_notes.items():
                latex.append(f"        \\tablefoottext{{{k}}}{{{v}}};")
        elif isinstance(tablefoot_notes, list):
            for note in tablefoot_notes:
                latex.append(f"        {note}")
        elif isinstance(tablefoot_notes, str):
            latex.append(f"        {tablefoot_notes}")
        latex.append(r"    }")

    # 4. Tablebib
    if include_tablebib:
        bib_map = dict(DEFAULT_TABLEBIB)
        if tablebib_mapping:
            bib_map.update(tablebib_mapping)

        normalized_bib_map = {_normalize_bib_key(k): v for k, v in bib_map.items()}

        bib_entries = []
        for _, subset in sections:
            if subset is None or subset.empty:
                continue
            for _, r in subset.iterrows():
                src_val = str(r.get("source", "")).strip()
                if not src_val:
                    continue
                tokens = [p.strip() for p in src_val.replace(";", ",").split(",") if p.strip()]
                for token in tokens:
                    entry = _resolve_bib_entry(token, bib_map, normalized_bib_map)
                    if entry and entry not in bib_entries:
                        bib_entries.append(entry)

        if bib_entries:
            latex.append(r"    \tablebib{")
            for i, entry in enumerate(bib_entries):
                sep = ";" if i < len(bib_entries) - 1 else "."
                latex.append(f"        {entry}{sep}")
            latex.append(r"    }")

    latex.append(r"\end{table}")
    return "\n".join(latex)


def save_stellar_table_latex(
    filepath: str | Path,
    star_name: str,
    id_df: pd.DataFrame,
    param_df: pd.DataFrame,
    **kwargs,
) -> Path:
    """Generate LaTeX table string and write it to a file.

    Parameters
    ----------
    filepath : str or Path
        Target file path to save the LaTeX table.
    star_name : str
        Name of the star.
    id_df : pd.DataFrame
        Identifier DataFrame.
    param_df : pd.DataFrame
        Parameters DataFrame.
    **kwargs
        Additional arguments passed to `generate_stellar_table_latex`.

    Returns
    -------
    Path
        Path object pointing to written file.
    """
    latex_str = generate_stellar_table_latex(star_name, id_df, param_df, **kwargs)
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex_str, encoding="utf-8")
    return out_path
