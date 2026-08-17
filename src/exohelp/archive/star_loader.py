import logging
from typing import ClassVar

import astropy.table
import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from astroquery.mast import Catalogs
from astroquery.vizier import Vizier

from exohelp.formatting import format_number, format_value_with_uncertainty
from exohelp.star.spectroscopy import sample_uvw_lsr

logger = logging.getLogger("stellar_loader")


class StarLoader:
    """

    Examples
    --------
    >>> from exohelp.archive.star_loader import StarLoader
    >>> loader = StarLoader()
    >>> tic_results = loader.get_tic_results("334632624")
    >>> full_table = loader.get_complete_system_data(tic_results[0])
    >>> full_table = loader.add_uvw(full_table)
    >>> id_table = loader.extract_identifier_data(tic_results[0])
    >>> print(loader.generate_stellar_table_latex("HD 81466", id_table, full_table, units_in_parameter=True, significant_digits=2))
    """

    CONFIG: ClassVar[dict] = {
        "gaia": {
            "catalog": "I/350/gaiaedr3",
            "source": "1; EGDR3",
            "id_col": "EDR3Name",
            "id_prefix": "Gaia EDR3 ",
            "columns": [
                "RA_ICRS",
                "DE_ICRS",
                "pmRA",
                "pmDE",
                "RVDR2",
                "Plx",
                "Gmag",
                "BPmag",
                "RPmag",
            ],
            "rename": {
                "RVDR2": "RV",
                "Plx": "Parallax",
                "RA_ICRS": "RA",
                "DE_ICRS": "Dec",
                "pmRA": r"$\mu_{\alpha^*}$",
                "pmDE": r"$\mu_\delta$",
                "BPmag": r"$G_\mathbb{BP}$",
                "RPmag": r"$G_\mathbb{RP}$",
                "Gmag": "$G$",
            },
        },
        "twomass": {
            "catalog": "II/246/out",
            "source": "3; 2MASS",
            "id_col": "2MASS",
            "id_prefix": "",
            "columns": ["Jmag", "Hmag", "Kmag"],
            "rename": {"Jmag": "J", "Hmag": "H", "Kmag": "K"},
        },
        "wise": {
            "catalog": "II/328/allwise",
            "source": "5; AllWISE",
            "id_col": "AllWISE",
            "id_prefix": "",
            "columns": ["W1mag", "W2mag"],
            "rename": {"W1mag": "W_1", "W2mag": "W_2"},
        },
        "tycho": {
            "catalog": "I/259/tyc2",
            "source": "2; Tycho-2",
            "id_col": "TYC",  # We will construct this manually in the Tycho logic
            "columns": ["VTmag", "BTmag"],
            "rename": {"VTmag": "V", "BTmag": "B"},
        },
    }

    def __init__(self, vizier_server: str = "vizier.cds.unistra.fr"):
        self.viz = Vizier(columns=["**"], row_limit=1, timeout=200, vizier_server=vizier_server)

    @property
    def catalogs(self):
        return [cfg["catalog"] for cfg in self.CONFIG.values()]

    def _get_catalog_table(self, query_results, config_key):
        """Look up a catalog table by name, tolerating missing catalogs."""
        catalog_name = self.CONFIG[config_key]["catalog"]
        for key in list(query_results.keys()):
            if catalog_name in key or key in catalog_name:
                return query_results[key]
        return None

    def _extract_basic_columns(self, table, config_key, target_id=None):
        """Extracts values, errors, and units, with optional ID matching."""
        if table is None or len(table) == 0:
            return []

        cfg = self.CONFIG[config_key]

        # --- ID Matching Logic ---
        matched_row = table[0]  # Default to first row

        if target_id is not None:
            if config_key == "tycho":
                # Special Tycho name construction
                ids = np.array([f"{r['TYC1']}-{r['TYC2']}-{r['TYC3']}" for r in table])
                mask = ids == self._normalize_tyc(target_id)
            else:
                # Standard matching using id_col from CONFIG
                id_col = cfg.get("id_col")
                prefix = cfg.get("id_prefix", "")
                mask = table[id_col] == f"{prefix}{target_id}"

            if np.any(mask):
                matched_row = table[mask][0]
            else:
                logger.warning(
                    f"ID mismatch in {config_key}: Expected {target_id}. Using first result."
                )

        # --- Extraction Logic ---
        extracted = []
        for col in cfg["columns"]:
            if col not in table.colnames:
                continue

            val = matched_row[col]
            unit = str(table[col].unit) if table[col].unit else ""

            err_col = f"e_{col}"
            err = matched_row[err_col] if err_col in table.colnames else np.nan

            name = cfg.get("rename", {}).get(col, col)
            extracted.append(
                {
                    "Parameter": name,
                    "Value": val,
                    "Uncertainty": err,
                    "Units": unit,
                    "Source": cfg["source"],
                }
            )

        if config_key == "gaia":
            corrected_plx_entry = self._correct_parallax(matched_row)
            if corrected_plx_entry:
                extracted.append(corrected_plx_entry)

                # 2. Distance Calculation
                # Use the corrected parallax for the distance calculation
                dist_entry = self._calculate_distance_from_parallax(
                    corrected_plx_entry["Value"], corrected_plx_entry["Uncertainty"]
                )
            else:
                # Fallback to raw parallax if correction fails
                dist_entry = self._calculate_distance_from_parallax(
                    matched_row["Plx"], matched_row["e_Plx"]
                )

            if dist_entry:
                extracted.append(dist_entry)

        return extracted

    def query_catalog_region(self, tic_result):
        """Query Vizier catalogs for the 2 arcsec region around the target.

        Use TIC coordinates rather than a Gaia ID to avoid DR3/EDR3 mismatch issues.
        Returns the raw ``TableList`` from Vizier so callers can inspect it directly.
        """
        if isinstance(tic_result, astropy.table.Table):
            if len(tic_result) == 1:
                tic_result = tic_result[0]
            else:
                raise ValueError("Expected a single row in tic_result table.")
        coord = SkyCoord(ra=tic_result["ra"], dec=tic_result["dec"], unit="deg")
        return self.viz.query_region(coord, radius=u.Quantity(2, "arcsec"), catalog=self.catalogs)

    def get_complete_system_data(self, tic_result) -> pd.DataFrame:
        """Main orchestrator mimicking the original function logic."""
        if isinstance(tic_result, astropy.table.Table):
            if len(tic_result) == 1:
                tic_result = tic_result[0]
            else:
                raise ValueError("Expected a single row in tic_result table.")

        gaia_id = tic_result.get("GAIA")

        query_results = self.query_catalog_region(tic_result)

        all_entries = []

        # Extract data — look up by catalog name so missing catalogs don't shift indices
        all_entries.extend(
            self._extract_basic_columns(
                self._get_catalog_table(query_results, "gaia"), "gaia", target_id=gaia_id
            )
        )
        all_entries.extend(
            self._extract_basic_columns(
                self._get_catalog_table(query_results, "twomass"),
                "twomass",
                target_id=tic_result.get("twomass_id"),
            )
        )
        all_entries.extend(
            self._extract_basic_columns(
                self._get_catalog_table(query_results, "wise"),
                "wise",
                target_id=tic_result.get("allwise_id"),
            )
        )
        all_entries.extend(
            self._extract_basic_columns(
                self._get_catalog_table(query_results, "tycho"),
                "tycho",
                target_id=tic_result.get("tyc_id"),
            )
        )

        # TESS Mag (from local tic_result)
        all_entries.append(
            {
                "Parameter": "TESS",
                "Value": tic_result.get("Tmag"),
                "Uncertainty": tic_result.get("e_Tmag"),
                "Units": "mag",
                "Source": "4; TESS",
            }
        )
        df = pd.DataFrame(all_entries)

        return df

    def extract_identifier_data(self, tic_result: dict):
        tic_id = tic_result["ID"]

        # NASA TOI Check
        toipfx = "N/A"
        try:
            toi_data = NasaExoplanetArchive.query_criteria(table="toi", where=f"tid = {tic_id}")
            if len(toi_data) > 0:
                toipfx = toi_data["toipfx"][0]
        except Exception:
            logger.warning(f"TOI query failed for TIC {tic_id}")

        return pd.DataFrame(
            {
                "Parameter": ["TOI", "TIC", "TYC", "Gaia DR3", "2MASS", "ALLWISE"],
                "Value": [
                    toipfx,
                    tic_result.get("ID"),
                    self._normalize_tyc(tic_result.get("TYC")),
                    tic_result.get("GAIA"),
                    tic_result.get("TWOMASS"),
                    tic_result.get("ALLWISE"),
                ],
                "Source": ["NASA"] + ["TIC"] * 5,
            }
        )

    @staticmethod
    def get_tic_results(tic_id):
        query_id = f"TIC {tic_id}" if not str(tic_id).startswith("TIC") else tic_id
        return Catalogs.query_object(objectname=query_id, radius=0.001, catalog="TIC")

    def _get_val(self, df, param):
        """Helper to get (value, uncertainty) for a parameter from the results DF."""
        mask = df["Parameter"] == param
        if not mask.any():
            return None, None
        row = df[mask].iloc[0]
        return row["Value"], row["Uncertainty"]

    def _calculate_distance_from_parallax(self, plx, e_plx):
        if not (plx and plx > 0):
            return None
        dist = 1000.0 / plx
        e_dist = (e_plx / plx) * dist
        return {
            "Parameter": "Distance",
            "Value": dist,
            "Uncertainty": e_dist,
            "Units": "pc",
            "Source": "0, inverse parallax",
        }

    def _correct_parallax(self, gaia_row):
        """Applies parallax zero-point correction based on Gaia EDR3."""
        try:
            from zero_point import zpt

            zpt.load_tables()
        except ImportError:
            logger.warning("zpt module not found. Parallax correction will be skipped.")
            return None

        try:

            def _unmasked(val):
                return np.float64(np.ma.filled(val, np.nan))

            zero_point = zpt.get_zpt(
                _unmasked(gaia_row["Gmag"]),
                _unmasked(gaia_row["nueff"]),
                _unmasked(gaia_row["pscol"]),
                _unmasked(gaia_row["GLAT"]),
                _unmasked(gaia_row["Solved"]),
            )

            parallax_corrected = gaia_row["Plx"] - zero_point
        except Exception as e:
            logger.warning(f"Parallax correction failed: {e}")
            return None

        return {
            "Parameter": "Parallax",
            "Value": parallax_corrected,
            "Uncertainty": gaia_row["e_Plx"],
            "Units": "mas",
            "Source": "1a; EGDR3",
        }

    def add_uvw(self, df, n_samples=100_000, seed=None):
        """Returns a new DataFrame with UVW space velocities appended."""

        def get_quantity(param):
            mask = df["Parameter"] == param
            if not mask.any():
                raise ValueError(f"{param} not found in DataFrame.")
            row = df[mask].iloc[0]
            return u.Quantity(row["Value"], row["Units"]), u.Quantity(
                row["Uncertainty"], row["Units"]
            )

        ra, e_ra = get_quantity("RA")
        dec, e_dec = get_quantity("Dec")
        pm_ra, e_pm_ra = get_quantity(r"$\mu_{\alpha^*}$")
        pm_dec, e_pm_dec = get_quantity(r"$\mu_\delta$")
        rv, e_rv = get_quantity("RV")
        distance, e_distance = get_quantity("Distance")
        result = sample_uvw_lsr(
            ra=ra,
            ra_err=e_ra,
            dec=dec,
            dec_err=e_dec,
            distance=distance,
            distance_err=e_distance,
            pm_ra_cosdec=pm_ra,
            pm_ra_cosdec_err=e_pm_ra,
            pm_dec=pm_dec,
            pm_dec_err=e_pm_dec,
            radial_velocity=rv,
            radial_velocity_err=e_rv,
            n_samples=n_samples,
            seed=seed,
        )

        new_rows = []
        for comp in ["U_lsr", "V_lsr", "W_lsr"]:
            mean = np.mean(result[comp])  # type: ignore
            std = np.std(result[comp])  # type: ignore
            new_rows.append(
                {
                    "Parameter": comp.replace("_lsr", r"_\mathrm{LSR}"),
                    "Value": mean.value,
                    "Uncertainty": std.value,
                    "Units": mean.unit.to_string("latex_inline"),
                    "Source": "0;",
                }
            )

        return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    @staticmethod
    def generate_stellar_table_latex(
        star_name,
        id_df,
        param_df,
        significant_digits=2,
        adaptive_sigfigs=False,
        units_in_parameter=False,
    ):
        latex = []

        # 1. Header and Caption
        latex.append(r"\begin{table}[!ht]")
        latex.append(rf"    \caption{{Stellar properties of {star_name}}}")
        latex.append(r"    \label{tab:stellar_properties}")

        # 2. Identifiers Section
        latex.append(r"    \textbf{Identifiers}\\")
        latex.append(r"    \begin{tabular}{ll}")
        for _, row in id_df.iterrows():
            # Safely extract scalar from potential 0-d or 1-d numpy objects
            v_raw = row["Value"]
            val = np.ravel(v_raw)[0] if hasattr(v_raw, "__iter__") and np.size(v_raw) > 0 else v_raw
            val_str = str(val).replace("_", r"\_")
            latex.append(f"    {row['Parameter']} & {val_str} \\\\")
        latex.append(r"    \end{tabular}\\")
        latex.append("\n")

        # 3. Parameters
        astrometric_params = [
            "RA",
            "Dec",
            r"$\mu_{\alpha^*}$",
            r"$\mu_\delta$",
            "RV",
            "Parallax",
            "Distance",
            r"U_\mathrm{LSR}",
            r"V_\mathrm{LSR}",
            r"W_\mathrm{LSR}",
        ]
        sections = [
            ("Astrometric Properties", astrometric_params),
            (
                "Photometric properties",
                # Must have mag as unit and not be in the astrometric list
                [
                    p
                    for p in param_df["Parameter"]
                    if p not in astrometric_params
                    and "mag" in param_df[param_df["Parameter"] == p]["Units"].to_numpy()[0]
                ],
            ),
        ]

        col_spec = r"lcl" if units_in_parameter else r"lcll"
        header = (
            r"Parameter & Value & Source \\"
            if units_in_parameter
            else r"Parameter & Value & Unit & Source \\"
        )

        for section_name, filter_list in sections:
            subset = param_df[param_df["Parameter"].isin(filter_list)]
            latex.append(f"    \\textbf{{{section_name}}}\\\\")
            latex.append(f"    \\begin{{tabular}}{{{col_spec}}}")
            latex.append(f"    {header}")

            for _, row in subset.iterrows():
                p_orig = row["Parameter"]
                p_tex = str(p_orig).replace("_", r"\_")

                # Extract scalars safely using np.ravel
                v_raw = row["Value"]
                e_raw = row["Uncertainty"]

                val = (
                    np.ravel(v_raw)[0]
                    if hasattr(v_raw, "__iter__") and np.size(v_raw) > 0
                    else v_raw
                )
                err = (
                    np.ravel(e_raw)[0]
                    if hasattr(e_raw, "__iter__") and np.size(e_raw) > 0
                    else e_raw
                )

                # Determine if we have numbers
                try:
                    # Convert to standard float to avoid MaskedArray issues
                    v_num = float(val)
                    is_num = not np.isnan(v_num)

                    try:
                        e_num = float(err)
                        has_err = not np.isnan(e_num) and e_num != 0
                    except (ValueError, TypeError):
                        has_err = False
                except (ValueError, TypeError):
                    is_num = False
                    has_err = False

                if is_num:
                    if has_err:
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

                unit_raw = str(row.get("Units", ""))
                try:
                    unit = u.Unit(unit_raw).to_string("latex_inline") if unit_raw else ""
                except Exception:
                    unit = unit_raw
                source = str(row.get("Source", "")).replace("_", r"\_")

                if units_in_parameter:
                    p_col = f"{p_tex}~$[{unit}]$" if unit else p_tex
                    latex.append(f"    {p_col} & {val_str} & {source} \\\\")
                else:
                    latex.append(f"    {p_tex} & {val_str} & {unit} & {source} \\\\")

            latex.append(r"    \end{tabular}\\")
            latex.append("\n")

        latex.append(r"\end{table}")
        return "\n".join(latex)

    @staticmethod
    def _normalize_tyc(tid):
        if not tid or tid.strip() == "":
            return None
        parts = tid.strip().split("-")
        return f"{int(parts[0])}-{int(parts[1])}-{int(parts[2])}"
