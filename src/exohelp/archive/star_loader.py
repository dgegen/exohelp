import json
import logging
from pathlib import Path
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
    """Loader and LaTeX table generator for stellar and system properties.

    Follows Astronomy & Astrophysics (A&A) editorial standards.

    Examples
    --------
    >>> from exohelp.archive.star_loader import StarLoader
    >>> loader = StarLoader()
    >>> tic_results = loader.get_tic_results("334632624")
    >>> full_table = loader.get_complete_system_data(tic_results[0])
    >>> full_table = loader.add_uvw(full_table)
    >>> id_table = loader.extract_identifier_data(
    ...     tic_results[0], star_name="HD 81466", spectral_type="G8 V"
    ... )
    >>> latex = loader.generate_stellar_table_latex(
    ...     "HD 81466", id_table, full_table, significant_digits=2, unit_brackets="square",
    ... )
    >>> print(latex)
    """

    CONFIG: ClassVar[dict] = {
        "gaia": {
            "catalog": "I/350/gaiaedr3",
            "source": "Gaia DR3",
            "id_col": "EDR3Name",
            "id_prefix": "Gaia EDR3 ",
            "columns": {
                "RA_ICRS": ("ra", r"$\alpha\ (\mathrm{J2016.0})$", "deg"),
                "DE_ICRS": ("dec", r"$\delta\ (\mathrm{J2016.0})$", "deg"),
                "pmRA": ("pm_ra", r"$\mu_\alpha \cos \delta$", "mas/yr"),
                "pmDE": ("pm_dec", r"$\mu_\delta$", "mas/yr"),
                "RVDR2": ("radial_velocity", r"$\gamma$", "km/s"),
                "Plx": ("parallax", r"$\varpi$", "mas"),
                "Gmag": ("g_mag", r"$G$", "mag"),
                "BPmag": ("bp_mag", r"$G_\mathrm{BP}$", "mag"),
                "RPmag": ("rp_mag", r"$G_\mathrm{RP}$", "mag"),
            },
        },
        "twomass": {
            "catalog": "II/246/out",
            "source": "2MASS",
            "id_col": "2MASS",
            "id_prefix": "",
            "columns": {
                "Jmag": ("j_mag", r"$J$", "mag"),
                "Hmag": ("h_mag", r"$H$", "mag"),
                "Kmag": ("ks_mag", r"$K_\mathrm{s}$", "mag"),
            },
        },
        "wise": {
            "catalog": "II/328/allwise",
            "source": "ALLWISE",
            "id_col": "AllWISE",
            "id_prefix": "",
            "columns": {
                "W1mag": ("w1_mag", r"$W_1$", "mag"),
                "W2mag": ("w2_mag", r"$W_2$", "mag"),
            },
        },
        "tycho": {
            "catalog": "I/259/tyc2",
            "source": "Tycho-2",
            "id_col": "TYC",
            "columns": {
                "VTmag": ("vt_mag", r"$V_\mathrm{T}$", "mag"),
                "BTmag": ("bt_mag", r"$B_\mathrm{T}$", "mag"),
            },
        },
    }

    def __init__(
        self,
        vizier_server: str = "vizier.cds.unistra.fr",
        cache_dir: str | Path | None = None,
    ):
        self.viz = Vizier(columns=["**"], row_limit=1, timeout=200, vizier_server=vizier_server)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def catalogs(self):
        return [cfg["catalog"] for cfg in self.CONFIG.values()]

    def _get_catalog_table(self, query_results, config_key):
        """Look up a catalog table by name, tolerating missing catalogs."""
        if query_results is None:
            return None
        catalog_name = self.CONFIG[config_key]["catalog"]
        keys = list(query_results.keys()) if hasattr(query_results, "keys") else []
        for key in keys:
            if isinstance(key, str) and (catalog_name in key or key in catalog_name):
                return query_results[key]
        return None

    def _extract_basic_columns(self, table, config_key, target_id=None):
        """Extracts values, errors, units, machine name, and symbol."""
        if table is None or len(table) == 0:
            return []

        cfg = self.CONFIG[config_key]

        # --- ID Matching Logic ---
        matched_row = table[0]  # Default to first row

        if target_id is not None:
            if config_key == "tycho":
                ids = np.array([f"{r['TYC1']}-{r['TYC2']}-{r['TYC3']}" for r in table])
                mask = ids == self._normalize_tyc(target_id)
            else:
                id_col = cfg.get("id_col")
                prefix = cfg.get("id_prefix", "")
                mask = table[id_col] == f"{prefix}{target_id}"

            if np.any(mask):
                matched_row = table[mask][0]
            else:
                logger.warning(
                    f"ID mismatch in {config_key}: Expected {target_id}. Using first result."
                )

        # --- Coordinates extraction formatting ---
        has_coords = "RA_ICRS" in matched_row.colnames and "DE_ICRS" in matched_row.colnames
        ra_sexagesimal = None
        dec_sexagesimal = None
        if (
            has_coords
            and not np.isnan(matched_row["RA_ICRS"])
            and not np.isnan(matched_row["DE_ICRS"])
        ):
            try:
                coord = SkyCoord(
                    ra=float(matched_row["RA_ICRS"]) * u.deg,
                    dec=float(matched_row["DE_ICRS"]) * u.deg,
                    frame="icrs",
                )
                ra_sexagesimal = coord.ra.to_string(unit=u.hour, sep=":", precision=2, pad=True)
                dec_sexagesimal = coord.dec.to_string(
                    unit=u.deg, sep=":", precision=1, pad=True, alwayssign=True
                )
            except Exception:
                pass

        # --- Extraction Logic ---
        extracted = []
        for col, (var_name, var_symbol, default_unit) in cfg["columns"].items():
            if col not in table.colnames:
                continue

            val = matched_row[col]
            if (hasattr(val, "mask") and val.mask) or str(val).strip() in (
                "--",
                "nan",
                "NaN",
                "None",
            ):
                val = np.nan

            err_col = f"e_{col}"
            err = matched_row[err_col] if err_col in table.colnames else np.nan
            if (hasattr(err, "mask") and err.mask) or str(err).strip() in (
                "--",
                "nan",
                "NaN",
                "None",
            ):
                err = np.nan

            unit = str(table[col].unit) if table[col].unit else default_unit

            if col == "RA_ICRS" and ra_sexagesimal is not None:
                val = ra_sexagesimal
                err = np.nan
                unit = ""
            elif col == "DE_ICRS" and dec_sexagesimal is not None:
                val = dec_sexagesimal
                err = np.nan
                unit = ""

            extracted.append(
                {
                    "name": var_name,
                    "symbol": var_symbol,
                    "value": val,
                    "uncertainty": err,
                    "unit": unit,
                    "source": cfg["source"],
                }
            )

        if config_key == "gaia":
            corrected_plx_entry = self._correct_parallax(matched_row)
            if corrected_plx_entry:
                extracted = [e for e in extracted if e["name"] != "parallax"]
                extracted.append(corrected_plx_entry)
                dist_entry = self._calculate_distance_from_parallax(
                    corrected_plx_entry["value"], corrected_plx_entry["uncertainty"]
                )
            else:
                dist_entry = self._calculate_distance_from_parallax(
                    matched_row["Plx"], matched_row["e_Plx"]
                )

            if dist_entry:
                extracted.append(dist_entry)

        return extracted

    def query_catalog_region(self, tic_result):
        """Query Vizier catalogs for the 2 arcsec region around the target.

        Use TIC coordinates rather than a Gaia ID to avoid DR3/EDR3 mismatch issues.
        Returns a dict of astropy Tables or raw TableList.
        """
        if isinstance(tic_result, astropy.table.Table):
            if len(tic_result) == 1:
                tic_result = tic_result[0]
            else:
                raise ValueError("Expected a single row in tic_result table.")

        tic_id = str(tic_result.get("ID", ""))

        if self.cache_dir is not None and tic_id:
            cache_file = self.cache_dir / f"vizier_tic_{tic_id}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        cached_data = json.load(f)
                    # Reconstruct dict of astropy Tables
                    return {k: astropy.table.Table(rows=v) for k, v in cached_data.items()}
                except Exception as e:
                    logger.warning(f"Failed to read cache {cache_file}: {e}")

        coord = SkyCoord(ra=tic_result["ra"], dec=tic_result["dec"], unit="deg")
        results = self.viz.query_region(
            coord, radius=u.Quantity(2, "arcsec"), catalog=self.catalogs
        )

        if self.cache_dir is not None and tic_id and results is not None:
            cache_file = self.cache_dir / f"vizier_tic_{tic_id}.json"
            try:
                serializable = {}
                keys = list(results.keys()) if hasattr(results, "keys") else []
                for key in keys:
                    tbl = results[key]
                    serializable[key] = [
                        dict(zip(tbl.colnames, [row[c] for c in tbl.colnames], strict=False))
                        for row in tbl
                    ]
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(serializable, f, indent=2, default=str)
            except Exception as e:
                logger.warning(f"Failed to write cache {cache_file}: {e}")

        return results

    def get_complete_system_data(self, tic_result) -> pd.DataFrame:
        """Main orchestrator to query and extract system astrometry and photometry."""
        if isinstance(tic_result, astropy.table.Table):
            if len(tic_result) == 1:
                tic_result = tic_result[0]
            else:
                raise ValueError("Expected a single row in tic_result table.")

        gaia_id = tic_result.get("GAIA")
        query_results = self.query_catalog_region(tic_result)

        all_entries = []

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

        # TESS Mag (from TIC)
        all_entries.append(
            {
                "name": "tess_mag",
                "symbol": r"$T$",
                "value": tic_result.get("Tmag"),
                "uncertainty": tic_result.get("e_Tmag"),
                "unit": "mag",
                "source": "Sta19",
            }
        )
        return pd.DataFrame(all_entries)

    def extract_identifier_data(
        self,
        tic_result,
        star_name: str | None = None,
        spectral_type: str | None = None,
        spectral_type_source: str = "This work",
        reference_style: str = "survey",
    ) -> pd.DataFrame:
        """Extracts names/identifiers and references for the top section of the table."""
        if isinstance(tic_result, astropy.table.Table):
            if len(tic_result) == 1:
                tic_result = tic_result[0]
            else:
                raise ValueError("Expected a single row in tic_result table.")

        tic_id = str(tic_result.get("ID", ""))
        toi_str = None

        if self.cache_dir is not None and tic_id:
            cache_file = self.cache_dir / f"toi_tic_{tic_id}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        toi_str = json.load(f).get("toi")
                except Exception as e:
                    logger.warning(f"Failed to read cache {cache_file}: {e}")

        if toi_str is None and tic_id:
            try:
                toi_data = NasaExoplanetArchive.query_criteria(
                    table="toi", where=f"tic_id={tic_id}"
                )
                if len(toi_data) > 0 and "toipfx" in toi_data.colnames:
                    pfx = toi_data["toipfx"][0]
                    toi_str = f"TOI-{pfx}" if not str(pfx).startswith("TOI-") else str(pfx)
                if self.cache_dir is not None:
                    cache_file = self.cache_dir / f"toi_tic_{tic_id}.json"
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump({"toi": toi_str}, f)
            except Exception:
                logger.warning(f"TOI query failed for TIC {tic_id}")

        names = []
        symbols = []
        vals = []
        sources = []

        if star_name is not None:
            names.append("star_name")
            symbols.append("Name")
            vals.append(star_name)
            sources.append("")

        if toi_str and toi_str != "N/A":
            names.append("toi")
            symbols.append("TOI")
            vals.append(toi_str)
            sources.append("ExoFOP")

        names.extend(["tic", "tyc", "gaia_dr3", "twomass", "allwise"])
        symbols.extend(["TIC", "TYC", "Gaia DR3", "2MASS", "ALLWISE"])
        vals.extend(
            [
                str(tic_result.get("ID")),
                self._normalize_tyc(tic_result.get("TYC")),
                str(tic_result.get("GAIA")),
                str(tic_result.get("TWOMASS")),
                str(tic_result.get("ALLWISE")),
            ]
        )
        if reference_style == "author":
            sources.extend(["Sta19", "Høg00", "Gaia DR3", "Skr06", "Cut13"])
        else:
            sources.extend(["", "", "", "", ""])

        if spectral_type is not None:
            names.append("spectral_type")
            symbols.append("Spectral type")
            vals.append(spectral_type)
            sources.append(spectral_type_source)

        return pd.DataFrame(
            {
                "name": names,
                "symbol": symbols,
                "value": vals,
                "uncertainty": [np.nan] * len(names),
                "unit": [""] * len(names),
                "source": sources,
            }
        )

    @classmethod
    def get_tic_results(cls, tic_id, cache_dir: str | Path | None = None):
        """Query TIC catalog with optional disk caching (in ECSV format)."""
        query_id = f"TIC {tic_id}" if not str(tic_id).startswith("TIC") else tic_id

        cache_path = None
        if cache_dir is not None:
            cache_path = Path(cache_dir) / f"tic_{tic_id}.ecsv"
            if cache_path.exists():
                try:
                    return astropy.table.Table.read(cache_path, format="ascii.ecsv")
                except Exception as e:
                    logger.warning(f"Failed to read TIC cache from {cache_path}: {e}")

        res = Catalogs.query_object(objectname=query_id, radius=0.001, catalog="TIC")

        if cache_path is not None and res is not None:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                res.write(cache_path, format="ascii.ecsv", overwrite=True)
            except Exception as e:
                logger.warning(f"Failed to save TIC cache to {cache_path}: {e}")

        return res

    def _calculate_distance_from_parallax(self, plx, e_plx):
        try:
            plx_val = float(plx)
            if np.isnan(plx_val) or plx_val <= 0:
                return None
        except (ValueError, TypeError):
            return None

        try:
            e_plx_val = float(e_plx)
            if np.isnan(e_plx_val):
                e_plx_val = np.nan
        except (ValueError, TypeError):
            e_plx_val = np.nan

        dist = 1000.0 / plx_val
        e_dist = (e_plx_val / plx_val) * dist if not np.isnan(e_plx_val) else np.nan
        return {
            "name": "distance",
            "symbol": r"$d$",
            "value": dist,
            "uncertainty": e_dist,
            "unit": "pc",
            "source": "This work",
        }

    def _correct_parallax(self, gaia_row):
        """Applies parallax zero-point correction based on Gaia EDR3 / DR3."""
        try:
            from zero_point import zpt

            zpt.load_tables()
        except ImportError:
            logger.warning("zpt module not found. Parallax correction will be skipped.")
            return None

        try:

            def _unmasked(val):
                if val is None or str(val).strip() in ("--", "nan", "NaN", "None", ""):
                    return np.nan
                try:
                    return np.float64(np.ma.filled(val, np.nan))
                except Exception:
                    return np.nan

            g_mag = _unmasked(gaia_row["Gmag"])
            nueff = _unmasked(gaia_row["nueff"])
            pscol = _unmasked(gaia_row["pscol"])
            glat = _unmasked(gaia_row["GLAT"])
            solved = _unmasked(gaia_row["Solved"])

            if np.isnan(g_mag) or np.isnan(glat) or np.isnan(solved):
                return None

            zero_point = float(
                zpt.get_zpt(
                    np.array([g_mag], dtype=np.float64),
                    np.array([nueff], dtype=np.float64),
                    np.array([pscol], dtype=np.float64),
                    np.array([glat], dtype=np.float64),
                    np.array([int(solved)]),
                )[0]
            )
            plx_raw = float(gaia_row["Plx"])
            parallax_corrected = plx_raw - zero_point
        except Exception as e:
            logger.warning(f"Parallax correction failed: {e}")
            return None

        e_plx = float(gaia_row["e_Plx"]) if "e_Plx" in gaia_row.colnames else np.nan

        return {
            "name": "parallax",
            "symbol": r"$\varpi$",
            "value": parallax_corrected,
            "uncertainty": e_plx,
            "unit": "mas",
            "source": "Gaia DR3; Lin21",
        }

    def add_uvw(
        self,
        df: pd.DataFrame,
        n_samples: int = 1_000_000,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Returns a new DataFrame with UVW space velocities appended."""

        def get_quantity(var_name, default_unit=None):
            mask = df["name"] == var_name
            if mask.any():
                row = df[mask].iloc[0]
                val = row["value"]
                unc = row["uncertainty"]
                unit_str = row["unit"]

                try:
                    unc_val = float(unc)
                    unc_val = 0.0 if np.isnan(unc_val) else unc_val
                except (ValueError, TypeError):
                    unc_val = 0.0

                if isinstance(val, str):
                    coord = SkyCoord(
                        val,
                        unit=(u.hourangle if "hour" in unit_str or ":" in val else u.deg),
                    )
                    return (
                        coord.deg * u.deg,
                        unc_val * u.deg,
                    )

                unit = (
                    u.Unit(unit_str)
                    if (unit_str and unit_str.strip())
                    else (u.Unit(default_unit) if default_unit else u.dimensionless_unscaled)
                )
                return u.Quantity(float(val), unit), u.Quantity(unc_val, unit)
            raise ValueError(f"'{var_name}' not found in DataFrame.")

        ra_val = df[df["name"] == "ra"].iloc[0]["value"]
        dec_val = df[df["name"] == "dec"].iloc[0]["value"]

        if isinstance(ra_val, str) and isinstance(dec_val, str):
            coord = SkyCoord(ra_val, dec_val, unit=(u.hourangle, u.deg), frame="icrs")
            ra = coord.ra
            dec = coord.dec
            e_ra = 0.0 * u.deg
            e_dec = 0.0 * u.deg
        else:
            ra, e_ra = get_quantity("ra", "deg")
            dec, e_dec = get_quantity("dec", "deg")

        pm_ra, e_pm_ra = get_quantity("pm_ra", "mas/yr")
        pm_dec, e_pm_dec = get_quantity("pm_dec", "mas/yr")
        rv, e_rv = get_quantity("radial_velocity", "km/s")
        distance, e_distance = get_quantity("distance", "pc")

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
            name = comp.lower()
            symbol = rf"${comp[0]}_\mathrm{{LSR}}$"
            new_rows.append(
                {
                    "name": name,
                    "symbol": symbol,
                    "value": mean.value,
                    "uncertainty": std.value,
                    "unit": "km/s",
                    "source": "This work",
                }
            )

        return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    def add_galactic_population(
        self,
        df: pd.DataFrame,
        population: str = "Thin disc",
        source: str = "This work",
    ) -> pd.DataFrame:
        """Appends Galactic population entry to the kinematics DataFrame."""
        row = {
            "name": "galactic_population",
            "symbol": "Galactic population",
            "value": population,
            "uncertainty": np.nan,
            "unit": "",
            "source": source,
        }
        return pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    @classmethod
    def save_system_data(
        cls,
        filepath: str | Path,
        id_df: pd.DataFrame,
        param_df: pd.DataFrame,
        metadata: dict | None = None,
    ) -> Path:
        """Save extracted identifier and system parameter tables to a human-readable JSON file (default) or CSV."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix == ".csv":
            id_tagged = id_df.copy()
            id_tagged["_section"] = "identifiers"
            param_tagged = param_df.copy()
            param_tagged["_section"] = "parameters"
            combined = pd.concat([id_tagged, param_tagged], ignore_index=True)
            combined.to_csv(path, index=False)
        else:
            if not path.suffix:
                path = path.with_suffix(".json")
            payload = {
                "metadata": metadata or {},
                "id_df": id_df.to_dict(orient="records"),
                "param_df": param_df.to_dict(orient="records"),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
        return path

    @classmethod
    def load_system_data(cls, filepath: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Load saved identifier and parameter tables from a JSON or CSV file."""
        path = Path(filepath)
        if not path.exists():
            if path.with_suffix(".json").exists():
                path = path.with_suffix(".json")
            else:
                raise FileNotFoundError(f"File not found: {filepath}")

        if path.suffix == ".csv":
            combined = pd.read_csv(path)
            id_df = combined[combined["_section"] == "identifiers"].drop(columns=["_section"])
            param_df = combined[combined["_section"] == "parameters"].drop(columns=["_section"])
            metadata = {}
        else:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            id_df = pd.DataFrame(payload["id_df"])
            param_df = pd.DataFrame(payload["param_df"])
            metadata = payload.get("metadata", {})

        return id_df, param_df, metadata

    @staticmethod
    def _format_unit_aa(unit_raw: str) -> str:
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
            "Rsun": r"R_\odot",
            "R_sun": r"R_\odot",
            "Msun": r"M_\odot",
            "M_sun": r"M_\odot",
            "Lsun": r"L_\odot",
            "L_sun": r"L_\odot",
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

    @classmethod
    def generate_stellar_table_latex(
        cls,
        star_name: str,
        id_df: pd.DataFrame,
        param_df: pd.DataFrame,
        significant_digits: int = 2,
        adaptive_sigfigs: bool = False,
        units_in_parameter: bool = True,
        unit_brackets: str = "round",
        section_hlines: bool = True,
        noalign_smallskip: bool = False,
        reference_style: str = "survey",
        custom_references: dict[str, str] | None = None,
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

        # 2. Categorization by machine-readable names with deterministic ordering
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
            order_dict = {k: i for i, k in enumerate(key_order)}
            ranks = df_subset["name"].map(lambda x: order_dict.get(x, 999))
            return df_subset.iloc[np.argsort(ranks, kind="stable")]

        id_subset = _sort_by_keys(id_df, identifier_keys)
        astrometric_subset = _sort_by_keys(
            param_df[param_df["name"].isin(astrometric_keys)], astrometric_keys
        )
        photo_raw = param_df[
            param_df["name"].isin(photometric_keys)
            | (
                ~param_df["name"].isin(astrometric_keys)
                & param_df["unit"].astype(str).str.contains("mag")
            )
        ]
        photo_subset = _sort_by_keys(photo_raw, photometric_keys)

        sections = [
            ("Basic identifiers and data", id_subset),
            ("Astrometric properties", astrometric_subset),
            ("Photometric properties", photo_subset),
        ]

        # Remaining parameters (Fundamental parameters / Activity)
        used_keys = set(id_df["name"]).union(astrometric_keys).union(photometric_keys)
        remaining = param_df[~param_df["name"].isin(used_keys)]
        if not remaining.empty:
            sections.append(("Fundamental parameters", remaining))

        open_b = "[" if unit_brackets in ("square", "[]") else "("
        close_b = "]" if unit_brackets in ("square", "[]") else ")"

        author_to_survey = {
            "Skr06": "2MASS",
            "Cut13": "ALLWISE",
            "Høg00": "Tycho-2",
            "Sta19": "TIC",
            "Gai23": "Gaia DR3",
        }
        survey_to_author = {
            "2MASS": "Skr06",
            "ALLWISE": "Cut13",
            "WISE": "Cut13",
            "Tycho-2": "Høg00",
            "Tycho": "Høg00",
            "TIC": "Sta19",
        }

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
                p_sym = str(row["symbol"])
                p_tex = p_sym if "$" in p_sym else p_sym.replace("_", r"\_")

                v_raw = row["value"]
                e_raw = row["uncertainty"]

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

                # Check if numerical
                try:
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

                if section_title == "Basic identifiers and data":
                    val_str = str(val).replace("_", r"\_")
                elif is_num:
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

                unit_raw = str(row.get("unit", ""))
                unit_fmt = cls._format_unit_aa(unit_raw)

                source_raw = str(row.get("source", ""))
                if reference_style == "survey":
                    source_mapped = author_to_survey.get(source_raw, source_raw)
                    # Omit self-referential catalog references in the identifiers section
                    if section_title == "Basic identifiers and data" and source_mapped in (
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
                    source_mapped = survey_to_author.get(source_raw, source_raw)
                else:
                    source_mapped = source_raw

                if custom_references and source_mapped in custom_references:
                    source_mapped = custom_references[source_mapped]

                source = source_mapped.replace("_", r"\_")

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

        latex.append(r"\end{table}")
        return "\n".join(latex)

    @staticmethod
    def _normalize_tyc(tid):
        if not tid or str(tid).strip() == "":
            return None
        parts = str(tid).strip().split("-")
        if len(parts) == 3:
            return f"{int(parts[0])}-{int(parts[1])}-{int(parts[2])}"
        return str(tid).strip()
