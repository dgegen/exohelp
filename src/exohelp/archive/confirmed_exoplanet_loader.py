import io
import os
from datetime import datetime

import pandas as pd
import numpy as np
import httpx


data = {
    "Unit": [
        "",
        r"$\mathrm{days}$",
        r"$\mathrm{AU}$",
        r"$\mathrm{R}_{\oplus}$",
        r"$\mathrm{R}_{\mathrm{J}}$",
        r"$\mathrm{M}_{\oplus}$",
        r"$\mathrm{M}_{\mathrm{J}}$",
        "",
        r"$\mathrm{S}_{\odot}$",
        r"$\mathrm{K}$",
        r"$\mathrm{K}$",
        r"$\mathrm{R}_{\odot}$",
        r"$\mathrm{M}_{\odot}$",
        "",
        r"$\mathrm{cm/s^2}$",
        r"$\mathrm{parsecs}$",
        r"$\mathrm{mag}$",
        r"$\mathrm{mag}$",
        r"$\mathrm{mag}$",
    ],
    "Latex name": [
        "Name",
        r"$P_{\mathrm{orb}}$",
        r"$a_{\mathrm{orb}}$",
        r"$R_{\mathrm{p}}$",
        r"$R_{\mathrm{j}}$",
        r"$M_{\mathrm{p}}$",
        r"$M_{\mathrm{j}}$",
        r"$e$",
        r"$S_{\mathrm{ins}}$",
        r"$T_{\mathrm{eq}}$",
        r"$T_{\mathrm{eff}}$",
        r"$R_{\star}$",
        r"$M_{\star}$",
        r"$Z$",
        r"$\log(g)$",
        r"$d_{\mathrm{sy}}$",
        r"$V_{\mathrm{mag}}$",
        r"$K_{\mathrm{mag}}$",
        r"$G_{\mathrm{mag}}$",
    ],
    "Common variable name": [
        "Planet Name",
        "Orbital Period",
        "Orbital Semi-Major Axis",
        "Planet Radius",
        "Jupiter Radius",
        "Planet Mass",
        "Jupiter Mass",
        "Orbital Eccentricity",
        "Incident Stellar Flux",
        "Equilibrium Temperature",
        "Stellar Effective Temperature",
        "Stellar Radius",
        "Stellar Mass",
        "Stellar Metallicity",
        "Stellar Surface Gravity",
        "System Distance",
        "System Visual Magnitude",
        "System K-band Magnitude",
        "System G-band Magnitude",
    ],
}

# Create the DataFrame

# Create the DataFrame
RENAME_DF = pd.DataFrame(
    data,
    index=[
        "pl_name",
        "pl_orbper",
        "pl_orbsmax",
        "pl_rade",
        "pl_radj",
        "pl_bmasse",
        "pl_bmassj",
        "pl_orbeccen",
        "pl_insol",
        "pl_eqt",
        "st_teff",
        "st_rad",
        "st_mass",
        "st_met",
        "st_logg",
        "sy_dist",
        "sy_vmag",
        "sy_kmag",
        "sy_gaiamag",
    ],
)

# Latex name with units, remove suffix $ of name prefix $ of unit, insert \, inbetween
RENAME_DF["Latex name with units"] = (
    RENAME_DF["Latex name"].str.removesuffix("$")
    + r"\,("
    + RENAME_DF["Unit"].str.replace("$", "")
    + ")$"
).where(
    RENAME_DF["Unit"] != "",
    RENAME_DF["Latex name"],
)


class ConfirmedExoplanetLoader:
    DEFAULT_COLUMNS = (
        "pl_name",
        "hostname",
        "default_flag",
        "sy_snum",
        "sy_pnum",
        "discoverymethod",
        "disc_year",
        "disc_facility",
        "soltype",
        "pl_controv_flag",
        "pl_refname",
        "pl_orbper",
        "pl_orbpererr1",
        "pl_orbpererr2",
        "pl_orbperlim",
        "pl_orbsmax",
        "pl_orbsmaxerr1",
        "pl_orbsmaxerr2",
        "pl_orbsmaxlim",
        "pl_rade",
        "pl_radeerr1",
        "pl_radeerr2",
        "pl_radelim",
        "pl_radj",
        "pl_radjerr1",
        "pl_radjerr2",
        "pl_radjlim",
        "pl_bmasse",
        "pl_bmasseerr1",
        "pl_bmasseerr2",
        "pl_bmasselim",
        "pl_bmassj",
        "pl_bmassjerr1",
        "pl_bmassjerr2",
        "pl_bmassjlim",
        "pl_bmassprov",
        "pl_orbeccen",
        "pl_orbeccenerr1",
        "pl_orbeccenerr2",
        "pl_orbeccenlim",
        "pl_insol",
        "pl_insolerr1",
        "pl_insolerr2",
        "pl_insollim",
        "pl_eqt",
        "pl_eqterr1",
        "pl_eqterr2",
        "pl_eqtlim",
        "ttv_flag",
        "st_refname",
        "st_spectype",
        "st_teff",
        "st_tefferr1",
        "st_tefferr2",
        "st_tefflim",
        "st_rad",
        "st_raderr1",
        "st_raderr2",
        "st_radlim",
        "st_mass",
        "st_masserr1",
        "st_masserr2",
        "st_masslim",
        "st_met",
        "st_meterr1",
        "st_meterr2",
        "st_metlim",
        "st_metratio",
        "st_logg",
        "st_loggerr1",
        "st_loggerr2",
        "st_logglim",
        "sy_refname",
        "rastr",
        "ra",
        "decstr",
        "dec",
        "sy_dist",
        "sy_disterr1",
        "sy_disterr2",
        "sy_vmag",
        "sy_vmagerr1",
        "sy_vmagerr2",
        "sy_kmag",
        "sy_kmagerr1",
        "sy_kmagerr2",
        "sy_gaiamag",
        "sy_gaiamagerr1",
        "sy_gaiamagerr2",
        "rowupdate",
        "pl_pubdate",
        "releasedate",
    )
    RENAME_DF = RENAME_DF

    def __init__(self, output_dir=None):
        self._output_dir = output_dir
        if self._output_dir is not None:
            os.makedirs(self._output_dir, exist_ok=True)

    @property
    def output_dir(self):
        if self._output_dir is None:
            raise ValueError("Output directory is not set.")
        return self._output_dir

    def load(self, reduced_columns=False, use_cache=True):
        # Check if the most recent file is already downloaded
        most_recent_file_path = self.get_most_recent_file_path()

        if most_recent_file_path and use_cache:
            df = pd.read_csv(most_recent_file_path)
        else:
            df = self.download_confirmed_exoplanets_dataframe(reduced_columns=reduced_columns)

        return df

    def download_confirmed_exoplanets_dataframe(self, reduced_columns=True):
        content = self.download_confirmed_exoplanets_csv(reduced_columns=reduced_columns)
        df = pd.read_csv(io.StringIO(content))
        if reduced_columns:
            df = df.loc[:, self.DEFAULT_COLUMNS]

        self.save_to_csv(df)
        return df

    def download_confirmed_exoplanets_csv(self, reduced_columns=True):
        httpx.get("https://duckduckgo.com/")  # Warm up the httpx client

        url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
        url += "query=select+*+from+ps+where+default_flag=1&format=csv"

        # Send a GET request to the API using httpx
        with httpx.Client() as client:
            response = client.get(url, timeout=60)
            if response.status_code == 200:
                content = response.text
            else:
                print("Error fetching data:", response.status_code)
                return None

        return content

    def save_to_csv(self, df):
        if self._output_dir is None:
            return None
        file_name = datetime.now().strftime("%Y_%m_%d_confirmed_exoplanets.csv")
        file_path = os.path.join(self.output_dir, file_name)

        df.to_csv(file_path, index=False)
        return file_path

    def get_most_recent_file_path(self):
        if self._output_dir is None:
            return None

        output_dir = self.output_dir

        files = [item for item in os.listdir(output_dir) if "confirmed_exoplanets" in item]
        files.sort()

        if len(files) == 0:
            return None

        return os.path.join(output_dir, files[-1])


class SolarSystemPlanetLoader:
    URL: str = "https://nssdc.gsfc.nasa.gov/planetary/factsheet/planet_table_ratio.html"

    def __init__(self, output_dir=None) -> None:
        self._output_dir = output_dir
        if self._output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

    @property
    def output_dir(self):
        if self._output_dir is None:
            raise ValueError("Output directory is not set.")
        return self._output_dir

    def load(self, use_cache=True):
        # Check if the most recent file is already downloaded
        file_path = self.get_path()

        if file_path and use_cache:
            df = pd.read_csv(file_path, index_col=0)
        else:
            df = self.download_and_clean_solar_system_planets_table()

        return df

    def download_and_clean_solar_system_planets_table(self):
        df = self.download_solar_system_planets_dataframe()
        df = self.clean_load_solar_system_planets_table(df)
        self.save_to_csv(df)

        return df

    def download_solar_system_planets_dataframe(self):
        df = pd.read_html(
            self.URL,
            na_values=["Unknown*", "Unknown"],
            header=0,
            index_col=0,
        )[0].T

        return df

    @staticmethod
    def clean_load_solar_system_planets_table(df):
        df = df.replace(to_replace=r"\*", value="", regex=True).iloc[:, :-1]
        df = df.replace({"No": "0", "Yes": "1"})
        df = df.astype("float")

        df.columns = df.columns.str.lower()
        df.columns = df.columns.str.replace(" ", "_")
        df.columns = df.columns.str.replace("?", "")  # Remove question marks
        df.columns = df.columns.str.replace(".", "")

        df.index = df.index.str.lower()
        return df

    def get_path(self):
        output_dir = self.output_dir

        files = [item for item in os.listdir(output_dir) if item == "solar_system_planets.csv"]
        files.sort()

        if len(files) == 0:
            return None

        return os.path.join(output_dir, files[-1])

    def save_to_csv(self, df):
        file_path = os.path.join(ConfirmedExoplanetLoader.OUTPUTS_DIR, "solar_system_planets.csv")

        df.to_csv(file_path, index=True)
        return file_path


def get_unique_counts(df, column):
    unique, count = np.unique(df[column], return_counts=True)
    sort_idx = np.argsort(count)[::-1]
    unique = unique[sort_idx]
    count = count[sort_idx]
    return dict(zip(unique, count))
