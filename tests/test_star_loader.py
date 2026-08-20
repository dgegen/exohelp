import numpy as np
import pandas as pd
import pytest

from exohelp.archive.star_loader import StarLoader


@pytest.fixture
def mock_star_data():
    id_df = pd.DataFrame(
        {
            "name": [
                "star_name",
                "toi",
                "tic",
                "tyc",
                "gaia_dr3",
                "twomass",
                "allwise",
                "spectral_type",
            ],
            "symbol": [
                "Name",
                "TOI",
                "TIC",
                "TYC",
                "Gaia DR3",
                "2MASS",
                "ALLWISE",
                "Spectral type",
            ],
            "value": [
                "HD 81466",
                "TOI-1234",
                "334632624",
                "5477-849-1",
                "5672159847123984",
                "09252341-1555194",
                "J092523.41-155519.4",
                "G8 V",
            ],
            "uncertainty": [np.nan] * 8,
            "unit": [""] * 8,
            "source": [
                "",
                "ExoFOP",
                "Sta19",
                "Høg00",
                "Gaia DR3",
                "Skr06",
                "Cut13",
                "This work",
            ],
        }
    )

    param_df = pd.DataFrame(
        [
            {
                "name": "ra",
                "symbol": r"$\alpha\ (\mathrm{J2016.0})$",
                "value": "09:25:23.41",
                "uncertainty": np.nan,
                "unit": "",
                "source": "Gaia DR3",
            },
            {
                "name": "dec",
                "symbol": r"$\delta\ (\mathrm{J2016.0})$",
                "value": "-15:55:19.4",
                "uncertainty": np.nan,
                "unit": "",
                "source": "Gaia DR3",
            },
            {
                "name": "pm_ra",
                "symbol": r"$\mu_\alpha \cos \delta$",
                "value": -24.52,
                "uncertainty": 0.03,
                "unit": "mas/yr",
                "source": "Gaia DR3",
            },
            {
                "name": "pm_dec",
                "symbol": r"$\mu_\delta$",
                "value": 12.18,
                "uncertainty": 0.02,
                "unit": "mas/yr",
                "source": "Gaia DR3",
            },
            {
                "name": "radial_velocity",
                "symbol": r"$\gamma$",
                "value": 31.45,
                "uncertainty": 0.12,
                "unit": "km/s",
                "source": "Gaia DR3",
            },
            {
                "name": "parallax",
                "symbol": r"$\varpi$",
                "value": 14.23,
                "uncertainty": 0.04,
                "unit": "mas",
                "source": "Gaia DR3; Lin21",
            },
            {
                "name": "distance",
                "symbol": r"$d$",
                "value": 70.27,
                "uncertainty": 0.20,
                "unit": "pc",
                "source": "This work",
            },
            {
                "name": "g_mag",
                "symbol": r"$G$",
                "value": 8.765,
                "uncertainty": 0.002,
                "unit": "mag",
                "source": "Gaia DR3",
            },
            {
                "name": "bp_mag",
                "symbol": r"$G_\mathrm{BP}$",
                "value": 9.120,
                "uncertainty": 0.003,
                "unit": "mag",
                "source": "Gaia DR3",
            },
            {
                "name": "rp_mag",
                "symbol": r"$G_\mathrm{RP}$",
                "value": 8.245,
                "uncertainty": 0.003,
                "unit": "mag",
                "source": "Gaia DR3",
            },
            {
                "name": "ks_mag",
                "symbol": r"$K_\mathrm{s}$",
                "value": 7.123,
                "uncertainty": 0.021,
                "unit": "mag",
                "source": "Skr06",
            },
            {
                "name": "tess_mag",
                "symbol": r"$T$",
                "value": 8.192,
                "uncertainty": 0.006,
                "unit": "mag",
                "source": "Sta19",
            },
        ]
    )

    return id_df, param_df


def test_generate_stellar_table_latex_structure(mock_star_data):
    id_df, param_df = mock_star_data
    latex = StarLoader.generate_stellar_table_latex(
        "HD 81466",
        id_df,
        param_df,
        significant_digits=2,
        tablefoot_notes={"1": r"\citet{GaiaCollaboration2023}", "a": "Zero-point corrected."},
    )

    # 1. Check single tabular structure
    assert r"\begin{tabular}{l c r}" in latex
    assert latex.count(r"\begin{tabular}") == 1
    assert latex.count(r"\end{tabular}") == 1

    # 2. Check centered italicized section headers with hlines
    assert r"\multicolumn{3}{c}{\textit{Basic identifiers and data}}" in latex
    assert r"\multicolumn{3}{c}{\textit{Astrometric properties}}" in latex
    assert r"\multicolumn{3}{c}{\textit{Photometric properties}}" in latex
    assert r"\noalign{\smallskip}" not in latex

    # 3. Check noalign_smallskip option
    latex_noalign = StarLoader.generate_stellar_table_latex(
        "HD 81466", id_df, param_df, noalign_smallskip=True
    )
    assert r"\noalign{\smallskip}" in latex_noalign

    # 4. Check units with parentheses (default)
    assert r"$\mu_\alpha \cos \delta\ (\mathrm{mas\,a^{-1}})$" in latex
    assert r"$\varpi\ (\mathrm{mas})$" in latex
    assert r"$d\ (\mathrm{pc})$" in latex
    assert r"$G\ (\mathrm{mag})$" in latex
    assert r"$K_\mathrm{s}\ (\mathrm{mag})$" in latex

    # 5. Check square brackets option
    latex_sq = StarLoader.generate_stellar_table_latex(
        "HD 81466", id_df, param_df, unit_brackets="square"
    )
    assert r"$\mu_\alpha \cos \delta\ [\mathrm{mas\,a^{-1}}]$" in latex_sq
    assert r"$\varpi\ [\mathrm{mas}]$" in latex_sq
    assert r"$d\ [\mathrm{pc}]$" in latex_sq

    # 6. Check sexagesimal coordinate formatting without \pm
    assert "09:25:23.41" in latex
    assert "-15:55:19.4" in latex

    # 7. Check survey reference style (default)
    assert "2MASS" in latex
    assert "TIC" in latex

    # 8. Check author reference style
    latex_auth = StarLoader.generate_stellar_table_latex(
        "HD 81466", id_df, param_df, reference_style="author"
    )
    assert "Skr06" in latex_auth
    assert "Sta19" in latex_auth

    # 9. Check custom reference mapping
    latex_custom = StarLoader.generate_stellar_table_latex(
        "HD 81466", id_df, param_df, custom_references={"2MASS": r"\citet{Cutri2003}"}
    )
    assert r"\citet{Cutri2003}" in latex_custom

    # 10. Check unruled section headers (section_hlines=False)
    latex_nohline = StarLoader.generate_stellar_table_latex(
        "HD 81466", id_df, param_df, section_hlines=False
    )
    assert r"\multicolumn{3}{l}{\textit{Basic identifiers and data}}" in latex_nohline
    assert r"\multicolumn{3}{l}{\textit{Astrometric properties}}" in latex_nohline

    # 11. Check tablefoot & tablebib
    assert r"\tablefoot{" in latex
    assert r"\tablefoottext{1}{\citet{GaiaCollaboration2023}};" in latex
    assert r"\tablefoottext{a}{Zero-point corrected.};" in latex
    assert r"\tablebib{" in latex
    assert r"\textit{Gaia} DR3: \citet{GaiaCollaboration2023}" in latex
    assert r"2MASS: \citet{Skrutskie2006}" in latex


def test_save_stellar_table_latex(tmp_path, mock_star_data):
    id_df, param_df = mock_star_data
    tex_path = tmp_path / "sub" / "table.tex"
    out_file = StarLoader.save_stellar_table_latex(
        tex_path, "HD 81466", id_df, param_df, unit_brackets="square"
    )
    assert out_file.exists()
    content = out_file.read_text()
    assert r"\begin{table}[!ht]" in content
    assert r"$\mu_\alpha \cos \delta\ [\mathrm{mas\,a^{-1}}]$" in content


def test_get_default_bibtex():
    bib_text = StarLoader.get_default_bibtex()
    assert "GaiaCollaboration2023" in bib_text
    assert "Skrutskie2006" in bib_text


def test_spectral_type_estimation():
    loader = StarLoader()
    assert loader.estimate_spectral_type(5956) == r"G0\,V"
    assert loader.estimate_spectral_type(5400) == r"G8\,V"
    assert loader.estimate_spectral_type(3500) == r"M\,V"
    assert loader.get_spectral_type(teff=5845) == r"G2\,V"


def test_custom_sections_and_asymmetric_uncertainties():
    id_df = pd.DataFrame(
        [
            {
                "section": "Identifiers",
                "name": "tic",
                "symbol": "TIC",
                "value": "334632624",
                "uncertainty": np.nan,
                "unit": "",
                "source": "TIC",
            }
        ]
    )
    param_df = pd.DataFrame(
        [
            {
                "section": "Fundamental parameters",
                "name": "m_star",
                "symbol": r"$M_\star$",
                "value": 1.045,
                "uncertainty": (0.057, -0.066),
                "unit": "M_sun",
                "source": r"Sect.~\ref{sec:mass_age}",
            }
        ]
    )
    latex = StarLoader.generate_stellar_table_latex(
        "HD 81466", id_df, param_df, unit_brackets="square"
    )
    assert r"\multicolumn{3}{c}{\textit{Identifiers}}" in latex
    assert r"\multicolumn{3}{c}{\textit{Fundamental parameters}}" in latex
    assert r"$M_\star\ [\mathrm{M}_\odot]$" in latex
    assert r"$1.045^{+0.057}_{-0.066}$" in latex
    assert r"Sect.~\ref{sec:mass_age}" in latex


def test_save_and_load_system_data(tmp_path, mock_star_data):
    id_df, param_df = mock_star_data
    save_file_json = tmp_path / "system_data.json"
    save_file_csv = tmp_path / "system_data.csv"

    # JSON save/load
    StarLoader.save_system_data(save_file_json, id_df, param_df, metadata={"star": "HD 81466"})
    assert save_file_json.exists()

    loaded_id_json, loaded_param_json, meta_json = StarLoader.load_system_data(save_file_json)
    assert len(loaded_id_json) == len(id_df)
    assert len(loaded_param_json) == len(param_df)
    assert meta_json.get("star") == "HD 81466"

    # CSV save/load
    StarLoader.save_system_data(save_file_csv, id_df, param_df)
    assert save_file_csv.exists()

    loaded_id_csv, loaded_param_csv, _ = StarLoader.load_system_data(save_file_csv)
    assert len(loaded_id_csv) == len(id_df)
    assert len(loaded_param_csv) == len(param_df)


def test_add_galactic_population(mock_star_data):
    _, param_df = mock_star_data
    loader = StarLoader()
    new_df = loader.add_galactic_population(param_df, population="Thin disc", source="This work")

    assert "galactic_population" in new_df["name"].to_numpy()
    row = new_df[new_df["name"] == "galactic_population"].iloc[0]
    assert row["value"] == "Thin disc"
    assert row["source"] == "This work"


def test_add_uvw(mock_star_data):
    _, param_df = mock_star_data
    loader = StarLoader()
    df_with_uvw = loader.add_uvw(param_df, n_samples=1000, seed=42)

    assert "u_lsr" in df_with_uvw["name"].to_numpy()
    assert "v_lsr" in df_with_uvw["name"].to_numpy()
    assert "w_lsr" in df_with_uvw["name"].to_numpy()
    assert r"$U_\mathrm{LSR}$" in df_with_uvw["symbol"].to_numpy()


def test_static_methods_callable_without_instance(mock_star_data):
    _, param_df = mock_star_data
    df_pop = StarLoader.add_galactic_population(param_df, population="Thick disc")
    assert "Thick disc" in df_pop["value"].to_numpy()

    df_uvw = StarLoader.add_uvw(param_df, n_samples=500, seed=1)
    assert "u_lsr" in df_uvw["name"].to_numpy()

    sp = StarLoader.get_spectral_type(teff=5850)
    assert sp == r"G2\,V"
