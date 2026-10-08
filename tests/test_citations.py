import re

import astropy.units as u
import pytest

import exohelp
from exohelp.citations import (
    REFERENCES,
    Reference,
    bibtex,
    cite,
    cite_keys,
    citation_tracker,
    citep,
    citet,
    resolve,
)
from exohelp.data import get_default_bibtex

_BIB_KEY_PATTERN = re.compile(r"^@\w+\{\s*([^,\s]+)\s*,", re.MULTILINE)


def _bundled_bib_keys() -> set[str]:
    return set(_BIB_KEY_PATTERN.findall(get_default_bibtex()))


# --- 1. Registry <-> references.bib consistency -----------------------------------


def test_every_reference_key_is_in_the_bundled_bib():
    bib_keys = _bundled_bib_keys()
    missing = sorted(set(REFERENCES) - bib_keys)
    assert not missing, f"REFERENCES keys missing from references.bib: {missing}"


def test_every_bundled_bib_key_is_in_the_registry():
    bib_keys = _bundled_bib_keys()
    missing = sorted(bib_keys - set(REFERENCES))
    assert not missing, f"references.bib keys missing from REFERENCES: {missing}"


def test_every_reference_bibtex_entry_is_retrievable():
    for ref in REFERENCES.values():
        entry = ref.bibtex()
        assert entry.startswith("@")
        assert ref.key in entry


# --- 2. Decorated functions and QTable columns resolve -----------------------------


def test_decorated_leaf_functions_have_resolvable_references():
    import exohelp.planet.astrometry as astrometry
    import exohelp.planet.escape as escape
    import exohelp.planet.properties as properties
    import exohelp.planet.rv as rv
    import exohelp.planet.spectroscopy as spectroscopy
    import exohelp.planet.tides as tides
    import exohelp.planet.transit as transit
    import exohelp.star.activity as activity
    import exohelp.star.limb_darkening as limb_darkening
    import exohelp.star.properties as star_properties
    import exohelp.star.spectroscopy as star_spectroscopy

    modules = [
        astrometry,
        escape,
        properties,
        rv,
        spectroscopy,
        tides,
        transit,
        activity,
        limb_darkening,
        star_properties,
        star_spectroscopy,
    ]
    decorated_count = 0
    for module in modules:
        for name in dir(module):
            obj = getattr(module, name)
            refs = getattr(obj, "references", None)
            if not refs:
                continue
            decorated_count += 1
            for ref in refs:
                assert ref.key in REFERENCES, (
                    f"{module.__name__}.{name} cites unknown key {ref.key!r}"
                )

    # Sanity check: the sweep actually found decorated functions.
    assert decorated_count > 20


@pytest.mark.parametrize(
    "table",
    [
        exohelp.planet.transit_quantities(365.25),
        exohelp.planet.derived_planet_quantities(
            period=3.0, r_planet=2.0, r_star=0.8, m_star=0.8, teff_star=4500, m_planet=8.0
        ),
        exohelp.star.sample_v_mic_and_v_mac(
            teff=5700.0, teff_err=100.0, logg=4.4, logg_err=0.1, n_samples=50, seed=42
        ),
        exohelp.star.sample_uvw_lsr(
            ra=10.0,
            ra_err=0.1,
            dec=45.0,
            dec_err=0.1,
            distance=10.0,
            distance_err=1.0,
            pm_ra_cosdec=5.0,
            pm_ra_cosdec_err=0.1,
            pm_dec=-5.0,
            pm_dec_err=0.1,
            radial_velocity=1.0,
            radial_velocity_err=0.1,
            n_samples=50,
            seed=42,
        ),
        exohelp.star.sample_rotation_period_and_age(
            log_rhk=-5.0,
            log_rhk_err=0.1,
            mag_b=10.0,
            mag_b_err=0.05,
            mag_v=9.5,
            mag_v_err=0.05,
            n_samples=50,
            seed=42,
        ),
        exohelp.star.derive_stellar_parameters(
            teff=5700.0,
            teff_err=100.0,
            logg=4.4,
            logg_err=0.1,
            mass=1.0,
            mass_err=0.05,
            radius=1.0,
            radius_err=0.05,
            vsini=2.0,
            vsini_err=0.5,
            n_samples=50,
            seed=42,
        ),
    ],
    ids=[
        "transit_quantities",
        "derived_planet_quantities",
        "sample_v_mic_and_v_mac",
        "sample_uvw_lsr",
        "sample_rotation_period_and_age",
        "derive_stellar_parameters",
    ],
)
def test_qtable_column_references_resolve(table):
    for name in table.colnames:
        for key in (table[name].info.meta or {}).get("references", ()):
            assert resolve(key) is not None, f"column {name!r} cites unresolvable key {key!r}"


def test_escape_qtable_references_resolve():
    from exohelp.planet.escape import (
        StellarXUVTrack,
        photoevaporation_evolution,
        photoevaporation_static,
    )

    track = StellarXUVTrack.nominal_solar()
    static_table = photoevaporation_static(
        m_planet=5.0, r_planet=2.0, semi_major_axis=0.05, track=track, age=3.0 * u.Gyr
    )
    evolution_table = photoevaporation_evolution(
        m_planet_init=5.0 * u.M_earth,
        r_planet_init=2.0 * u.R_earth,
        semi_major_axis=0.05 * u.AU,
        track=track,
        age=1.0 * u.Gyr,
    )
    for table in (static_table, evolution_table):
        for key in (table.meta or {}).get("references", ()):
            assert resolve(key) is not None


# --- 3. cite()/bibtex() introspection ----------------------------------------------


def test_cite_on_qtable_matches_column_meta():
    table = exohelp.star.sample_v_mic_and_v_mac(
        teff=5700.0, teff_err=100.0, logg=4.4, logg_err=0.1, n_samples=50, seed=42
    )
    refs = cite(table)
    keys = {r.key for r in refs}
    assert keys == {"Bruntt2010", "Doyle2014"}


def test_bibtex_returns_only_referenced_entries():
    table = exohelp.star.sample_v_mic_and_v_mac(
        teff=5700.0, teff_err=100.0, logg=4.4, logg_err=0.1, n_samples=50, seed=42
    )
    text = bibtex(table)
    assert "@ARTICLE{Bruntt2010" in text
    assert "@ARTICLE{Doyle2014" in text
    # A paper this table does not use must not appear.
    assert "Kempton2018" not in text
    assert "@ARTICLE{Watson1981" not in text


def test_bibtex_writes_to_path(tmp_path):
    out = tmp_path / "refs.bib"
    text = bibtex(REFERENCES["Kempton2018"], path=out)
    assert out.exists()
    assert out.read_text(encoding="utf-8").strip() == text.strip()


def test_cite_on_function_uses_static_references_attribute():
    from exohelp.planet.spectroscopy import transmission_spectroscopy_metric

    refs = cite(transmission_spectroscopy_metric)
    assert [r.key for r in refs] == ["Kempton2018"]


def test_citet_and_citep_formatting():
    assert citet(REFERENCES["Kempton2018"]) == r"\citet{Kempton2018}"
    assert citep(REFERENCES["Kempton2018"]) == r"\citep{Kempton2018}"
    assert citet([]) == ""


def test_cite_keys_preserves_order_and_dedupes():
    keys = cite_keys(["Kempton2018", "Winn2010", "Kempton2018"])
    assert keys == ["Kempton2018", "Winn2010"]


# --- 4. CitationTracker --------------------------------------------------------------


def test_tracker_records_across_a_multi_function_analysis():
    from exohelp.star.activity import age_mamajek2008, tau_c_noyes1984

    with citation_tracker() as tracker:
        age_mamajek2008(-4.5)
        tau_c_noyes1984(0.65)

    assert tracker.keys == ["MamajekHillenbrand2008", "Noyes1984"]
    assert len(tracker) == 2


def test_tracker_deduplicates_repeated_calls():
    from exohelp.star.activity import age_mamajek2008

    with citation_tracker() as tracker:
        age_mamajek2008(-4.5)
        age_mamajek2008(-4.7)

    assert tracker.keys == ["MamajekHillenbrand2008"]


def test_tracker_nests_and_both_levels_record():
    from exohelp.star.activity import age_mamajek2008, tau_c_noyes1984

    with citation_tracker() as outer:
        age_mamajek2008(-4.5)
        with citation_tracker() as inner:
            tau_c_noyes1984(0.65)
        assert inner.keys == ["Noyes1984"]

    assert set(outer.keys) == {"MamajekHillenbrand2008", "Noyes1984"}


def test_tracker_empty_outside_context():
    from exohelp.star.activity import age_mamajek2008

    tracker = citation_tracker()
    age_mamajek2008(-4.5)  # not inside the `with` block
    assert len(tracker) == 0


def test_tracker_write_bib(tmp_path):
    from exohelp.star.activity import age_mamajek2008

    with citation_tracker() as tracker:
        age_mamajek2008(-4.5)

    out = tracker.write_bib(tmp_path / "refs.bib")
    assert "@ARTICLE{MamajekHillenbrand2008" in out.read_text(encoding="utf-8")


# --- 5. Alias resolution -------------------------------------------------------------


@pytest.mark.parametrize(
    "token",
    ["2MASS", "Skr06", "ALLWISE", "WISE", "Cut13", "Tycho-2", "Tycho", "TIC", "Sta19"],
)
def test_survey_and_author_aliases_resolve(token):
    assert resolve(token) is not None


def test_bibcode_resolves_to_the_same_reference_as_its_key():
    assert resolve("2018PASP..130k4401K").key == "Kempton2018"
    assert resolve("Kempton2018").key == "Kempton2018"


@pytest.mark.parametrize("token", ["This work", "this work", "nan", "None", "--", "", None])
def test_non_reference_tokens_resolve_to_none(token):
    assert resolve(token) is None


@pytest.mark.parametrize("token", [r"\citet{GaiaCollaboration2023}", r"\ref{tab:foo}"])
def test_tokens_already_containing_latex_commands_resolve_to_none(token):
    assert resolve(token) is None


def test_resolve_passthrough_for_reference_instance():
    ref = REFERENCES["Kempton2018"]
    assert resolve(ref) is ref


def test_reference_urls():
    ref = REFERENCES["Kempton2018"]
    assert ref.adsurl == "https://ui.adsabs.harvard.edu/abs/2018PASP..130k4401K/abstract"
    assert ref.doi_url == "https://doi.org/10.1088/1538-3873/aadf6f"

    book_ref = REFERENCES["Winn2010"]
    assert book_ref.doi is None
    assert book_ref.doi_url is None


def test_with_note_preserves_identity_for_dedup():
    base = REFERENCES["Kempton2018"]
    noted = base.with_note("Eq. 1")
    assert noted.key == base.key
    assert noted.note == "Eq. 1"
    assert base.note is None


def test_unknown_cites_key_raises_at_decoration_time():
    from exohelp.citations import cites

    with pytest.raises(KeyError):

        @cites("NotARealKey2099")
        def _dummy():
            pass


def test_reference_is_a_dataclass_instance():
    assert isinstance(REFERENCES["Kempton2018"], Reference)
