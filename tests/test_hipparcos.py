from unittest.mock import patch

from astropy.table import Table

from exohelp.archive.hipparcos import (
    HGCARecord,
    HipparcosStatus,
    check_hipparcos_gaia_combinable,
    extract_hip_id,
    in_hipparcos_by_name,
    query_hgca,
)
from exohelp.citations import cite_keys


def test_extract_hip_id():
    assert extract_hip_id("HIP 27989") == 27989
    assert extract_hip_id("hip 27989") == 27989
    assert extract_hip_id("HIP27989") == 27989
    assert extract_hip_id("27989") == 27989
    assert extract_hip_id(27989) == 27989
    assert extract_hip_id(-1) is None
    assert extract_hip_id("HD 81466") is None
    assert extract_hip_id("Betelgeuse") is None


def test_in_hipparcos_by_name_fast_path():
    observed, hip_id = in_hipparcos_by_name("HIP 27989")
    assert observed is True
    assert hip_id == "HIP 27989"

    observed, hip_id = in_hipparcos_by_name("hip 1234")
    assert observed is True
    assert hip_id == "HIP 1234"


def test_in_hipparcos_by_name_simbad_mock_success_uppercase_col():
    mock_tbl = Table({"ID": ["* alf Ori", "HD 39801", "HIP 27989", "HR 2061"]})
    with patch("astroquery.simbad.Simbad.query_objectids", return_value=mock_tbl):
        observed, hip_id = in_hipparcos_by_name("Betelgeuse")
        assert observed is True
        assert hip_id == "HIP 27989"


def test_in_hipparcos_by_name_simbad_mock_success_lowercase_col():
    mock_tbl = Table({"id": ["HD 81466", "TIC 334632624", "HIP 46193"]})
    with patch("astroquery.simbad.Simbad.query_objectids", return_value=mock_tbl):
        observed, hip_id = in_hipparcos_by_name("HD 81466")
        assert observed is True
        assert hip_id == "HIP 46193"


def test_in_hipparcos_by_name_simbad_mock_no_hip():
    mock_tbl = Table({"id": ["TOI-9999", "TIC 999999999"]})
    with patch("astroquery.simbad.Simbad.query_objectids", return_value=mock_tbl):
        observed, hip_id = in_hipparcos_by_name("TOI-9999")
        assert observed is False
        assert hip_id is None


def test_in_hipparcos_by_name_simbad_mock_none_or_empty():
    with patch("astroquery.simbad.Simbad.query_objectids", return_value=None):
        observed, hip_id = in_hipparcos_by_name("NonexistentStar")
        assert observed is False
        assert hip_id is None

    with patch("astroquery.simbad.Simbad.query_objectids", return_value=Table({"id": []})):
        observed, hip_id = in_hipparcos_by_name("EmptyStar")
        assert observed is False
        assert hip_id is None


def test_in_hipparcos_by_name_simbad_exception():
    with patch("astroquery.simbad.Simbad.query_objectids", side_effect=ConnectionError("Timeout")):
        observed, hip_id = in_hipparcos_by_name("TimeoutStar")
        assert observed is False
        assert hip_id is None


def test_query_hgca_invalid_id():
    assert query_hgca("invalid") is None


def test_query_hgca_success():
    mock_tbl = Table(
        {
            "HIP": [27989],
            "gaia_source_id": [3224428429244456448],
            "chisq": [25.4],
            "pmRA_hip": [27.3],
            "e_pmRA_hip": [1.0],
            "pmDE_hip": [10.8],
            "e_pmDE_hip": [0.8],
            "pmRA_gaia": [26.1],
            "e_pmRA_gaia": [0.2],
            "pmDE_gaia": [9.9],
            "e_pmDE_gaia": [0.2],
            "pmRA_hg": [26.5],
            "e_pmRA_hg": [0.05],
            "pmDE_hg": [10.2],
            "e_pmDE_hg": [0.05],
            "epRA_gaia": [2016.0],
            "epDE_gaia": [2016.0],
        }
    )

    with patch("astroquery.vizier.VizierClass.query_constraints", return_value=[mock_tbl]):
        record = query_hgca(27989)
        assert isinstance(record, HGCARecord)
        assert record.hip_number == 27989
        assert record.gaia_source_id == 3224428429244456448
        assert record.chisq == 25.4
        assert record.significance_sigma is not None
        assert record.significance_sigma > 4.0
        assert record.pmra_hip == 27.3
        assert record.pmra_gaia == 26.1
        assert record.pmra_hg == 26.5


def test_query_hgca_not_found():
    with patch("astroquery.vizier.VizierClass.query_constraints", return_value=[]):
        record = query_hgca(999999)
        assert record is None


def test_query_hgca_exception():
    with patch(
        "astroquery.vizier.VizierClass.query_constraints", side_effect=RuntimeError("Vizier error")
    ):
        record = query_hgca(27989)
        assert record is None


def test_check_hipparcos_gaia_combinable_not_in_hipparcos():
    with patch("exohelp.archive.hipparcos.in_hipparcos_by_name", return_value=(False, None)):
        status = check_hipparcos_gaia_combinable("FaintTarget")
        assert isinstance(status, HipparcosStatus)
        assert status.in_hipparcos is False
        assert status.combinable is False
        assert status.in_hgca is False


def test_check_hipparcos_gaia_combinable_skip_hgca():
    with patch("exohelp.archive.hipparcos.in_hipparcos_by_name", return_value=(True, "HIP 46193")):
        status = check_hipparcos_gaia_combinable("HD 81466", check_hgca=False)
        assert status.in_hipparcos is True
        assert status.hip_id == "HIP 46193"
        assert status.hip_number == 46193
        assert status.combinable is True
        assert status.in_hgca is False


def test_check_hipparcos_gaia_combinable_in_hgca():
    fake_record = HGCARecord(
        hip_number=46193,
        gaia_source_id=5683148692262391808,
        chisq=12.5,
        significance_sigma=3.2,
    )
    with (
        patch("exohelp.archive.hipparcos.in_hipparcos_by_name", return_value=(True, "HIP 46193")),
        patch("exohelp.archive.hipparcos.query_hgca", return_value=fake_record),
    ):
        status = check_hipparcos_gaia_combinable("HD 81466", check_hgca=True)
        assert status.in_hipparcos is True
        assert status.in_hgca is True
        assert status.combinable is True
        assert status.gaia_source_id == 5683148692262391808
        assert status.hgca_chisq == 12.5
        assert status.pm_anomaly_significance == 3.2
        assert status.hgca_record == fake_record


def test_check_hipparcos_gaia_combinable_not_in_hgca():
    with (
        patch("exohelp.archive.hipparcos.in_hipparcos_by_name", return_value=(True, "HIP 1")),
        patch("exohelp.archive.hipparcos.query_hgca", return_value=None),
    ):
        status = check_hipparcos_gaia_combinable("HIP 1", check_hgca=True)
        assert status.in_hipparcos is True
        assert status.in_hgca is False
        assert status.combinable is False


def test_hipparcos_citations_registered():
    assert "Perryman1997" in cite_keys(in_hipparcos_by_name)
    assert "Brandt2021" in cite_keys(query_hgca)
    assert "Perryman1997" in cite_keys(check_hipparcos_gaia_combinable)
    assert "Brandt2021" in cite_keys(check_hipparcos_gaia_combinable)
