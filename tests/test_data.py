import exohelp
from exohelp.data import get_default_bibtex, get_default_bibtex_path


def test_get_default_bibtex_path():
    path = get_default_bibtex_path()
    assert path.name == "references.bib"
    assert path.is_file()


def test_get_default_bibtex_content():
    content = get_default_bibtex()
    assert "@article{GaiaCollaboration2023" in content
    assert "@article{Skrutskie2006" in content
    assert "@article{Stassun2019" in content


def test_exohelp_data_submodule_accessible():
    assert hasattr(exohelp, "data")
    assert callable(exohelp.data.get_default_bibtex)
