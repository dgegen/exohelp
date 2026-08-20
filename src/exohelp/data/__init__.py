from importlib import resources
from pathlib import Path


def get_default_bibtex_path() -> Path:
    """Return the traversable/path to the bundled references.bib file."""
    return resources.files("exohelp").joinpath("data", "references.bib")  # type: ignore[return-value]


def get_default_bibtex() -> str:
    """Return the default BibTeX entries for standard catalog references."""
    return get_default_bibtex_path().read_text(encoding="utf-8")


__all__ = [
    "get_default_bibtex",
    "get_default_bibtex_path",
]
