"""
citations.py

Standardised citation provenance for exohelp.

Every function that implements a published relation is decorated with `@cites`,
and every table column derived from such a function carries the same canonical
key(s) in ``col.info.meta["references"]``. This module is the single place that
turns those keys into a :class:`Reference` (with ``doi``/``bibcode``/``adsurl``),
into BibTeX, or into a ``\\citet{}``/``\\citep{}`` string ready to paste into a
manuscript.

Getting the citations for a result
-----------------------------------
>>> import exohelp
>>> refs = exohelp.cite(exohelp.planet.transmission_spectroscopy_metric)
>>> refs[0].key
'Kempton2018'

Getting the citations for a whole analysis
-------------------------------------------
>>> from exohelp.star.activity import age_mamajek2008
>>> with exohelp.citation_tracker() as tracker:
...     _ = age_mamajek2008(-4.5)
>>> tracker.keys
['MamajekHillenbrand2008']
"""

from __future__ import annotations

import contextvars
import functools
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypeVar

from .data import get_default_bibtex

__all__ = [
    "REFERENCES",
    "CitationTracker",
    "Reference",
    "bibtex",
    "citation_tracker",
    "cite",
    "cite_keys",
    "citep",
    "cites",
    "citet",
    "resolve",
]

_NON_REFERENCE_TOKENS = {"this work", "nan", "none", "--", ""}


@dataclass(frozen=True)
class Reference:
    """A single citable paper, with enough structure to render as BibTeX or LaTeX.

    Parameters
    ----------
    key : str
        Canonical BibTeX key (``AuthorYear``), matching an entry in the bundled
        ``references.bib``.
    label : str
        Human-readable citation, e.g. ``"Kempton et al. (2018)"``.
    bibcode : str, optional
        ADS bibcode.
    doi : str, optional
        Bare DOI (no ``https://doi.org/`` prefix).
    note : str, optional
        Per-use detail such as ``"Eq. 9, full sample"``. Not part of identity —
        two :class:`Reference` objects with the same ``key`` are equal for
        de-duplication purposes regardless of ``note``.
    """

    key: str
    label: str
    bibcode: str | None = None
    doi: str | None = None
    note: str | None = None

    @property
    def adsurl(self) -> str | None:
        """ADS abstract page URL, or None if no bibcode is known."""
        if self.bibcode is None:
            return None
        return f"https://ui.adsabs.harvard.edu/abs/{self.bibcode}/abstract"

    @property
    def doi_url(self) -> str | None:
        """Resolvable DOI URL, or None if no DOI is known."""
        if self.doi is None:
            return None
        return f"https://doi.org/{self.doi}"

    def bibtex(self) -> str:
        """The bundled BibTeX entry for this reference."""
        return _bibtex_entry(self.key)

    def citet(self) -> str:
        r"""``\citet{key}`` for this reference."""
        return rf"\citet{{{self.key}}}"

    def citep(self) -> str:
        r"""``\citep{key}`` for this reference."""
        return rf"\citep{{{self.key}}}"

    def with_note(self, note: str) -> Reference:
        """Return a copy of this reference carrying a per-use ``note``."""
        return replace(self, note=note)


def _ref(key: str, label: str, *, bibcode: str | None = None, doi: str | None = None) -> Reference:
    return Reference(key=key, label=label, bibcode=bibcode, doi=doi)


REFERENCES: dict[str, Reference] = {
    r.key: r
    for r in [
        # --- catalogs / surveys ---
        _ref(
            "GaiaCollaboration2023",
            "Gaia Collaboration et al. (2023)",
            bibcode="2023A&A...674A...1G",
            doi="10.1051/0004-6361/202243940",
        ),
        _ref(
            "lindegren2021",
            "Gaia Collaboration et al. (2021)",
            bibcode="2021A&A...649A...9G",
            doi="10.1051/0004-6361/202039734",
        ),
        _ref("Hoeg2000", "H\u00f8g et al. (2000)", bibcode="2000A&A...355L..27H"),
        _ref(
            "Stassun2019",
            "Stassun et al. (2019)",
            bibcode="2019AJ....158..138S",
            doi="10.3847/1538-3881/ab3467",
        ),
        _ref(
            "Skrutskie2006",
            "Skrutskie et al. (2006)",
            bibcode="2006AJ....131.1163S",
            doi="10.1086/498708",
        ),
        _ref(
            "Wright2010",
            "Wright et al. (2010)",
            bibcode="2010AJ....140.1868W",
            doi="10.1088/0004-6256/140/6/1868",
        ),
        # --- atmospheric escape / photoevaporation (planet/escape.py) ---
        _ref(
            "Watson1981",
            "Watson, Donahue & Walker (1981)",
            bibcode="1981Icar...48..150W",
            doi="10.1016/0019-1035(81)90161-5",
        ),
        _ref(
            "Ribas2005",
            "Ribas et al. (2005)",
            bibcode="2005ApJ...622..680R",
            doi="10.1086/427977",
        ),
        _ref(
            "Valencia2006",
            "Valencia, O'Connell & Sasselov (2006)",
            bibcode="2006Icar..181..545V",
            doi="10.1016/j.icarus.2005.11.021",
        ),
        _ref(
            "Erkaev2007",
            "Erkaev et al. (2007)",
            bibcode="2007A&A...472..329E",
            doi="10.1051/0004-6361:20066929",
        ),
        _ref(
            "FortneyMarleyBarnes2007",
            "Fortney, Marley & Barnes (2007)",
            bibcode="2007ApJ...659.1661F",
            doi="10.1086/512120",
        ),
        _ref(
            "MurrayClay2009",
            "Murray-Clay, Chiang & Murray (2009)",
            bibcode="2009ApJ...693...23M",
            doi="10.1088/0004-637X/693/1/23",
        ),
        _ref(
            "Jackson2012",
            "Jackson, Davis & Wheatley (2012)",
            bibcode="2012MNRAS.424...11J",
            doi="10.1111/j.1365-2966.2012.21160.x",
        ),
        _ref(
            "OwenWu2013",
            "Owen & Wu (2013)",
            bibcode="2013ApJ...775..105O",
            doi="10.1088/0004-637X/775/2/105",
        ),
        _ref(
            "LopezFortney2013",
            "Lopez & Fortney (2013)",
            bibcode="2013ApJ...776....2L",
            doi="10.1088/0004-637X/776/1/2",
        ),
        _ref(
            "Salz2016",
            "Salz et al. (2016)",
            bibcode="2016A&A...586A..75S",
            doi="10.1051/0004-6361/201526104",
        ),
        _ref(
            "OwenWu2017",
            "Owen & Wu (2017)",
            bibcode="2017ApJ...847...29O",
            doi="10.3847/1538-4357/aa890a",
        ),
        _ref(
            "Ginzburg2018",
            "Ginzburg, Schlichting & Sari (2018)",
            bibcode="2018MNRAS.476..759G",
            doi="10.1093/mnras/sty290",
        ),
        _ref(
            "GuptaSchlichting2019",
            "Gupta & Schlichting (2019)",
            bibcode="2019MNRAS.487...24G",
            doi="10.1093/mnras/stz1230",
        ),
        _ref(
            "Johnstone2021",
            "Johnstone, Bartel & G\u00fcdel (2021)",
            bibcode="2021A&A...649A..96J",
            doi="10.1051/0004-6361/202038407",
        ),
        _ref(
            "Caldiroli2021",
            "Caldiroli et al. (2021)",
            bibcode="2021A&A...655A..30C",
            doi="10.1051/0004-6361/202141570",
        ),
        _ref(
            "Caldiroli2022",
            "Caldiroli et al. (2022)",
            bibcode="2022A&A...663A.122C",
            doi="10.1051/0004-6361/202142750",
        ),
        # --- stellar activity / rotation / age (star/activity.py) ---
        _ref(
            "Noyes1984",
            "Noyes et al. (1984)",
            bibcode="1984ApJ...279..763N",
            doi="10.1086/161945",
        ),
        _ref(
            "MamajekHillenbrand2008",
            "Mamajek & Hillenbrand (2008)",
            bibcode="2008ApJ...687.1264M",
            doi="10.1086/591785",
        ),
        _ref(
            "Mittag2018",
            "Mittag, Schmitt & Schr\u00f6der (2018)",
            bibcode="2018A&A...618A..48M",
            doi="10.1051/0004-6361/201833498",
        ),
        _ref(
            "SuarezMascareno2015",
            "Su\u00e1rez Mascare\u00f1o et al. (2015)",
            bibcode="2015MNRAS.452.2745S",
            doi="10.1093/mnras/stv1441",
        ),
        _ref(
            "Barnes2010",
            "Barnes (2010)",
            bibcode="2010ApJ...722..222B",
            doi="10.1088/0004-637X/722/1/222",
        ),
        # --- stellar spectroscopy (star/spectroscopy.py) ---
        _ref(
            "Bruntt2010",
            "Bruntt et al. (2010)",
            bibcode="2010MNRAS.405.1907B",
            doi="10.1111/j.1365-2966.2010.16575.x",
        ),
        _ref(
            "Doyle2014",
            "Doyle et al. (2014)",
            bibcode="2014MNRAS.444.3592D",
            doi="10.1093/mnras/stu1692",
        ),
        _ref(
            "Bensby2014",
            "Bensby, Feltzing & Oey (2014)",
            bibcode="2014A&A...562A..71B",
            doi="10.1051/0004-6361/201322631",
        ),
        _ref(
            "Santerne2015",
            "Santerne et al. (2015)",
            bibcode="2015MNRAS.451.2337S",
            doi="10.1093/mnras/stv1080",
        ),
        # --- planet properties / transit / spectroscopy / rv / tides ---
        _ref(
            "Kempton2018",
            "Kempton et al. (2018)",
            bibcode="2018PASP..130k4401K",
            doi="10.1088/1538-3873/aadf6f",
        ),
        _ref("Winn2010", "Winn (2010)", bibcode="2010exop.book...55W"),
        _ref(
            "deWitSeager2013",
            "de Wit & Seager (2013)",
            bibcode="2013Sci...342.1473D",
            doi="10.1126/science.1245450",
        ),
        _ref(
            "MandelAgol2002",
            "Mandel & Agol (2002)",
            bibcode="2002ApJ...580L.171M",
            doi="10.1086/345520",
        ),
        _ref(
            "HamiltonBurns1992",
            "Hamilton & Burns (1992)",
            bibcode="1992Icar...96...43H",
            doi="10.1016/0019-1035(92)90005-R",
        ),
        _ref("LovisFischer2010", "Lovis & Fischer (2010)", bibcode="2010exop.book...27L"),
        _ref(
            "KennedyKenyon2008",
            "Kennedy & Kenyon (2008)",
            bibcode="2008ApJ...673..502K",
            doi="10.1086/524130",
        ),
        _ref(
            "Quirrenbach2022",
            "Quirrenbach (2022)",
            bibcode="2022RNAAS...6...56Q",
            doi="10.3847/2515-5172/ac5f0d",
        ),
        _ref(
            "Jackson2008",
            "Jackson, Greenberg & Barnes (2008)",
            bibcode="2008ApJ...678.1396J",
            doi="10.1086/529187",
        ),
        _ref(
            "Jackson2009",
            "Jackson, Greenberg & Barnes (2009)",
            bibcode="2009ApJ...698.1357J",
            doi="10.1088/0004-637X/698/2/1357",
        ),
    ]
}

# Legacy/alternate tokens that resolve to a canonical key above but are not the
# key itself: survey names, author-year abbreviations, or bibcodes already in
# circulation (e.g. as `StarLoader` `source` values). Bibcodes from REFERENCES
# are added automatically below.
_EXTRA_ALIASES: dict[str, str] = {
    "2MASS": "Skrutskie2006",
    "Skr06": "Skrutskie2006",
    "ALLWISE": "Wright2010",
    "WISE": "Wright2010",
    "Cut13": "Wright2010",
    "Tycho-2": "Hoeg2000",
    "Tycho": "Hoeg2000",
    r"H\o g00": "Hoeg2000",
    "Hog00": "Hoeg2000",
    "TIC": "Stassun2019",
    "Sta19": "Stassun2019",
    "Gaia DR3": "GaiaCollaboration2023",
    r"\textit{Gaia} DR3": "GaiaCollaboration2023",
    "Gai23": "GaiaCollaboration2023",
    "Lin21": "lindegren2021",
}


def _normalize_bib_key(key: str) -> str:
    """Normalize a bibliography key by stripping LaTeX formatting and casing."""
    cleaned = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", str(key))
    cleaned = cleaned.replace(r"\_", "_").replace(r"\ ", " ").replace("\\", "")
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def _build_alias_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for ref in REFERENCES.values():
        index[_normalize_bib_key(ref.key)] = ref.key
        if ref.bibcode:
            index[_normalize_bib_key(ref.bibcode)] = ref.key
    for alias, key in _EXTRA_ALIASES.items():
        index[_normalize_bib_key(alias)] = key
    return index


_ALIAS_INDEX: dict[str, str] = _build_alias_index()


def resolve(token: str | Reference | None) -> Reference | None:
    """Resolve a canonical key, bibcode, survey/author alias, or DEFAULT_TABLEBIB-style
    token to its :class:`Reference`.

    Returns None for tokens that are not references at all — "This work", "nan",
    "none", "--", empty strings, and tokens that already contain a LaTeX
    ``\\ref``/``\\cite`` command.

    Examples
    --------
    >>> resolve("2MASS").key
    'Skrutskie2006'
    >>> resolve("2018PASP..130k4401K").key
    'Kempton2018'
    >>> resolve("This work") is None
    True
    """
    if isinstance(token, Reference):
        return token
    if not token:
        return None
    token = str(token)
    if token.strip().lower() in _NON_REFERENCE_TOKENS:
        return None
    if r"\ref" in token or r"\cite" in token:
        return None
    if token in REFERENCES:
        return REFERENCES[token]
    key = _ALIAS_INDEX.get(_normalize_bib_key(token))
    return REFERENCES[key] if key else None


def _refs_from(tokens: Iterable[Any]) -> Iterable[Reference]:
    for token in tokens:
        ref = token if isinstance(token, Reference) else resolve(token)
        if ref is not None:
            yield ref


def cite(obj: Any) -> list[Reference]:
    """Return the de-duplicated list of references behind a computed result.

    Accepts a `QTable`, a table column (or `Quantity`/`Row` with `.info.meta`),
    a `@cites`-decorated function, a `Reference`, a bare key/bibcode/alias
    string, or an iterable of any of the above. Order follows first appearance.
    """
    ordered_keys: list[str] = []
    seen: set[str] = set()

    def add_all(refs: Iterable[Reference]) -> None:
        for ref in refs:
            if ref.key not in seen:
                seen.add(ref.key)
                ordered_keys.append(ref.key)

    if isinstance(obj, Reference):
        add_all([obj])
    elif isinstance(obj, str):
        add_all(_refs_from([obj]))
    elif hasattr(obj, "colnames"):  # QTable / Table
        add_all(_refs_from((obj.meta or {}).get("references", ())))
        for name in obj.colnames:
            info = obj[name].info
            add_all(_refs_from((info.meta or {}).get("references", ())))
    elif hasattr(obj, "info") and hasattr(obj.info, "meta"):  # a single column/Quantity/Row
        add_all(_refs_from((obj.info.meta or {}).get("references", ())))
    elif callable(obj):
        add_all(_refs_from(getattr(obj, "references", ())))
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            add_all(cite(item))
    else:
        raise TypeError(f"exohelp.cite() cannot resolve citations for {type(obj)!r}")

    return [REFERENCES[k] for k in ordered_keys]


def cite_keys(obj: Any) -> list[str]:
    """Like `cite`, but returning bare canonical keys."""
    return [ref.key for ref in cite(obj)]


def citet(obj: Any) -> str:
    r"""``\citet{key1,key2,...}`` for every reference behind `obj`."""
    keys = cite_keys(obj)
    return rf"\citet{{{','.join(keys)}}}" if keys else ""


def citep(obj: Any) -> str:
    r"""``\citep{key1,key2,...}`` for every reference behind `obj`."""
    keys = cite_keys(obj)
    return rf"\citep{{{','.join(keys)}}}" if keys else ""


@functools.lru_cache(maxsize=1)
def _bibtex_entries() -> dict[str, str]:
    """Split the bundled `references.bib` into {key: raw entry text}."""
    text = get_default_bibtex()
    pattern = re.compile(r"^@\w+\{\s*([^,\s]+)\s*,", re.MULTILINE)
    matches = list(pattern.finditer(text))
    entries: dict[str, str] = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        entries[match.group(1)] = text[match.start() : end].strip()
    return entries


def _bibtex_entry(key: str) -> str:
    entries = _bibtex_entries()
    if key not in entries:
        raise KeyError(f"No bundled BibTeX entry for key {key!r} in references.bib")
    return entries[key]


def bibtex(obj: Any, path: str | Path | None = None) -> str:
    """BibTeX entries for exactly the references behind `obj` (not the whole file).

    If `path` is given, also writes the entries there.
    """
    text = "\n\n".join(_bibtex_entry(ref.key) for ref in cite(obj))
    if path is not None:
        Path(path).write_text(text + ("\n" if text else ""), encoding="utf-8")
    return text


F = TypeVar("F", bound=Callable[..., Any])

_tracker_stack: contextvars.ContextVar[tuple["CitationTracker", ...]] = contextvars.ContextVar(
    "exohelp_citation_tracker_stack", default=()
)


def cites(*keys: str, note: str | None = None) -> Callable[[F], F]:
    """Decorator recording the paper(s) a function implements.

    Sets ``func.references`` (a tuple of `Reference`) so `exohelp.cite(func)`
    works without calling it, and notifies any active `citation_tracker()`
    each time the function is actually called.

    Raises `KeyError` at import time if `keys` are not in `REFERENCES` — add
    the paper there first.
    """
    refs = tuple(REFERENCES[key].with_note(note) if note else REFERENCES[key] for key in keys)

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for tracker in _tracker_stack.get():
                tracker._record(refs)
            return func(*args, **kwargs)

        wrapper.references = refs  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


class CitationTracker:
    """Records every `@cites`-decorated function called inside a `with` block.

    Use via `citation_tracker()`, not by constructing directly.
    """

    def __init__(self) -> None:
        self._seen: dict[str, Reference] = {}
        self._token: contextvars.Token[tuple[CitationTracker, ...]] | None = None

    def _record(self, refs: Sequence[Reference]) -> None:
        for ref in refs:
            self._seen.setdefault(ref.key, ref)

    @property
    def references(self) -> list[Reference]:
        """References recorded so far, in order of first use."""
        return list(self._seen.values())

    @property
    def keys(self) -> list[str]:
        """Canonical keys recorded so far, in order of first use."""
        return list(self._seen.keys())

    def bibtex(self) -> str:
        """BibTeX entries for every reference recorded so far."""
        return "\n\n".join(_bibtex_entry(key) for key in self._seen)

    def write_bib(self, path: str | Path) -> Path:
        """Write the recorded BibTeX entries to `path`; returns the `Path`."""
        path = Path(path)
        text = self.bibtex()
        path.write_text(text + ("\n" if text else ""), encoding="utf-8")
        return path

    def citet(self) -> str:
        r"""``\citet{...}`` for everything recorded so far."""
        return citet(self.references)

    def citep(self) -> str:
        r"""``\citep{...}`` for everything recorded so far."""
        return citep(self.references)

    def __enter__(self) -> CitationTracker:
        self._token = _tracker_stack.set((*_tracker_stack.get(), self))
        return self

    def __exit__(self, *exc_info: Any) -> None:
        if self._token is not None:
            _tracker_stack.reset(self._token)
            self._token = None

    def __len__(self) -> int:
        return len(self._seen)

    def __repr__(self) -> str:
        return f"CitationTracker({list(self._seen)!r})"


def citation_tracker() -> CitationTracker:
    """Return a new `CitationTracker` context manager.

    Examples
    --------
    >>> import exohelp
    >>> from exohelp.star.activity import age_mamajek2008
    >>> with exohelp.citation_tracker() as tracker:
    ...     _ = age_mamajek2008(-4.5)
    >>> tracker.keys
    ['MamajekHillenbrand2008']
    """
    return CitationTracker()
