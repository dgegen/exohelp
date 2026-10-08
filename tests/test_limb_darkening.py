import os

import astropy.units as u
import numpy as np
import pandas as pd
import pytest

import exohelp
from exohelp.star.limb_darkening import (
    claret_quadratic_coefficients,
    limb_darkening_prior,
    q_to_u,
    sample_limb_darkening,
    u_to_q,
)

TEFF = np.array([5000.0, 5500.0, 6000.0])
LOGG = np.array([4.0, 4.5])
FEH = np.array([-0.5, 0.0, 0.5])


def _u1(teff, logg, feh):
    return 0.5 - 1e-4 * (teff - 5000) + 0.05 * (logg - 4) + 0.02 * feh


def _u2(teff, logg, feh):
    return 0.2 + 2e-5 * (teff - 5000) - 0.02 * (logg - 4) - 0.01 * feh


def _write_grid(cache_dir, band, model, rows):
    df = pd.DataFrame(rows, columns=["teff", "logg", "feh", "vturb"])
    df["u1"] = _u1(df.teff, df.logg, df.feh)
    df["u2"] = _u2(df.teff, df.logg, df.feh)
    df.to_csv(cache_dir / f"claret_{band.lower()}_{model.lower()}.csv", index=False)


@pytest.fixture
def cache_dir(tmp_path):
    """Synthetic grids: ATLAS with a hole at (6000 K, 4.0, +0.5), PHOENIX at solar only."""
    atlas = [
        (t, g, z, 2.0) for t in TEFF for g in LOGG for z in FEH if (t, g, z) != (6000.0, 4.0, 0.5)
    ]
    atlas += [(t, g, 0.0, 0.0) for t in TEFF for g in LOGG]
    _write_grid(tmp_path, "CHEOPS", "ATLAS", atlas)
    _write_grid(tmp_path, "TESS", "PHOENIX", [(t, g, 0.0, 2.0) for t in TEFF for g in LOGG])
    return tmp_path


def test_exact_at_nodes(cache_dir):
    u1, u2 = claret_quadratic_coefficients(5500, 4.5, 0.5, band="CHEOPS", cache_dir=cache_dir)
    assert u1 == pytest.approx(_u1(5500, 4.5, 0.5))
    assert u2 == pytest.approx(_u2(5500, 4.5, 0.5))


def test_linear_between_nodes_and_units(cache_dir):
    # The synthetic grid is linear, so multilinear interpolation must reproduce it exactly.
    u1, u2 = claret_quadratic_coefficients(
        5250 * u.K, 4.25 * u.dex, -0.25, band="CHEOPS", cache_dir=cache_dir
    )
    assert u1 == pytest.approx(_u1(5250, 4.25, -0.25))
    assert u2 == pytest.approx(_u2(5250, 4.25, -0.25))


def test_broadcast_shape(cache_dir):
    u1, _ = claret_quadratic_coefficients(
        [[5100, 5200]], [4.1, 4.2], band="CHEOPS", cache_dir=cache_dir
    )
    assert u1.shape == (1, 2)


def test_masked_outside_grid_and_next_to_hole(cache_dir):
    u1, u2 = claret_quadratic_coefficients(
        [4900, 5750, 5750, 6000], [4.5, 4.25, 4.75, 4.5], [0.0, 0.25, 0.0, 0.5],
        band="CHEOPS", cache_dir=cache_dir,
    )  # fmt: skip
    # Outside in Teff; cell touching the hole; outside in logg; node next to the hole.
    assert u1.mask.tolist() == [True, True, True, False]
    assert u2.mask.tolist() == u1.mask.tolist()


def test_nan_input_raises(cache_dir):
    with pytest.raises(ValueError, match="finite"):
        claret_quadratic_coefficients(np.nan, 4.5, band="CHEOPS", cache_dir=cache_dir)
    with pytest.raises(ValueError, match="finite"):
        claret_quadratic_coefficients(5500, 4.5, np.nan, band="CHEOPS", cache_dir=cache_dir)


def test_metallicity_only_where_tabulated(cache_dir):
    # PHOENIX is solar-only: feh=None (or 0) works, anything else raises.
    claret_quadratic_coefficients(5500, 4.5, band="TESS", model="PHOENIX", cache_dir=cache_dir)
    with pytest.raises(ValueError, match="feh"):
        claret_quadratic_coefficients(
            5500, 4.5, 0.1, band="TESS", model="PHOENIX", cache_dir=cache_dir
        )
    # ATLAS at vturb != 2 is solar-only.
    u1, _ = claret_quadratic_coefficients(5500, 4.5, band="CHEOPS", vturb=0, cache_dir=cache_dir)
    assert u1 == pytest.approx(_u1(5500, 4.5, 0.0))
    with pytest.raises(ValueError, match="feh"):
        claret_quadratic_coefficients(5500, 4.5, 0.1, band="CHEOPS", vturb=0, cache_dir=cache_dir)


def test_untabulated_vturb_raises(cache_dir):
    with pytest.raises(ValueError, match="not tabulated"):
        claret_quadratic_coefficients(5500, 4.5, band="CHEOPS", vturb=1.5, cache_dir=cache_dir)


def test_unknown_band_raises(cache_dir):
    with pytest.raises(ValueError, match="Unsupported"):
        claret_quadratic_coefficients(5500, 4.5, band="Kepler", cache_dir=cache_dir)


def test_u_q_round_trip():
    u1, u2 = np.array([0.4, 0.1, 0.6]), np.array([0.2, 0.3, -0.1])
    q1, q2 = u_to_q(u1, u2)
    np.testing.assert_allclose(q_to_u(q1, q2), (u1, u2))


def test_sample_table(cache_dir):
    # Samples stay inside the hole-free cell 5000-5500 K, 4.0-4.5, -0.5-0.
    table = sample_limb_darkening(
        5250, 50, 4.25, 0.05, -0.25, 0.05, band="CHEOPS", n_samples=500, seed=1, cache_dir=cache_dir
    )
    assert table.colnames == ["teff", "logg", "feh", "u1", "u2", "q1", "q2"]
    assert table.meta["masked_fraction"] == 0.0
    assert exohelp.cite_keys(table) == ["Claret2021", "Kipping2013"]
    assert table["u1"].info.meta["references"] == ["Claret2021"]


def test_prior(cache_dir):
    prior = limb_darkening_prior(
        5250, 50, 4.25, 0.05, -0.25, 0.05, band="CHEOPS", n_samples=2000, cache_dir=cache_dir
    )
    q1, q2 = u_to_q(_u1(5250, 4.25, -0.25), _u2(5250, 4.25, -0.25))
    assert prior["q1"][0] == pytest.approx(q1, abs=2e-3)
    assert prior["q2"][0] == pytest.approx(q2, abs=2e-3)
    assert 0 < prior["q1"][1] < 0.05

    floored = limb_darkening_prior(
        5250,
        50,
        4.25,
        0.05,
        -0.25,
        0.05,
        band="CHEOPS",
        min_sigma=0.1,
        n_samples=2000,
        cache_dir=cache_dir,
    )
    assert floored["q1"][1] == 0.1
    assert floored["q2"][1] == 0.1

    u_prior = limb_darkening_prior(
        5250,
        50,
        4.25,
        0.05,
        -0.25,
        0.05,
        band="CHEOPS",
        parametrization="u",
        n_samples=2000,
        cache_dir=cache_dir,
    )
    assert set(u_prior) == {"u1", "u2"}


def test_prior_warns_at_grid_edge(cache_dir):
    with pytest.warns(UserWarning, match="outside"):
        limb_darkening_prior(
            5000, 100, 4.3, 0.05, band="CHEOPS", n_samples=500, cache_dir=cache_dir
        )
    with pytest.raises(ValueError, match="All samples"):
        limb_darkening_prior(8000, 10, 4.3, 0.05, band="CHEOPS", n_samples=100, cache_dir=cache_dir)


def test_citation_tracker_records_only_the_band_used(cache_dir):
    with exohelp.citation_tracker() as tracker:
        claret_quadratic_coefficients(5500, 4.5, band="TESS", model="PHOENIX", cache_dir=cache_dir)
    assert tracker.keys == ["Claret2017"]
    assert exohelp.cite_keys(claret_quadratic_coefficients) == ["Claret2017", "Claret2021"]


def test_citation_tracker_records_prior_provenance(cache_dir):
    with exohelp.citation_tracker() as tracker:
        limb_darkening_prior(
            5250, 50, 4.25, 0.05, band="CHEOPS", n_samples=100, cache_dir=cache_dir
        )
    assert tracker.keys == ["Claret2021", "Kipping2013"]


@pytest.mark.skipif(
    not os.environ.get("EXOHELP_NETWORK_TESTS"),
    reason="downloads the Claret tables from VizieR; set EXOHELP_NETWORK_TESTS=1",
)
@pytest.mark.parametrize("band", ["CHEOPS", "TESS"])
@pytest.mark.parametrize("model", ["ATLAS", "PHOENIX"])
def test_real_tables(band, model, tmp_path):
    table = exohelp.star.load_claret_table(band, model, cache_dir=tmp_path)
    assert not table.duplicated(["teff", "logg", "feh", "vturb"]).any()
    # A Sun-like star is inside every grid and has typical coefficients.
    u1, u2 = claret_quadratic_coefficients(5770, 4.44, band=band, model=model, cache_dir=tmp_path)
    assert 0.3 < u1 < 0.6
    assert 0.1 < u2 < 0.35


def test_recommended_model():
    from exohelp.star.limb_darkening import recommended_model

    assert recommended_model(3200) == "PHOENIX"
    assert recommended_model(3999.0 * u.K) == "PHOENIX"
    assert recommended_model(4000) == "ATLAS"
    assert recommended_model(5700) == "ATLAS"
    with pytest.raises(ValueError):
        recommended_model([3000, 5000])
    with pytest.raises(ValueError, match="finite"):
        recommended_model(np.nan)


def test_auto_model_selects_atlas_for_fgk(cache_dir):
    table = sample_limb_darkening(
        5250, 50, 4.25, 0.05, -0.25, 0.05,
        band="CHEOPS", model="auto", n_samples=200, seed=1, cache_dir=cache_dir,
    )  # fmt: skip
    assert table.meta["model"] == "ATLAS"
    assert np.ptp(table["feh"]) > 0


def test_auto_model_selects_phoenix_and_drops_feh(cache_dir, monkeypatch):
    import exohelp.star.limb_darkening as ld

    # The synthetic PHOENIX grid spans 5000-6000 K; move the threshold into it.
    monkeypatch.setattr(ld, "PHOENIX_TEFF_THRESHOLD", 5600.0)
    with pytest.warns(UserWarning, match="ignored"):
        table = sample_limb_darkening(
            5250, 50, 4.25, 0.05, 0.3, 0.05,
            band="TESS", model="auto", n_samples=200, seed=1, cache_dir=cache_dir,
        )  # fmt: skip
    assert table.meta["model"] == "PHOENIX"
    assert np.all(table["feh"] == 0.0)
    prior = limb_darkening_prior(
        5250, 50, 4.25, 0.05, band="TESS", model="auto", n_samples=200, cache_dir=cache_dir
    )
    assert set(prior) == {"q1", "q2"}


def test_auto_model_not_accepted_pointwise(cache_dir):
    with pytest.raises(ValueError, match="Unsupported"):
        claret_quadratic_coefficients(5500, 4.5, band="CHEOPS", model="auto", cache_dir=cache_dir)
