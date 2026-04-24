# exohelp

A Python library of tools for exoplanetary and stellar astrophysics. Provides vectorized, unit-aware functions built on `astropy` and `numpy`.

## Installation

```bash
pip install exohelp
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv add exohelp
```

## Quick Start

```python
import exohelp as eh

# Kepler's third law
period = eh.keplers_third_law(semi_major_axis=1.0, mass=1.0)  # → days

# Transit geometry for a planet
from exohelp.planet.summary import derived_planet_quantities
qtable = derived_planet_quantities(
    period=3.5,       # days
    r_planet=2.0,     # R_earth
    r_star=1.0,       # R_sun
    m_star=1.0,       # M_sun
    teff_star=5778,   # K
    m_planet=10.0,    # M_earth
    j_mag=8.5,
)
print(qtable)
```

All public functions accept plain floats (in canonical units), numpy arrays, or `astropy.Quantity` objects.

## Modules

### Body properties

```python
from exohelp.body import bulk_density, surface_gravity, log_surface_gravity

bulk_density(mass, radius)          # M_earth, R_earth → g/cm³
surface_gravity(mass, radius)       # M_earth, R_earth → m/s²
log_surface_gravity(mass, radius)   # M_earth, R_earth → dex (cgs)
```

### Kepler's third law

```python
from exohelp import keplers_third_law

# Solve for any one parameter by passing it as None (default)
period = keplers_third_law(semi_major_axis=1.0, mass=1.0)       # → days
a      = keplers_third_law(period=365.25, mass=1.0)             # → AU
mass   = keplers_third_law(period=365.25, semi_major_axis=1.0)  # → M_sun
```

Default units: period in days, semi-major axis in AU, mass in M_sun.

### Planet

#### Transit geometry

```python
from exohelp.planet.transit import (
    impact_parameter, orbital_inclination, transit_depth,
    transit_duration_total, transit_duration_flat, transit_duration_ingress,
    a_over_r_star, geometric_transit_probability, transit_quantities,
)
```

`transit_quantities(period, r_planet, r_star, m_star, b, eccentricity, omega)` returns a `QTable` with all geometry columns at once.

#### Planetary properties

```python
from exohelp.planet.properties import (
    insolation_flux,          # L_sun, AU → S/S⊕
    equilibrium_temperature,  # K, AU → K
    hill_sphere_radius,       # AU, M_earth, M_sun → AU
)
```

#### Radial velocity

```python
from exohelp.planet.rv import planet_mass_from_rv, rv_semi_amplitude

m_planet = planet_mass_from_rv(rv_semi_amplitude=5.0, period=10.0)  # → M_earth
K        = rv_semi_amplitude(m_planet=10.0, period=10.0)            # → m/s
```

Uses fixed-point iteration on the exact Keplerian relation (no small-planet approximation).

#### Spectroscopy metrics

```python
from exohelp.planet.spectroscopy import (
    scale_height,                  # K, m/s², amu → km
    transmission_signal_size,      # km, R_earth, R_sun → ppm
    transmission_spectroscopy_metric,  # TSM (Kempton et al. 2018)
    emission_spectroscopy_metric,      # ESM (Kempton et al. 2018)
)
```

#### Summary table

```python
from exohelp.planet.summary import derived_planet_quantities
```

Returns a `QTable` with transit geometry, RV, bulk density, surface gravity, scale height, TSM, ESM, and more — depending on which optional inputs are provided.

### Star

#### Activity and age

```python
from exohelp.star.activity import sample_rotation_period_and_age

result = sample_rotation_period_and_age(
    log_rhk=-4.5, log_rhk_err=0.05,
    mag_b=6.0,    mag_b_err=0.01,
    mag_v=5.5,    mag_v_err=0.01,
    n_samples=10_000,
)
```

Returns a `QTable` with sampled rotation periods (`prot_noyes`, `prot_mamajek`) and ages (`age_mamajek_gyro`, `age_mamajek_chromo`) based on Noyes (1984) and Mamajek & Hillenbrand (2008).

#### Spectroscopic velocities

```python
from exohelp.star.spectroscopy import sample_v_mic_and_v_mac

result = sample_v_mic_and_v_mac(
    teff=5778, teff_err=50,
    logg=4.44, logg_err=0.05,
    n_samples=10_000,
)
# Columns: v_mic, v_mac_bruntt (Bruntt 2010), v_mac_doyle (Doyle 2014)
```

#### Rotation period from v sin i

```python
from exohelp.star.spectroscopy import sample_rotation_period_from_vsini

result = sample_rotation_period_from_vsini(
    vsini=3.0, vsini_err=0.3,
    r_star=1.0, r_star_err=0.05,
)
# Column: max_rotation_period (days) — upper limit when inclination is unknown
```

### Archive

Requires the `archive` dependency group (`httpx`, `pandas`):

```bash
uv sync --group archive
```

```python
from exohelp.archive import ConfirmedExoplanetLoader

loader = ConfirmedExoplanetLoader()
df = loader.load()  # fetches from NASA Exoplanet Archive, caches locally
```

## Conventions

- **Units:** Plain floats are interpreted in canonical units (documented per function). Pass `astropy.Quantity` for explicit unit control.
- **Vectorized:** All functions accept numpy arrays and `astropy.Quantity` arrays.
- **Masked arrays:** Functions with validity ranges (e.g., chromospheric age) return `numpy.ma` masked arrays rather than raising errors.
- **Monte Carlo functions:** Return `astropy.table.QTable` with column descriptions citing source equations.

## Development

```bash
uv sync --group dev --group test
pre-commit install

pytest                        # run tests
pre-commit run --all-files    # lint and format
```

## Citation

If you use `exohelp` in your research, please cite it:

[![DOI](https://zenodo.org/badge/1077113378.svg)](https://doi.org/10.5281/zenodo.19255214)

```bibtex
@software{exohelp,
  author  = {Degen, David},
  title   = {exohelp},
  doi     = {10.5281/zenodo.19255214},
  url     = {https://github.com/dgegen/exohelp},
  license = {MIT},
}
```

## License

MIT
