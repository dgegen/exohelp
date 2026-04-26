import astropy.units as u

__all__ = ["S_earth"]


# Nominal total solar irradiance at 1 AU (solar constant).
S_earth = u.def_unit(
    ["S_earth", "S_oplus"],
    u.Quantity(1361, "W / m2"),
    format={"latex": r"S_{\oplus}"},
    doc="Nominal total solar irradiance at 1 AU (Solar Constant)",
)
