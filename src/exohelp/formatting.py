from typing import Any, Literal

import numpy as np

__all__ = [
    "decimal_places_from_sigfigs",
    "format_number",
    "format_summary_table",
    "format_value_with_uncertainty",
]


def decimal_places_from_sigfigs(number: float, significant_digits: int = 0) -> int:
    return (
        0 if number == 0 else max(0, significant_digits - 1 - int(np.floor(np.log10(abs(number)))))
    )


def _use_adaptive_sigfigs(uncertainty: float) -> int:
    """Return 2 sig figs if the first digit of uncertainty is 1 or 2, else 1.

    Rationale: the relative jump between 1 and 2 is large (100%), so retaining
    a second digit is informative.  For digits 3-9 the relative jump is smaller
    and one digit suffices.
    """
    if uncertainty == 0:
        return 1
    first_digit = int(f"{abs(uncertainty):.10e}"[0])
    return 2 if first_digit <= 2 else 1


def _format_scientific_latex(f: float, significant_digits: int = 1) -> str:
    if f == 0:
        return "0"
    float_str = "{0:.{1}g}".format(f, significant_digits)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        exponent_str = r"10^{{{0}}}".format(int(exponent))
        if base == "1":
            return exponent_str
        return r"{0} \times {1}".format(base, exponent_str)
    return float_str


def format_number(
    num: float,
    decimal_places: int | None = None,
    sci_notation_threshold: int = 5,
    significant_digits: int = 1,
    strip_trailing_zeros: bool = False,
) -> str:
    if decimal_places is None:
        decimal_places = decimal_places_from_sigfigs(num, significant_digits=significant_digits)
    if abs(num) < float(f"1e-{sci_notation_threshold}") or abs(num) > float(
        f"1e{sci_notation_threshold}"
    ):
        result = _format_scientific_latex(num, significant_digits=significant_digits)
    else:
        result = f"{num:.{decimal_places}f}"

    if strip_trailing_zeros and "." in result:
        result = result.rstrip("0").rstrip(".")
        if result in {"", "-"}:
            result = "0"

    return result


def _format_asymmetric(
    mean: float,
    lower_uncertainty: float,
    upper_uncertainty: float,
    significant_digits: int,
    sci_notation_threshold: int,
    adaptive_sigfigs: bool,
    pm_if_equal: bool,
) -> str:
    if adaptive_sigfigs:
        upper_sig = _use_adaptive_sigfigs(upper_uncertainty)
        lower_sig = _use_adaptive_sigfigs(lower_uncertainty)
    else:
        upper_sig = lower_sig = significant_digits

    upper_prec = int(decimal_places_from_sigfigs(upper_uncertainty, significant_digits=upper_sig))
    lower_prec = int(decimal_places_from_sigfigs(lower_uncertainty, significant_digits=lower_sig))

    if upper_prec <= lower_prec:
        mean_prec, mean_sig = upper_prec, upper_sig
    else:
        mean_prec, mean_sig = lower_prec, lower_sig

    fmt = {"sci_notation_threshold": sci_notation_threshold}
    upper_str = format_number(
        upper_uncertainty, decimal_places=upper_prec, significant_digits=upper_sig, **fmt
    )
    lower_str = format_number(
        lower_uncertainty, decimal_places=lower_prec, significant_digits=lower_sig, **fmt
    )
    mean_str = format_number(mean, decimal_places=mean_prec, significant_digits=mean_sig, **fmt)

    if pm_if_equal and upper_str == lower_str:
        return rf"${mean_str} \pm {upper_str}$"
    mean_grouped = f"{{{mean_str}}}" if "^" in mean_str else mean_str
    return rf"${mean_grouped}^{{+{upper_str}}}_{{-{lower_str}}}$"


def _format_pm(
    mean: float,
    uncertainty: float,
    significant_digits: int,
    sci_notation_threshold: int,
    adaptive_sigfigs: bool,
) -> str:
    sig_digits = _use_adaptive_sigfigs(uncertainty) if adaptive_sigfigs else significant_digits
    decimal_places = int(decimal_places_from_sigfigs(uncertainty, significant_digits=sig_digits))
    fmt = {
        "decimal_places": decimal_places,
        "sci_notation_threshold": sci_notation_threshold,
        "significant_digits": sig_digits,
    }
    return rf"${format_number(mean, **fmt)} \pm {format_number(uncertainty, **fmt)}$"


def format_value_with_uncertainty(
    mean: float,
    lower_uncertainty: float,
    upper_uncertainty: float | None = None,
    significant_digits: int = 1,
    sci_notation_threshold: int = 5,
    adaptive_sigfigs: bool = False,
    uncertainty_style: Literal["asymmetric", "pm"] = "asymmetric",
    pm_if_equal: bool = True,
) -> str:
    """Return a LaTeX string formatting mean with its uncertainty.

    For uncertainty_style="asymmetric", lower_uncertainty and upper_uncertainty
    are the lower and upper deviations from mean (both positive).
    For uncertainty_style="pm", lower_uncertainty is used as the symmetric ± value.
    """
    if upper_uncertainty is None:
        upper_uncertainty = lower_uncertainty

    if uncertainty_style == "asymmetric":
        return _format_asymmetric(
            mean,
            lower_uncertainty,
            upper_uncertainty,
            significant_digits,
            sci_notation_threshold,
            adaptive_sigfigs,
            pm_if_equal,
        )
    return _format_pm(
        mean, lower_uncertainty, significant_digits, sci_notation_threshold, adaptive_sigfigs
    )


def format_summary_table(
    df: Any,
    center_col: str = "mean",
    lower_col: str = "std",
    upper_col: str | None = None,
    **kwargs,
) -> np.ndarray:
    """Apply format_value_with_uncertainty row-wise to columns of a table or DataFrame.

    Parameters
    ----------
    df : pd.DataFrame or astropy.table.Table
        Input table, e.g. from arviz.summary.
    center_col : str
        Column of central values (mean or median).
    lower_col : str
        Symmetric uncertainty column (std / mad), or lower quantile when upper_col is given.
    upper_col : str, optional
        Upper quantile column. When provided, lower_col and upper_col are treated as
        quantile values (e.g. 16th / 84th percentile) and the asymmetric deviations
        center - lower and upper - center are computed automatically.
    **kwargs
        Forwarded to format_value_with_uncertainty.

    Returns
    -------
    np.ndarray of str
    """
    center = np.asarray(df[center_col], dtype=float)
    lower = np.asarray(df[lower_col], dtype=float)

    if upper_col is None:
        return np.array(
            [format_value_with_uncertainty(c, lo, **kwargs) for c, lo in zip(center, lower)]
        )

    upper = np.asarray(df[upper_col], dtype=float)

    return np.array(
        [
            format_value_with_uncertainty(c, c - lo, up - c, **kwargs)
            for c, lo, up in zip(center, lower, upper)
        ]
    )
