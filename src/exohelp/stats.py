from typing import Any

import astropy.units as u
import numpy as np
from scipy.stats import truncnorm

from .type import QuantityLike

__all__ = ["truncated_normal"]


def truncated_normal(
    mean: QuantityLike,
    std: QuantityLike,
    size: int | tuple[int, ...],
    lower: QuantityLike = -np.inf,
    upper: QuantityLike = np.inf,
    rng: Any = None,
) -> np.ndarray | u.Quantity:
    """
    Generate samples from a truncated normal distribution.

    Supports scalar and array inputs, as well as ``astropy.units.Quantity`` objects.
    If units are provided, all boundary quantities are converted to the mean's unit
    and the returned samples will carry that unit.

    Parameters
    ----------
    mean : QuantityLike
        Mean of the unclipped normal distribution (scalar or array).
    std : QuantityLike
        Standard deviation of the unclipped normal distribution (scalar or array).
    size : int or tuple of ints
        Output shape of the samples.
    lower : QuantityLike, optional
        Lower truncation bound (default: -infinity).
    upper : QuantityLike, optional
        Upper truncation bound (default: +infinity).
    rng : np.random.Generator, int, or None, optional
        Pseudorandom number generator state or seed.

    Returns
    -------
    np.ndarray or astropy.units.Quantity
        Samples drawn from the truncated normal distribution.

    Examples
    --------
    >>> import numpy as np
    >>> from exohelp.stats import truncated_normal
    >>> samples = truncated_normal(mean=10.0, std=2.0, size=100, lower=0.0, rng=42)
    >>> np.all(samples >= 0.0)
    np.True_
    """
    unit = None
    if isinstance(mean, u.Quantity):
        unit = mean.unit
        mean_val = np.asanyarray(mean.value)
        std_val = np.asanyarray(std.to_value(unit) if isinstance(std, u.Quantity) else std)
        lower_val = np.asanyarray(lower.to_value(unit) if isinstance(lower, u.Quantity) else lower)
        upper_val = np.asanyarray(upper.to_value(unit) if isinstance(upper, u.Quantity) else upper)
    else:
        if isinstance(std, u.Quantity):
            unit = std.unit
        mean_val = np.asanyarray(mean)
        std_val = np.asanyarray(
            std.to_value(unit)
            if isinstance(std, u.Quantity) and unit is not None
            else (std.value if isinstance(std, u.Quantity) else std)
        )
        lower_val = np.asanyarray(
            lower.to_value(unit)
            if isinstance(lower, u.Quantity) and unit is not None
            else (lower.value if isinstance(lower, u.Quantity) else lower)
        )
        upper_val = np.asanyarray(
            upper.to_value(unit)
            if isinstance(upper, u.Quantity) and unit is not None
            else (upper.value if isinstance(upper, u.Quantity) else upper)
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        safe_std = np.where(std_val == 0.0, 1.0, std_val)
        a = (lower_val - mean_val) / safe_std
        b = (upper_val - mean_val) / safe_std
        res = truncnorm.rvs(a, b, loc=mean_val, scale=safe_std, size=size, random_state=rng)
        if np.any(std_val == 0.0):
            res = np.where(std_val == 0.0, mean_val, res)

    if unit is not None:
        return u.Quantity(res, unit)
    return res
