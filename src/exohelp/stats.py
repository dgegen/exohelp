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

    Supports both dimensionless numeric inputs and ``astropy.units.Quantity`` objects.
    If units are provided, all boundary quantities are converted to the mean's unit
    and the returned samples will carry that unit.

    Parameters
    ----------
    mean : QuantityLike
        Mean of the unclipped normal distribution.
    std : QuantityLike
        Standard deviation of the unclipped normal distribution.
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
        mean_val = float(mean.value)
        std_val = float(std.to_value(unit) if isinstance(std, u.Quantity) else std)
        lower_val = float(lower.to_value(unit)) if isinstance(lower, u.Quantity) else float(lower)
        upper_val = float(upper.to_value(unit)) if isinstance(upper, u.Quantity) else float(upper)
    else:
        mean_val = float(mean)
        if isinstance(std, u.Quantity):
            unit = std.unit
            std_val = float(std.value)
        else:
            std_val = float(std)

        lower_val = (
            float(lower.to_value(unit))
            if isinstance(lower, u.Quantity) and unit is not None
            else (float(lower.value) if isinstance(lower, u.Quantity) else float(lower))
        )
        upper_val = (
            float(upper.to_value(unit))
            if isinstance(upper, u.Quantity) and unit is not None
            else (float(upper.value) if isinstance(upper, u.Quantity) else float(upper))
        )

    if std_val == 0.0:
        res = np.full(size, mean_val)
    else:
        a = (lower_val - mean_val) / std_val
        b = (upper_val - mean_val) / std_val
        res = truncnorm.rvs(a, b, loc=mean_val, scale=std_val, size=size, random_state=rng)

    if unit is not None:
        return u.Quantity(res, unit)
    return res
