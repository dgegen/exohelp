from typing import TypeAlias

import astropy.units as u
import numpy as np
import numpy.typing as npt

ArrayLike: TypeAlias = float | npt.NDArray[np.float64]
QuantityLike: TypeAlias = u.Quantity | ArrayLike
