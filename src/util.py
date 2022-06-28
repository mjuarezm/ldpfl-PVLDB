## DATA GENERATION UTILS

import numpy as np
import scipy.stats as stats


def trunc_norm(size: int = 1, loc: float = 0, scale: float = 1,
               lower: float = -1., upper: float = 1., seed: int = 0) -> np.ndarray:
    return stats.truncnorm((lower - loc) / scale, (upper - loc) / scale, loc=loc, scale=scale).rvs(size)
