from hashlib import sha512 as _sha512
import numpy as np
import math
import random
from numba import jit
import _sample

class Sample(_sample.Sample):

    def __init__(self, x=None):
        pass

    def seed(self, x):
        """Taken from Python library."""
        if isinstance(x, (str, bytes, bytearray)):
            if isinstance(x, str):
                x = x.encode()
            x += _sha512(x).digest()
            x = int.from_bytes(x, 'big')
        super().seed(x)

    def multivariate_normal(self, mean, cov):
        """Sample from 2d normal"""
        L = np.array([[math.sqrt(cov[0,0]), 0], [1/math.sqrt(cov[0,0]) * (cov[1,0]), math.sqrt(cov[1,1] - (1/cov[0,0] * (cov[1,0])**2))]])
        z = np.array([self.c_normal(0,1), self.c_normal(0,1)])
        x = mean + L @ z
        return x


    def multivariate_normal_fast(self, mean, cov):
        """Sample from 2d normal.

        Parameters
        ---------
        mean : 1d array
        cov : 2 x 2 array
        """
        SQRT_COV_00 = math.sqrt(cov[0,0])
        z0 = self.c_normal(0,1)
        mean[0] += SQRT_COV_00 * z0
        mean[1] += cov[1,0] / SQRT_COV_00 * z0 + math.sqrt(cov[1,1] - cov[1,0]**2 / cov[0,0]) * self.c_normal(0,1)

        return mean


    def multivariate_normal2(self, mean, cov):
        """Sample from nd normal."""
        L = np.linalg.cholesky(cov)

        z = []
        for _ in range(len(mean)):
            z.append(self.c_normal(0,1))
        z = np.array(z)

        x = mean + L @ z
        return x


    def categorical_log(self, log_p):
        """Generate one sample from a categorical distribution with event
        probabilities provided in log-space.

        Parameters
        ----------
        log_p : array_like
            logarithms of event probabilities, which need not be normalized

        Returns
        -------
        int
            One sample from the categorical distribution, given as the index of that
            event from log_p.
        """
        exp_sample = math.log(random.random())
        events = np.logaddexp.accumulate(np.hstack([[-np.inf], log_p]))
        events -= events[-1]
        sample = next(x[0]-1 for x in enumerate(events) if x[1] >= exp_sample)
        return sample


_inst = Sample()
seed = _inst.seed
standard_uniform = _inst.c_standard_uniform
uniform = _inst.c_uniform
exponential = _inst.c_exponential
normal = _inst.c_normal
truncated_normal = _inst.c_truncated_normal
multivariate_normal = _inst.multivariate_normal
multivariate_normal_fast = _inst.multivariate_normal_fast
multivariate_normal2 = _inst.multivariate_normal2
categorical_log = _inst.categorical_log
