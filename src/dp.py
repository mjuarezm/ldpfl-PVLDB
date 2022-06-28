import pandas as pd
import numpy as np
from numpy.random import binomial, choice, multivariate_normal, laplace
import scipy.stats as stats
from scipy.special import expit
from utils2 import *
from optim import *


class Mechanism():
    """Implements a template class for a DP mechanism."""
    def __init__(self, ε:float=1) -> None:
        """Initialize the mechanism."""
        self.ε = ε
    
    def perturb(self, x: np.ndarray) -> np.ndarray:
        """Template method for the perturbation."""
        pass

    def _debias(self, y: np.ndarray) -> np.ndarray:
        pass

    def mean_estimate(self, y: np.ndarray) -> float:
        pass


class RRMechanism(Mechanism):
    """
    Implements a generic Randomized Response mechanism parametrized
    by ε, the differential privacy parameter. The inputs are arrays of
    binary vaues that are flipped according to ε.
    """

    @property
    def γ(self) -> float:
        """
        Return the probability of not flipping the value.

        :return: the probability of not flipping the value.
        :rtype: float
        """
        return expit(self.ε)
    
    def perturb(self, x:np.ndarray) -> np.ndarray:
        """
        Return the perturbed values `x` with Randomized Response.
        
        :param np.ndarray x: a 2D array of an arbitrary shape.
        :return: the perturbed values.
        :rtype: np.ndarray
        """
        assert(isbinary(x))
        y = x.copy()
        mask = binomial(1, 1 - self.γ, size=x.shape).astype(bool)
        y[mask] = 1 - y[mask]
        return y

    def _debias(self, y: np.ndarray) -> np.ndarray:
        """
        Return the debiased values to estimate the mean.
        
        :param np.ndarray y: a 2D array of an arbitrary shape.
        :return: the debiased values.
        :rtype: np.ndarray
        """
        return y * (2 * self.γ - 1) - self.γ + 1

    def mean_estimate(self, y:np.ndarray) -> float:
        """
        Return the mean estimate from the perturbed values.
        
        :param np.ndarray y: a 2D array of an arbitrary shape.
        :return: the mean estimate.
        :rtype: float
        """
        return self._debias(y, self.ε).mean()


class LaplaceMechanism(Mechanism):
    def __init__(self, ε: float = 1, sensitivity: float = 2.0) -> None:
        self.sensitivity = sensitivity
        super().__init__(ε)

    def laplacian_noise(self, size: int) -> np.ndarray:
        noise = laplace(loc=0.0, scale=self.sensitivity / self.ε, size=size)
        return noise

    def perturb(self, x: np.ndarray) -> np.ndarray:
        '''Return perturbed x with the Laplacian mechanism.
        
        If the statistic is the group mean value, then the  sensitivity depends on the
        size of the smallest group.

        We also clip the perturbed values to be within the range [-1, 1]. We do this instead
        of using the truncated Laplace distribution because it has been shown that the mechanism
        that uses the truncated Laplace distribution is not epsilon DP.
            https://arxiv.org/pdf/1808.10410.pdf
        On the other hand, clipping is a post-processing operation, which preserves DP.
        However, clipping biases the expected values of the estimations.
        Update: WE DO NOT NEED TO CLIP...!
        '''
        perturbed = x + self.laplacian_noise(x.shape)
        # perturbed = np.clip(perturbed, -1, 1)
        return perturbed

    def mean_estimate(self, y: np.ndarray) -> float:
        return y.mean()


class GVComposition(Mechanism):
    """Template class for Group-Value composite mechanisms."""
    def __init__(self, ε_group, ε_value, new_values=(0., 0.)):
        self.ε1 = ε_group
        self.ε2 = ε_value
        self.new_values = new_values

    def bound(self):
        """Return the epsilon guaranteed by the mechanism."""
        pass

    def __str__(self):
        return self.__class__.__name__

    def group_mean_estimates(self, g: np.ndarray, v: np.ndarray) -> np.ndarray:
        pass
    

    def group_count_estimates(self, g: np.ndarray, v: np.ndarray) -> np.ndarray:
        pass


class NaiveMechanism(GVComposition):
    def __init__(self, ε_group, ε_value):
        self.group_mech = RRMechanism(ε=ε_group)
        self.value_mech = LaplaceMechanism(ε=ε_value)
        super().__init__(ε_group, ε_value)

    def perturb(self, g: np.ndarray, v: np.ndarray) -> np.ndarray:
        new_g = self.group_mech.perturb(g)
        new_v = self.value_mech.perturb(v)
        return new_g, new_v

    def bound(self):
        # TODO: analysis of DP guarantee
        pass

    def expected_group_mean(self):
        #  TODO: what is the expected value of the group means after perturbation
        pass

    def group_mean_estimates(self, g: np.ndarray, v: np.ndarray) -> np.ndarray:
        pass
    
    def group_count_estimates(self, g: np.ndarray, v: np.ndarray) -> np.ndarray:
        pass


class KVMechanism(GVComposition):
    def __init__(self, ε):
        self.k = None
        self.group_mech = RRMechanism(ε=ε)
        self.value_mech = RRMechanism(ε=ε)
        super().__init__(ε, ε)

    def discretize(self, v):
        '''Discretize value to {-1, 1} with probabilities proportional to the value.'''
        p = (1 + v) / 2
        return binomial(1, p)
    
    def perturb(self, g: np.ndarray, v: np.ndarray) -> np.ndarray:
        new_g = self.group_mech.perturb(g)
        new_v = v.copy()
        is_kept = new_g == g
        new_v[~is_kept & (new_g == 1)] = self.new_values[0]
        new_v[~is_kept & (new_g == 0)] = self.new_values[1]
        new_v = self.discretize(new_v)
        new_v = self.value_mech.perturb(new_v)
        new_v = 2 * new_v - 1
        return new_g, new_v

    def bound(self):
        return max(self.ε1, self.ε2)
    
    def expected_group_means(self, means):
        #  minority, majority
        return self.group_mech.γ * (2 * self.value_mech.γ - 1) * means

    def group_mean_estimates(self, g: np.ndarray, v: np.ndarray) -> np.ndarray:
        '''Unbiases the group means estimates.'''
        return group_sums(g, v) / (100 * self.group_mech.γ * (2 * self.value_mech.γ - 1))
    
    def group_count_estimates(self, g: np.ndarray, v: np.ndarray) -> np.ndarray:
        return group_counts(g, v)
    

class RLMechanism(GVComposition):

    def __init__(self, ε):
        ε_group, ε_value, self.k = self.optimize_budget(ε)
        self.group_mech = RRMechanism(ε=ε_group)
        self.value_mech = LaplaceMechanism(ε=ε_value)

        #  add less noise to those who swap
        s = sensitivity(self.k, 1)
        self.value_mech_k = LaplaceMechanism(ε=ε_value, sensitivity=s)
        
        super().__init__(ε_group, ε_value)

    def optimize_budget(self, ε):
        return ε / 2., ε, 2.

    def perturb(self, g: np.ndarray, v: np.ndarray) -> np.ndarray:
        new_g = self.group_mech.perturb(g)
        new_v = v.copy()
        is_kept = new_g == g
        new_v[~is_kept & (new_g == 1)] = self.new_values[0]
        new_v[~is_kept & (new_g == 0)] = self.new_values[1]
        new_v[is_kept] = self.value_mech.perturb(new_v[is_kept])
        new_v[~is_kept] = self.value_mech_k.perturb(new_v[~is_kept])
        return new_g, new_v

    def bound(self):
        return max(self.ε2, self.ε2 / 2 + self.ε1)

    def expected_group_means(self, means):
        #  minority, majority
        return self.group_mech.γ * means

    def group_mean_estimates(self, g: np.ndarray, v: np.ndarray) -> np.ndarray:
        #  estimate the expected number of elements that stay in the group
        #  problem: it has a high variance np(1-p)  (binomial)
        # counts = group_counts(g, v)
        # sums = group_sums(g, v)
        # return sums / counts
        
        #  estimate as if size of perturbed group was fixed
        # (works well in the balanced case but can't work out the maths for the error)
        # return group_means(g, v) / self.group_mech.γ
        return group_sums(g, v) / (100 * self.group_mech.γ)

    def group_count_estimates(self, g: np.ndarray, v: np.ndarray) -> np.ndarray:
        return group_counts(g, v) / self.group_mech.γ
    

class RLMechanismK(RLMechanism):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def optimize_budget(self, ε):
        return np.log(3) - 0.5 * ε, ε, 2. / 3

    def bound(self):
        m1 = self.ε2
        m2 = np.log(2.0 / self.k) + self.ε2 * 0.5 - self.ε1
        m3 = np.log(0.5 * self.k) + self.ε2 / self.k + self.ε1
        return max(m1, m2, m3)

    
class RLMechanismKN(RLMechanismK):
    def __init__(self, ε_group, ε_value, k=2, n=1):
        super().__init__(ε_group, ε_value, k)
        s = sensitivity(k, n)
        self.value_mech_k = LaplaceMechanism(ε=ε_value, sensitivity=s)
        
        
class IterativeMechanism(GVComposition):
    def __init__(self, iterations, *params, **kwparams):
        self.iterations = iterations
        super().__init__(*params, **kwparams)


class RLMechanismKIterative(IterativeMechanism):
    def perturb(self, g: np.ndarray, v: np.ndarray) -> np.ndarray:
        for i in range(self.iterations):
            perturbed = super().perturb(g, v)
            self.new_values = self.group_mean_estimates(*perturbed)
