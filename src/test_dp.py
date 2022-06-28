import unittest
import numpy as np

from dp import *
from datagen import *
from utils import *


class TestRRMechanism(unittest.TestCase):
    def setUp(self):
        self.m = RRMechanism(ε=1)
        self.x = np.random.choice([0, 1], size=(100,)).reshape((5, 20))
        self.y = self.m.perturb(self.x)

    def test_perturb_sizes(self):
        self.assertEqual(self.x.shape, self.y.shape)

    def test_output_binary(self):
        self.assertTrue(isbinary(self.y))

    def test_input_not_a_binary_array(self):
        with self.assertRaises(AssertionError):
            self.m.perturb(np.array([1, 2, 3]))

    def test_gamma_probability(self):
        p = (self.y == self.x).sum() / self.y.size  # estimate probability of not flipping
        self.assertAlmostEqual(self.m.γ, p, delta=0.1)


class TestLaplaceMechanism(unittest.TestCase):
    def setUp(self):
        self.m = LaplaceMechanism(ε=1)
        self.x = trunc_norm(100, 100)
        self.y = self.m.perturb(self.x)
        self.noise = self.y - self.x

    def test_perturb_sizes(self):
        self.assertEqual(self.x.shape, self.y.shape)

    def test_perturb_output_range(self):
        self.assertTrue(((self.y >= -1) & (self.y <= 1)).all())

    def test_mean_noise(self):
        noise = self.m.laplacian_noise(10000)
        self.assertAlmostEqual(noise.mean(), 0.0, delta=0.1)

    def test_scale_noise(self):
        noise = self.m.laplacian_noise(10000)
        self.assertAlmostEqual(noise.std(), np.sqrt(2) * self.m.sensitivity(k=2, n=1), delta=0.1)


class TestGVComposition(unittest.TestCase):
    def test_build_compositions(self):
        ε_group, ε_value = 0.5, 0.5
        M_NA = NaiveMechanism(ε_group, ε_value)
        M_KV = KVMechanism(ε_group, ε_value)
        M_RL = RLMechanism(ε_group, ε_value)
        M_RK = RLMechanism(ε_group, ε_value, k=1.5)
        
        data = datagen(n=100)
        M_NA.perturb(*data)
        M_KV.perturb(*data)
        M_RL.perturb(*data)
        M_RK.perturb(*data)
    

if __name__ == '__main__':
    unittest.main()
