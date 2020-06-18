import unittest

import numpy as np

from distributions import Distribution, Uniform


class TestDistribution(unittest.TestCase):

    def seed(self):
        np.random.seed(1)

    def test_sample(self):
        u1 = Uniform(1, 2)
        u1.sample(5)
        u1.sample(10)

    def test_multiply(self):
        u1 = Uniform(1, 2)
        u2 = Uniform(4, 5)
        m1 = Distribution.multiply(u1, u2)
        self.seed()
        ground_truth = np.prod([u1.sample(10), u2.sample(10)], axis=0)
        self.seed()
        np.testing.assert_almost_equal(m1.sample(10), ground_truth)

    def test_sum(self):
        u1 = Uniform(1, 2)
        u2 = Uniform(4, 5)
        m1 = Distribution.sum(u1, u2)
        self.seed()
        ground_truth = np.sum([u1.sample(10), u2.sample(10)], axis=0)
        self.seed()
        np.testing.assert_almost_equal(m1.sample(10), ground_truth)

    def test_operation(self):
        u1 = Uniform(1, 2)
        m1 = Distribution.operation(u1, np.sin)
        self.seed()
        ground_truth = np.sin(u1.sample(10))
        self.seed()
        np.testing.assert_almost_equal(m1.sample(10), ground_truth)

        m1 = Distribution.operation(u1, np.cos)
        self.seed()
        ground_truth = np.cos(u1.sample(10))
        self.seed()
        np.testing.assert_almost_equal(m1.sample(10), ground_truth)

        m1 = Distribution.operation(u1, np.square)
        self.seed()
        ground_truth = np.square(u1.sample(10))
        self.seed()
        np.testing.assert_almost_equal(m1.sample(10), ground_truth)

        cube = lambda x: np.power(x, 3)
        m1 = Distribution.operation(u1, cube)
        self.seed()
        ground_truth = cube(u1.sample(10))
        self.seed()
        np.testing.assert_almost_equal(m1.sample(10), ground_truth)
