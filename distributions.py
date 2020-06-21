
# from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Union, List, Callable, Tuple

import numpy as np


Num = Union[int, float]


class Distribution(ABC):
    """Define a probability distribution."""
    KEEP = False

    def __init__(self):
        self.sampled = None

    def sample(self, n: int) -> Union[np.ndarray, list]:
        """Method to sample from distribution.
        
        This method shouldn't be overridden, the one to override in order
        to define the distribution sample is get_sample.
        """
        if self.sampled is not None and Distribution.KEEP:
            return self.sampled

        else:
            samples = self.get_sample(n)
            self.sampled = samples if Distribution.KEEP else None
            return samples

    @abstractmethod
    def get_sample(self, n: int) -> Union[np.ndarray, list]:
        """Sample the distribution.
        
        This method has to be overriden by subclasses in order to define
        the distribution itself.
        """
        raise NotImplementedError("sample not implemented")

    @staticmethod
    def multiply(*dists: 'Distribution') -> 'Distribution':
        return type("MultDist", (Distribution, object), {"get_sample": lambda self, n: np.prod([d.sample(n) for d in dists], axis=0)})()

    @staticmethod
    def sum(*dists: 'Distribution') -> 'Distribution':
        return type("SumDist", (Distribution, object), {"get_sample": lambda self, n: np.sum([d.sample(n) for d in dists], axis=0)})()

    @staticmethod
    def operation(dist: 'Distribution', oper: Callable) -> 'Distribution':
        return type("OpDist", (Distribution, object), {"get_sample": lambda self, n: oper(dist.sample(n))})()


class Distribution2D(Distribution):
    """Define a two dimensional probability distribution."""

    def __init__(self, d1: Distribution, d2: Distribution):
        super().__init__()
        self.d1 = d1
        self.d2 = d2

    def get_sample(self, n: int) -> list:
        return list(zip(self.d1.sample(n), self.d2.sample(n)))


class Uniform(Distribution):
    """Uniform probability distribution."""

    def __init__(self, a: Num, b: Num):
        super().__init__()
        self.a = a
        self.b = b

    def get_sample(self, n: int) -> np.ndarray:
        return np.random.uniform(self.a, self.b, size=n)


class Normal(Distribution):
    """Gaussian probability distribution."""

    def __init__(self, mean: Num, std: Num):
        super().__init__()
        self.mean = mean
        self.std = std

    def get_sample(self, n: int) -> np.ndarray:
        return np.random.normal(self.mean, self.std, size=n)


class MultivariateNormal(Distribution):
    """Multivariate Gaussian probability distribution."""    

    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        super().__init__()
        self.mean = mean
        self.cov = cov

    def get_sample(self, n: int) -> np.ndarray:
        return np.random.multivariate_normal(self.mean, self.cov, size=n)


class RandomVar:
    """Defines a random variable with a given distribution."""

    def __init__(self, dist: Distribution):
        self._dist = dist

    @property
    def dist(self):
        return self._dist

    def sample(self, n: int) -> Union[np.ndarray, list]:
        try:
            return self.dist.sample(n)

        except AttributeError:
            if self._dist is None:
                raise AttributeError("Distribution is not defined for this random variable")
            
            else:
                raise

    @staticmethod
    def multiply(*rvars: 'RandomVar') -> 'RandomVar':
        return RandomVar(Distribution.multiply(*[rvar.dist for rvar in rvars]))

    @staticmethod
    def sum(*rvars: 'RandomVar') -> 'RandomVar':
        return RandomVar(Distribution.sum(*[rvar.dist for rvar in rvars]))

    @staticmethod
    def operation(rvar: 'RandomVar', oper: Callable) -> 'RandomVar':
        return RandomVar(Distribution.operation(rvar.dist, oper))


class Joint(RandomVar):
    """Defines a random variable that has a joint distribution.
    
    The Joint distribution is defined by two previous random variables.
    """

    def __init__(self, x: RandomVar, y: RandomVar):
        super().__init__(Distribution2D(x.dist, y.dist))
        self._x = x
        self._y = y
    
    def sample(self, n: int) -> np.ndarray:
        Distribution.KEEP = True
        xy = self.dist.sample(n)
        Distribution.KEEP = False

        return np.array(xy)
