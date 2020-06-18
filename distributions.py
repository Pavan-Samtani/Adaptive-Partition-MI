from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class Distribution(ABC):
    KEEP = False

    def __init__(self):
        self.sampled = None

    def sample(self, n):
        if self.sampled is not None and Distribution.KEEP:
            return self.sampled

        else:
            samples = self.get_sample(n)
            self.sampled = samples if Distribution.KEEP else None
            return samples

    @abstractmethod
    def get_sample(self, n):
        raise NotImplementedError("sample not implemented")

    @staticmethod
    def multiply(*dists):
        return type("MultDist", (Distribution, object), {"get_sample": lambda self, n: np.prod([d.sample(n) for d in dists], axis=0)})()

    @staticmethod
    def sum(*dists):
        return type("SumDist", (Distribution, object), {"get_sample": lambda self, n: np.sum([d.sample(n) for d in dists], axis=0)})()

    @staticmethod
    def operation(dist, oper):
        return type("OpDist", (Distribution, object), {"get_sample": lambda self, n: oper(dist.sample(n))})()


class Distribution2D(Distribution):
    def __init__(self, d1, d2):
        super().__init__()
        self.d1 = d1
        self.d2 = d2

    def get_sample(self, n):
        return self.d1.sample(n), self.d2.sample(n)


class Uniform(Distribution):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    def get_sample(self, n):
        return np.random.uniform(self.a, self.b, size=n)


class Normal(Distribution):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def get_sample(self, n):
        return np.random.normal(self.mean, self.std, size=n)


class RandomVar:
    def __init__(self, dist: Distribution=None):
        self._dist = dist

    @property
    def dist(self):
        return self._dist

    def sample(self, n):
        try:
            return self.dist.sample(n)

        except AttributeError:
            if self._dist is None:
                raise AttributeError("Distribution is not defined for this random variable")
            
            else:
                raise

    @staticmethod
    def multiply(*rvars):
        return RandomVar(Distribution.multiply(*[rvar.dist for rvar in rvars]))

    @staticmethod
    def sum(*rvars):
        return RandomVar(Distribution.sum(*[rvar.dist for rvar in rvars]))

    @staticmethod
    def operation(rvar, oper):
        return RandomVar(Distribution.operation(rvar.dist, oper))


class Joint(RandomVar):
    def __init__(self, x: RandomVar, y: RandomVar):
        super().__init__()
        self._x = x
        self._y = y
        self._dist = Distribution2D(x.dist, y.dist)
    
    def sample(self, n):
        Distribution.KEEP = True
        x, y = self.dist.sample(n)
        Distribution.KEEP = False

        return x, y

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    U = RandomVar(Uniform(-np.pi, np.pi))
    x = RandomVar.operation(U, np.cos)
    y = RandomVar.operation(U, np.sin)

    xy = Joint(x, y)

    xs, ys = xy.sample(100)

    plt.plot(xs, ys, 'o')
    plt.plot(x.sample(100), y.sample(100), 'o')
    plt.show()
