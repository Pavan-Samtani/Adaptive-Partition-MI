from __future__ import annotations
from typing import Union, List
import numpy as np


class SmartRectangle:

    def __init__(self, xlim, ylim, plane):
        self.xlim = xlim
        self.ylim = ylim
        self.plane = plane
        self.samples_inside = plane.get_samples(xlim, ylim)
    
    def split(self) -> List[SmartRectangle]:
        pass


class Plane:

    def __init__(self, sample):
        self.sample = sample

        xmin, ymin = np.min(sample, axis=0)
        xmax, ymax = np.max(sample, axis=0)

        self.xlim = [xmin, xmax]
        self.ylim = [ymin, ymax]

    def get_samples(self, xlim, ylim):
        xlimit = (self.sample[:, 0] >= xlim[0]) & (self.sample[:, 0] <= xlim[1])
        ylimit = (self.sample[:, 1] >= ylim[0]) & (self.sample[:, 1] <= ylim[1])
        return self.sample[xlimit & ylimit]

class AdaptivePartition:

    def __init__(self, plane, r, s):
        self.plane = plane
        self.current_partition = []
        self.changing = True
        self.r = r
        self.s = s

    def initialize_partition(self):
        # Generate equiprobable partition
        xmarg, ymarg = list(zip(*self.plane.sample))

        xpartition = self.plane.xlim
        ypartition = self.plane.ylim

        for j in range(1, self.r):
            xpartition.insert(j, np.quantile(xmarg, j / self.r))
            ypartition.insert(j, np.quantile(ymarg, j / self.r))

        partition = self.points_to_partition(xpartition, ypartition)
        return partition

    def points_to_partition(self, xpart, ypart):
        smart_rects = []
        for ix in range(len(xpart) - 1):
            for iy in range(len(ypart) - 1):
                smart_rects.append(SmartRectangle([xpart[ix], xpart[ix + 1]], [ypart[iy], ypart[iy + 1]], self.plane))

        return smart_rects

    def update_partition(self):
        pass


class AdaptiveAlgorithm:

    def __init__(self, sample, delta, r, s):
        self.sample = sample
        self.plane = Plane(sample)
        self.adaptive_partition = AdaptivePartition(plane, r, s)

    def run(self):
        # Step 0: Generate first partition
        self.adaptive_partition.initialize_partition()

        # While partition is changing
        while self.adaptive_partition.changing:
            # Step 1: Generate next partition
            r_k = self.adaptive_partition.current_partition


        # Step 2: End if partition didn't change in last iteration
        self.final_partition = self.adaptive_partition.current_partition



if __name__ == "__main__":
    from distributions import Uniform, RandomVar, Joint
    from matplotlib import pyplot as plt

    U = RandomVar(Uniform(-np.pi, np.pi))
    x = RandomVar.operation(U, np.cos)
    y = RandomVar.operation(U, np.sin)

    xy = Joint(x, y)

    xy_sample = xy.sample(100)

    delta = 0.1
    r = 2
    s = 2

    # Adaptive partition
    plane = Plane(xy_sample)
    apartition = AdaptivePartition(plane, r, s)

    for r in apartition.initialize_partition():
        print(r.xlim, r.ylim, r.samples_inside)

    # adaptive_algorithm = AdaptiveAlgorithm(xy_sample, delta, r, s)
    # final_partition = adaptive_algorithm.run()
