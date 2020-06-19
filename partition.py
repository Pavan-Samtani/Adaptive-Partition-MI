from __future__ import annotations
from typing import Union, List
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections


class CustomRect(collections.LineCollection):
    def __init__(self, xlim, ylim, *args, **kwargs):
        lines = [
            [(xlim[0], ylim[0]), (xlim[0], ylim[1])],
            [(xlim[0], ylim[0]), (xlim[1], ylim[0])],
            [(xlim[0], ylim[1]), (xlim[1], ylim[1])],
            [(xlim[1], ylim[0]), (xlim[1], ylim[1])]
        ]
        super().__init__(lines, *args, **kwargs)


class SmartRectangle:

    def __init__(self, xlim, ylim, plane):
        if len(xlim) != 2 or len(ylim) != 2:
            raise ValueError("xlim or ylim cannot have length different than 2")

        self.xlim = xlim
        self.ylim = ylim
        self.plane = plane
        self.samples_inside = plane.get_samples(xlim, ylim)

    def get_plot_rect(self):
        return CustomRect(self.xlim, self.ylim)

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


class AdaptiveAlgorithm:

    def __init__(self, sample, delta, r, s):
        self.sample = sample
        self.plane = Plane(sample)
        self.current_partition = None
        self.delta = delta
        self.r = r
        self.s = s

    def initialize_partition(self):
        # Generate equiprobable partition
        xmarg, ymarg = list(zip(*self.plane.sample))

        xpartition = self.plane.xlim[:]
        ypartition = self.plane.ylim[:]

        for j in range(1, self.r):
            xpartition.insert(j, np.quantile(xmarg, j / self.r))
            ypartition.insert(j, np.quantile(ymarg, j / self.r))

        self.current_partition = self.points_to_partition(xpartition, ypartition)
        self.rfinal = []

    def points_to_partition(self, xpart, ypart):
        smart_rects = []
        for ix in range(len(xpart) - 1):
            for iy in range(len(ypart) - 1):
                smart_rects.append(SmartRectangle([xpart[ix], xpart[ix + 1]], [ypart[iy], ypart[iy + 1]], self.plane))

        return smart_rects

    def plot_partition(self, ax, partition):
        for rect in partition:
            r_fig = rect.get_plot_rect()
            ax.add_collection(r_fig)

    def plot_data(self, ax, *args, **kwargs):
        ax.plot(*list(zip(*self.sample)), 'o', *args, **kwargs)

    def rectangle_subpartition(self, rect, partition_size):
        xmarg_rect = SmartRectangle(rect.xlim, self.plane.ylim, self.plane)
        ymarg_rect = SmartRectangle(self.plane.xlim, rect.ylim, self.plane)

        margx, _ = list(zip(*xmarg_rect.samples_inside))
        _, margy = list(zip(*ymarg_rect.samples_inside))

        condx = np.array(margx)
        condy = np.array(margy)

        xpartition = rect.xlim[:]
        ypartition = rect.ylim[:]

        for j in range(1, partition_size):
            xpartition.insert(j, np.quantile(condx, j / partition_size))
            ypartition.insert(j, np.quantile(condy, j / partition_size))

        return self.points_to_partition(xpartition, ypartition), xmarg_rect, ymarg_rect

    def run(self):
        # Step 0: Generate first partition
        self.initialize_partition()

        # While partition is changing
        while True:
            # Step 1: Generate next partition
            r_k = self.current_partition[:]
            r_next = []

            # For each rectangle in current partition
            for rect in r_k:
                # If the rectangle doesnt have samples inside, dont divide it
                if len(rect.samples_inside) == 0:
                    self.rfinal.append(rect)

                # Otherwise...
                else:
                    # Calculate subpartition of rectangle with s parameter
                    subp, xmarg_rect, ymarg_rect = self.rectangle_subpartition(rect, self.s)

                    joint_abl = np.array([len(r.samples_inside) / len(rect.samples_inside) for r in subp])
                    ARl_rects = [SmartRectangle(r.xlim, self.plane.ylim, self.plane) for r in subp]
                    RBl_rects = [SmartRectangle(self.plane.xlim, r.ylim, self.plane) for r in subp]

                    marginal_a = np.array([len(r.samples_inside) / len(xmarg_rect.samples_inside) for r in ARl_rects])
                    marginal_b = np.array([len(r.samples_inside) / len(ymarg_rect.samples_inside) for r in RBl_rects])

                    divergence = np.nansum(joint_abl * np.log(joint_abl / (marginal_a * marginal_b)))

                    if divergence > self.delta:
                        r_subp, _, _ = self.rectangle_subpartition(rect, self.r)
                        r_next.extend(r_subp)
                    
                    else:
                        self.rfinal.append(rect)

            self.current_partition = r_next

            # Step 2: End if partition didn't change in last iteration
            if len(self.current_partition) == 0:
                break

        return self.rfinal
