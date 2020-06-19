from distributions import Uniform, RandomVar, Joint
from partition import AdaptiveAlgorithm
from matplotlib import pyplot as plt
import numpy as np


def main():
    sample_size = 1000
    delta = 0.03
    r = 2
    s = 2

    U = RandomVar(Uniform(-np.pi, np.pi))
    x = RandomVar.operation(U, np.cos)
    y = RandomVar.operation(U, np.sin)

    xy = Joint(x, y)

    xy_sample = xy.sample(sample_size)

    # Adaptive algorithm
    adaptive_algorithm = AdaptiveAlgorithm(xy_sample, delta, r, s)
    final_partition = adaptive_algorithm.run()

    fig, ax = plt.subplots()
    adaptive_algorithm.plot_data(ax, color='red', alpha=0.5)
    adaptive_algorithm.plot_partition(ax, final_partition)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Adaptive Partition and samples")
    plt.show()


if __name__ == "__main__":
    main()
