from distributions import Uniform, RandomVar, Joint
from partition import AdaptiveAlgorithm, NonAdaptivePartition
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import chi2


def main():
    sample_size = 5000
    delta = lambda x: chi2.ppf(0.97, x ** 2 - 1)
    r = 2
    s = 2

    U = RandomVar(Uniform(-np.pi, np.pi))
    x = RandomVar.operation(U, np.cos)
    y = RandomVar.operation(U, np.sin)

    xy = Joint(x, y)

    xy_sample = xy.sample(sample_size)

    # Adaptive algorithm
    adaptive_algorithm = AdaptiveAlgorithm(xy_sample, delta, r, s)

    non_adaptive_part = NonAdaptivePartition(xy_sample, bins=[50, 50]).run()
    final_partition = adaptive_algorithm.run()

    print(len(non_adaptive_part))
    print(len(final_partition))

    fig, ax = plt.subplots()
    adaptive_algorithm.plot_data(ax, color='red', alpha=0.5, markersize=2)
    adaptive_algorithm.plot_partition(ax, final_partition)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Adaptive Partition and samples")
    plt.show()


if __name__ == "__main__":
    main()
