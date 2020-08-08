import time

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats.stats import pearsonr   
from scipy.stats import chi2

from distributions import MultivariateNormal
from divergence_utils import kl_estimate
from partition import AdaptiveAlgorithm, NonAdaptivePartition, Plane
from table_gen import generate_timing_table


def main():

    sample_sizes = [250, 500, 1000, 2000, 10000]
    rhos = [0, 0.3, 0.6, 0.9]
    r = 2
    s = 2
    K = 50

    time_results = {rho: [{ssize: [] for ssize in sample_sizes}, None] for rho in rhos}
    all_results = []

    for rho in rhos:
        for sample_size in sample_sizes:

            cov = np.array([[1., rho], [rho, 1.]])
            dist = MultivariateNormal(mean=np.zeros(2), cov=cov)

            t_ci, t_nad, t_ml = [], [], []

            delta = lambda x: chi2.ppf(0.97, x ** 2 - 1)

            print(f"Timing samples {sample_size} for r = {rho}")

            for k in range(K):
                xy_sample = dist.sample(sample_size)

                plane = Plane(xy_sample)

                # Adaptive algorithm
                t0_ad = time.time()
                ad = AdaptiveAlgorithm(xy_sample, delta, r, s).run()
                t_ci.append(time.time() - t0_ad)

                t0_nad = time.time()
                nad = NonAdaptivePartition(xy_sample, bins=[50, 50]).run()
                t_nad.append(time.time() - t0_nad)

                t0_ml = time.time()
                ml = - np.log(1 - pearsonr(xy_sample[:, 0], xy_sample[:, 1])[0] ** 2) / 2
                t_ml.append(time.time() - t0_ml)

                all_results.append((ad, nad, ml))

            time_results[rho][0][sample_size] = [np.mean(t_ml), np.mean(t_ci), np.mean(t_nad)]

            print(f"Times: ML: {np.mean(t_ml)}, CI: {np.mean(t_ci)}, NAD: {np.mean(t_nad)}")

    generate_timing_table(time_results)

    print(len(all_results))


if __name__ == "__main__":
    main()
