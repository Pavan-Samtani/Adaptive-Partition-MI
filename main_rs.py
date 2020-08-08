
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import chi2

from distributions import MultivariateNormal
from divergence_utils import kl_estimate
from partition import AdaptiveAlgorithm, Plane
from table_gen import generate_rs_table


def main():

    sample_sizes = [250, 500, 1000, 2000, 10000]
    rhos = [0, 0.3, 0.6, 0.9]
    K = 20

    rs_2 = []
    rs_4 = []
    rs_5 = []
    rs_10 = []

    results = {rho: [{ssize: [] for ssize in sample_sizes}, None] for rho in rhos}

    for rho in rhos:

        rs_2_std = []
        rs_4_std = []
        rs_5_std = []
        rs_10_std = []

        real_mi = - np.log(1 - rho ** 2) / 2

        results[rho][1] = real_mi

        for sample_size in sample_sizes:

            cov = np.array([[1., rho], [rho, 1.]])
            dist = MultivariateNormal(mean=np.zeros(2), cov=cov)

            rs_2_l, rs_4_l, rs_5_l, rs_10_l = [], [], [], []

            delta = lambda x: chi2.ppf(0.97, x ** 2 - 1)

            for k in range(K):
                
                xy_sample = dist.sample(sample_size)


                # Adaptive algorithm
                plane = Plane(xy_sample)
                rs_2_l.append(kl_estimate(plane, AdaptiveAlgorithm(xy_sample, delta, 2, 2).run()))
                plane = Plane(xy_sample)
                rs_4_l.append(kl_estimate(plane, AdaptiveAlgorithm(xy_sample, delta, 4, 4).run()))
                plane = Plane(xy_sample)
                rs_5_l.append(kl_estimate(plane, AdaptiveAlgorithm(xy_sample, delta, 5, 5).run()))
                plane = Plane(xy_sample)
                rs_10_l.append(kl_estimate(plane, AdaptiveAlgorithm(xy_sample, delta, 10, 10).run()))


            results[rho][0][sample_size] = [np.mean(rs_2_l), np.mean(rs_4_l), np.mean(rs_5_l), np.mean(rs_10_l)]

            print("---------------------------------------------------------------------------------------------")
            print("rho: %.2f, Sample Size: %d, Real MI: %.4f" % (rho, sample_size, real_mi))
            print("r=s=2: %.4f, r=s=4: %.4f, r=s=5: %.4f, r=s=10: %.4f" % 
                  (np.mean(rs_2_l), np.mean(rs_4_l), np.mean(rs_5_l), np.mean(rs_5_l)))

            rs_2_std.append(np.std(rs_2_l))
            rs_4_std.append(np.std(rs_4_l))
            rs_5_std.append(np.std(rs_5_l))
            rs_10_std.append(np.std(rs_5_l))

        rs_2.append(rs_2_std)
        rs_4.append(rs_4_std)
        rs_5.append(rs_5_std)
        rs_10.append(rs_10_std)

    generate_rs_table(results)

    all_std = [rs_2, rs_4, rs_5, rs_10]
    for i, _ in enumerate(["r=s=2", "r=s=4", "r=s=5", "r=s=10"]):

        plt.figure()
        plt.semilogx(sample_sizes, all_std[i][0], '-o', label=r'$\rho$ =' + f'{0.0}')
        plt.semilogx(sample_sizes, all_std[i][1], '-o', label=r'$\rho$ =' + f'{0.3}')
        plt.semilogx(sample_sizes, all_std[i][2], '-o', label=r'$\rho$ =' + f'{0.6}')
        plt.semilogx(sample_sizes, all_std[i][3], '-o', label=r'$\rho$ =' + f'{0.9}')
        plt.xlabel('$\log_{10}$ of sample size')

        if i == 0:
            plt.ylabel("std($\hat{I}_{CI}^{r=s=2}$)")
            plt.title("Standard deviation of MI estimator $I_{CI}$ with $r=s=2$")
        elif i == 1:
            plt.ylabel("std($\hat{I}_{CI}^{r=s=4}$)")
            plt.title("Standard deviation of MI estimator $I_{CI}$ with $r=s=4$")
        elif i == 2:
            plt.ylabel("std($\hat{I}_{CI}^{r=s=5}$)")
            plt.title("Standard deviation of MI estimator $I_{CI}$ with $r=s=5$")
        else:
            plt.ylabel("std($\hat{I}_{CI}^{r=s=10}$)")
            plt.title("Standard deviation of MI estimator $I_{CI}$ with $r=s=10$")

        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
