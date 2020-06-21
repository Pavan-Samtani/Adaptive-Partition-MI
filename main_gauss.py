
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats.stats import pearsonr   
from scipy.stats import chi2

from distributions import MultivariateNormal
from divergence_utils import kl_estimate
from partition import AdaptiveAlgorithm, NonAdaptivePartition, Plane


def main():

    sample_sizes = [250, 500, 1000, 2000, 10000]
    rhos = [0, 0.3, 0.6, 0.9]
    r = 2
    s = 2
    K = 50

    ci_mi_all_std = []
    na_mi_all_std = []
    ml_mi_all_std = []

    for rho in rhos:

        ci_mi_std = []
        na_mi_std = []
        ml_mi_std = []

        real_mi = - np.log(1 - rho ** 2) / 2

        for sample_size in sample_sizes:

            cov = np.array([[1., rho], [rho, 1.]])
            dist = MultivariateNormal(mean=np.zeros(2), cov=cov)

            ci_mi_l, ml_mi_l, na_mi_l = [], [], []

            delta = lambda x: chi2.ppf(0.97, x ** 2 - 1)

            for k in range(K):
                
                xy_sample = dist.sample(sample_size)

                plane = Plane(xy_sample)

                # Adaptive algorithm
                ci_mi_l.append(kl_estimate(plane, AdaptiveAlgorithm(xy_sample, delta, r, s).run()))

                na_mi_l.append(kl_estimate(plane, NonAdaptivePartition(xy_sample, bins=[50, 50]).run()))

                ml_mi_l.append(- np.log(1 - pearsonr(xy_sample[:, 0], xy_sample[:, 1])[0] ** 2) / 2)

            print("---------------------------------------------------------------------------------------------")
            print("rho: %.2f, Sample Size: %d, Real MI: %.4f" % (rho, sample_size, real_mi))
            print("Adaptive Partition MI: %.4f, NA Partition MI: %.4f, ML MI: %.4f" % 
                  (np.mean(ci_mi_l), np.mean(na_mi_l), np.mean(ml_mi_l)))

            ci_mi_std.append(np.std(ci_mi_l))
            na_mi_std.append(np.std(na_mi_l))
            ml_mi_std.append(np.std(ml_mi_l))

        ci_mi_all_std.append(ci_mi_std)
        na_mi_all_std.append(na_mi_std)
        ml_mi_all_std.append(ml_mi_std)

    all_std = [ci_mi_all_std, na_mi_all_std, ml_mi_all_std]
    for i, _ in enumerate(["CI", "NA", "ML"]):

        plt.figure()
        plt.semilogx(sample_sizes, all_std[i][0], '-o', label=r'$\rho$ =' + f'{0.0}')
        plt.semilogx(sample_sizes, all_std[i][1], '-o', label=r'$\rho$ =' + f'{0.3}')
        plt.semilogx(sample_sizes, all_std[i][2], '-o', label=r'$\rho$ =' + f'{0.6}')
        plt.semilogx(sample_sizes, all_std[i][3], '-o', label=r'$\rho$ =' + f'{0.9}')
        plt.xlabel('$\log_{10}$ of sample size')

        if i == 0:
            plt.ylabel("std($\hat{I}_{CI}$)")
            plt.title("Standard deviation of MI estimator $I_{CI}$")
        elif i == 1:
            plt.ylabel("std($\hat{I}_{NA}$)")
            plt.title("Standard deviation of MI estimator $I_{NA}$")
        else:
            plt.ylabel("std($\hat{I}_{ML}$)")
            plt.title("Standard deviation of MI estimator $I_{ML}$")

        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
