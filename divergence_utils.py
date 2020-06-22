
import numpy as np



def kl_estimate(plane, rectangles, log_base=np.exp(1)):

    mi_estimate = 0
    N = plane.sample.shape[0]

    ctr = 0

    for rect in rectangles:

        joint_n = len(rect.samples_inside)

        if joint_n == 0:
            continue

        ctr += joint_n

        xlim, ylim = rect.xlim, rect.ylim

        x_marg_n = plane.get_samples(xlim, plane.ylim).shape[0]
        y_marg_n = plane.get_samples(plane.xlim, ylim).shape[0]

        marg_n = x_marg_n * y_marg_n

        mi_estimate += joint_n * np.log(N * joint_n / (marg_n))

    mi_estimate /= (N * np.log(log_base))

    return mi_estimate
