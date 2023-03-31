"""
private hyper-parameter tuning (enable data-dependent analysis being applied)
The goal is to estimate private mean with smooth sensitivity based analysis.
smooth sensitivity will affect privacy cost?
1. Generate n*d N(0, sigma**2) data
2. Consider a list of hyper-parameter (sigma, t), t is the smooth sensitivity.
3. Compute smooth sensitivity of each of them.
4. Privately release semoo
"""
import numpy as np



def compute_ss(beta, x, t_max = 10):
    """
    Return the smooth sensitivity of function f with parameter beta.
    consider replacing strategy

    :param beta:
    :param f:
    :return:
    """
    n = len(x)
    m = int(len(x+1)/2)
    smooth = 0
    for k in range(t_max):
        # compute the local sensitivity at y, where y is constructed by replacing t data from x.

        s1 =[x[int(m+t)] -x[int(m+t-k-1)] for t in range(int(k+1))]
        smooth = max(max(s1)* np.exp(-k*beta), smooth)
    return smooth




def main(n, eps, delta, tunc_t =50):
    """
    Return the variance between the ground-truth median and the privately sanitized median.
    :param n: number of data
    :param trunc_t: Assume data is bounded by [0, trunc_t]
    :param eps: privacy parameter
    :param delta: privacy parameter
    :return:
    """
    x = np.random.normal(0, 1, 1000)
    keep_idx = np.where( x<50)
    x = x[keep_idx[0]]

    print('number of data', len(x))
    np.sort(x)
    m = int(len(x)/2)
    gt_median = x[m]
    beta = eps/(4*(1 + np.log(2/delta)))
    alpha = eps/(5*np.sqrt(2*np.log(2/delta)))
    # consider Gaussian mechanism with smooth sensitivity based calibration
    # A(x) = f(x) + s(x)/alpha *Z， where Z is normal distribution
    ss = compute_ss(beta, x)
    gau_noise = np.random.normal(0,1)
    ss_result = x[m] + ss/alpha*gau_noise
    print('smooth sensitivity', ss)
    # Compute standard Gaussian mechanism with L2 sensitivity 100
    print('ss_result', ss_result, 'gt_result', gt_median)

main(5000, 1.0, 1e-4)










