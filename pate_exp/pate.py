# Copyright 2017 The 'Scalable Private Learning with PATE' Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Performs privacy analysis of GNMax with smooth sensitivity.
A script in support of the paper "Scalable Private Learning with PATE" by
Nicolas Papernot, Shuang Song, Ilya Mironov, Ananth Raghunathan, Kunal Talwar,
Ulfar Erlingsson (https://arxiv.org/abs/1802.08908).
Several flavors of the GNMax algorithm can be analyzed.
  - Plain GNMax (argmax w/ Gaussian noise) is assumed when arguments threshold
    and sigma2 are missing.
  - Confident GNMax (thresholding + argmax w/ Gaussian noise) is used when
    threshold, sigma1, and sigma2 are given.
  - Interactive GNMax (two- or multi-round) is triggered by specifying
    baseline_file, which provides baseline values for votes selection in Step 1.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys

sys.path.append('pate_data/')  # Main modules reside in the parent directory.

from absl import app
from absl import flags
import numpy as np
import core as pate
from scipy.stats import norm
import smooth_pate as pate_ss
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

flags.DEFINE_string('counts_file', None, 'Counts file.')
flags.DEFINE_string('baseline_file', None, 'File with baseline scores.')
flags.DEFINE_boolean('data_independent', False,
                     'Force data-independent bounds.')
flags.DEFINE_float('threshold', None, 'Threshold for step 1 (selection).')
flags.DEFINE_float('sigma1', None, 'Sigma for step 1 (selection).')
flags.DEFINE_float('sigma2', 40, 'Sigma for step 2 (argmax).')
flags.DEFINE_integer('queries', 100, 'Number of queries made by the student.')
flags.DEFINE_float('delta', 1e-5, 'Target delta.')
flags.DEFINE_float(
    'order', None,
    'Fixes a Renyi DP order (if unspecified, finds an optimal order from a '
    'hardcoded list).')
flags.DEFINE_integer(
    'teachers', None,
    'Number of teachers (if unspecified, derived from the counts file).')


flags.mark_flag_as_required('sigma2')


# generate high-confidence and low-confidence vote results
def _load_votes(n=500, d=10):

    #votes = np.load(counts_file_expanded)
    import pickle
    with open('cov_teachers_pred.pkl', 'rb') as f:
        votes = pickle.load(f)
    #votes = np.zeros([n, d])
    #for i in range(len(votes)):
    #    votes[i, 0]=50
    t = FLAGS.queries
    votes = votes.transpose()
    threshold = 90
    result = []
    for i in range(len(votes)):
        x = np.bincount(votes[i], minlength=10)
        if max(x)< 170 and max(x)>120:
            result.append(x)
        # if max(x) > 140:
        #    result.append(x)
    result = np.array(result)
    result = result *2
    result = result[:t]
    baseline = np.zeros_like(result)
    return result, baseline


def _count_teachers(votes):
  s = np.sum(votes, axis=1)
  num_teachers = int(max(s))
  print('num of teacher', num_teachers)
  if min(s) != num_teachers:
    raise ValueError(
        'Matrix of votes is malformed: the number of votes is not the same '
        'across rows.')
  return num_teachers


def _check_conditions(sigma, num_classes, orders):
  """Symbolic-numeric verification of conditions C5 and C6.
  The conditions on the beta function are verified by constructing the beta
  function symbolically, and then checking that its derivative (computed
  symbolically) is non-negative within the interval of conjectured monotonicity.
  The last check is performed numerically.
  """

  print('Checking conditions C5 and C6 for all orders.')
  sys.stdout.flush()
  conditions_hold = True

  for order in orders:
    cond5, cond6 = pate_ss.check_conditions(sigma, num_classes, order)
    conditions_hold &= cond5 and cond6
    if not cond5:
      print('Condition C5 does not hold for order =', order)
    elif not cond6:
      print('Condition C6 does not hold for order =', order)

  if conditions_hold:
    print('Conditions C5-C6 hold for all orders.')
  sys.stdout.flush()
  return conditions_hold


def _compute_rdp(votes, baseline, threshold, sigma1, sigma2, delta, orders,
                 data_ind):
  """Computes the (data-dependent) RDP curve for Confident GNMax."""
  rdp_cum = np.zeros(len(orders))
  rdp_sqrd_cum = np.zeros(len(orders))
  answered = 0

  for i, v in enumerate(votes):
    if threshold is None:
      logq_step1 = 0  # No thresholding, always proceed to step 2.
      rdp_step1 = np.zeros(len(orders))
    else:
      logq_step1 = pate.compute_logpr_answered(threshold, sigma1,
                                               v - baseline[i,])
      if data_ind:
        rdp_step1 = pate.compute_rdp_data_independent_threshold(sigma1, orders)
      else:
        rdp_step1 = pate.compute_rdp_threshold(logq_step1, sigma1, orders)

    if data_ind:
      rdp_step2 = pate.rdp_data_independent_gaussian(sigma2, orders)
    else:
      logq_step2 = pate.compute_logq_gaussian(v, sigma2)
      rdp_step2 = pate.rdp_gaussian(logq_step2, sigma2, orders)

    q_step1 = np.exp(logq_step1)
    rdp = rdp_step1 + rdp_step2 * q_step1
    # The expression below evaluates
    #     E[(cost_of_step_1 + Bernoulli(pr_of_step_2) * cost_of_step_2)^2]
    rdp_sqrd = (
        rdp_step1**2 + 2 * rdp_step1 * q_step1 * rdp_step2 +
        q_step1 * rdp_step2**2)
    rdp_sqrd_cum += rdp_sqrd

    rdp_cum += rdp
    answered += q_step1
    if ((i + 1) % 1000 == 0) or (i == votes.shape[0] - 1):
      rdp_var = rdp_sqrd_cum / i - (
          rdp_cum / i)**2  # Ignore Bessel's correction.
      eps_total, order_opt = pate.compute_eps_from_delta(orders, rdp_cum, delta)
      order_opt_idx = np.searchsorted(orders, order_opt)
      eps_std = ((i + 1) * rdp_var[order_opt_idx])**.5  # Std of the sum.
      print(
          'queries = {}, E[answered] = {:.2f}, E[eps] = {:.3f} (std = {:.5f}) '
          'at order = {:.2f} (contribution from delta = {:.3f})'.format(
              i + 1, answered, eps_total, eps_std, order_opt,
              -math.log(delta) / (order_opt - 1)))
      sys.stdout.flush()

    _, order_opt = pate.compute_eps_from_delta(orders, rdp_cum, delta)

  return order_opt


def _find_optimal_smooth_sensitivity_parameters(
    votes, baseline, num_teachers, threshold, sigma1, sigma2, delta, ind_step1,
    ind_step2, order):
  """Optimizes smooth sensitivity parameters by minimizing a cost function.
  The cost function is
        exact_eps + cost of GNSS + two stds of noise,
  which captures that upper bound of the confidence interval of the sanitized
  privacy budget.
  Since optimization is done with full view of sensitive data, the results
  cannot be released.
  """
  rdp_cum = 0
  answered_cum = 0
  ls_cum = 0

  # Define a plausible range for the beta values.
  betas = np.arange(.1 / order, .495 / order, .01 / order)
  #betas = np.arange(0.1 / order, .2 / order, .01 / order)
  cost_delta = math.log(2 / delta) / (order - 1)

  for i, v in enumerate(votes):
    if threshold is None:
      log_pr_answered = 0
      rdp1 = 0
      ls_step1 = np.zeros(num_teachers)
    else:
      log_pr_answered = pate.compute_logpr_answered(threshold, sigma1,
                                                    v - baseline[i,])
      if ind_step1:  # apply data-independent bound for step 1 (thresholding).
        rdp1 = pate.compute_rdp_data_independent_threshold(sigma1, order)
        ls_step1 = np.zeros(num_teachers)
      else:
        rdp1 = pate.compute_rdp_threshold(log_pr_answered, sigma1, order)
        ls_step1 = pate_ss.compute_local_sensitivity_bounds_threshold(
            v - baseline[i,], num_teachers, threshold, sigma1, order)

    pr_answered = math.exp(log_pr_answered)
    answered_cum += pr_answered

    if ind_step2:  # apply data-independent bound for step 2 (GNMax).
      rdp2 = pate.rdp_data_independent_gaussian(sigma2, order)
      ls_step2 = np.zeros(num_teachers)
    else:
      logq_step2 = pate.compute_logq_gaussian(v, sigma2)
      # logq_step2 denotes the probability that the output isn't the gt output. smaller the better.
      #print('log_q', logq_step2)
      rdp2 = pate.rdp_gaussian(logq_step2, sigma2, order)
      # Compute smooth sensitivity.
      ls_step2 = pate_ss.compute_local_sensitivity_bounds_gnmax(
          v, num_teachers, sigma2, order)

    rdp_cum += rdp1 + pr_answered * rdp2
    ls_cum += ls_step1 + pr_answered * ls_step2  # Expected local sensitivity.

    if ind_step1 and ind_step2:
      # Data-independent bounds.
      cost_opt, beta_opt, ss_opt, sigma_ss_opt = None, 0., 0., np.inf
    else:
      # Data-dependent bounds.
      cost_opt, beta_opt, ss_opt, sigma_ss_opt = np.inf, None, None, None

      for beta in betas:
        ss = pate_ss.compute_discounted_max(beta, ls_cum)

        # Solution to the minimization problem:
        #   min_sigma {order * exp(2 * beta)/ sigma^2 + 2 * ss * sigma}
        sigma_ss = ((order * math.exp(2 * beta)) / ss)**(1 / 3)
        cost_ss = pate_ss.compute_rdp_of_smooth_sensitivity_gaussian(
            beta, sigma_ss, order)

        # Cost captures exact_eps + cost of releasing SS + two stds of noise.
        cost = rdp_cum + cost_ss + 2 * ss * sigma_ss
        #cost = rdp_cum + 2*cost_ss + 2 * ss * sigma_ss + np.sqrt(np.log(1/delta))*ss*sigma_ss
        if cost < cost_opt:
          cost_opt, beta_opt, ss_opt, sigma_ss_opt = cost, beta, ss, sigma_ss

    if ((i + 1) % 100 == 0) or (i == votes.shape[0] - 1):
      eps_before_ss = rdp_cum + cost_delta
      eps_with_ss = (
          eps_before_ss + pate_ss.compute_rdp_of_smooth_sensitivity_gaussian(
              beta_opt, sigma_ss_opt, order))
      print('{}: E[answered queries] = {:.1f}, RDP at {} goes from {:.3f} to '
            '{:.3f} +/- {:.3f} (ss = {:.4}, beta = {:.4f}, sigma_ss = {:.3f})'.
            format(i + 1, answered_cum, order, eps_before_ss, eps_with_ss,
                   ss_opt * sigma_ss_opt, ss_opt, beta_opt, sigma_ss_opt))
      sys.stdout.flush()

  # Return optimal parameters for the last iteration.
  # ss_opt denotes the smooth sensitivity
  return beta_opt, ss_opt, sigma_ss_opt, order, eps_before_ss, eps_with_ss


####################
# HELPER FUNCTIONS #
####################


def _is_data_ind_step1(num_teachers, threshold, sigma1, orders):
  if threshold is None:
    return True
  return np.all(
      pate.is_data_independent_always_opt_threshold(num_teachers, threshold,
                                                    sigma1, orders))


def _is_data_ind_step2(num_teachers, num_classes, sigma, orders):
  return np.all(
      pate.is_data_independent_always_opt_gaussian(num_teachers, num_classes,
                                                   sigma, orders))






def main(argv):


  #pate_low = 'pate_high.pkl'
  #if os.path.exists(pate_low):
  #  plot(pate_low)
  #  return


  if (FLAGS.threshold is None) != (FLAGS.sigma1 is None):
    raise ValueError(
        '--threshold flag and --sigma1 flag must be present or absent '
        'simultaneously.')

  if FLAGS.order is None:
    # Long list of orders.
    orders = np.concatenate((np.arange(2, 100 + 1, .5),
                             np.logspace(np.log10(100), np.log10(500),
                                         num=100)))
    # Short list of orders.
    # orders = np.round(
    #     np.concatenate((np.arange(2, 50 + 1, 1),
    #                     np.logspace(np.log10(50), np.log10(1000), num=20))))
  else:
    orders = np.array([FLAGS.order])

  votes, baseline = _load_votes()

  if FLAGS.teachers is None:
    num_teachers = _count_teachers(votes)
  else:
    num_teachers = FLAGS.teachers

  num_classes = votes.shape[1]

  sigma_list = [1.3 ** x for x in np.arange( 10, 15, 0.2)]
  #sigma_list = [1.2 ** x for x in np.arange(10, 30, 0.5)]
  eps_ind = []
  eps_dep = []
  eps_ptr = []
  ptr_std = []
  print('sigma_list', sigma_list)
  for sigma in sigma_list:
    eps_dependent, eps_p, eps_p_std = data(votes, baseline, orders, num_teachers, num_classes, sigma)
    eps_ptr.append(eps_p)
    eps_dep.append(eps_dependent)
    ptr_std.append(eps_p_std)
    t = FLAGS.queries
    eps_independent = 2*np.sqrt(t*np.log(1/FLAGS.delta))/sigma + t/(sigma**2)
    print('eps_indendent', eps_independent, 'eps_ptr', eps_p, 'eps_dependent', eps_dependent)
    eps_ind.append(eps_independent)

  print('eps_dependent', eps_dep)
  import pickle
  #pate_low = 'pate_high.pkl'
  pate_low = 'pate_data/pate_low.pkl'

  result = {}
  result['eps_ptr'] = eps_ptr
  result['ptr_std'] = ptr_std
  result['sigma_list'] = sigma_list
  result['eps_ind'] = eps_ind
  result['eps_dep'] = eps_dep
  with open(pate_low, 'wb') as f:
      pickle.dump(result, f)


  #plt.show()



def plot(save_path):
    import pickle
    with open(save_path, 'rb') as f:
        result = pickle.load(f)
    eps_ptr = result['eps_ptr']
    eps_dep = result['eps_dep']
    eps_ind = result['eps_ind']
    ptr_std = result['ptr_std']
    sigma_list = result['sigma_list']
    print(sigma_list)

    eps_ptr = np.array(eps_ptr)
    ptr_std = np.array(ptr_std)*0.5
    plt.figure(num=0, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(sigma_list, eps_ind, 'g', linewidth=2)
    plt.plot(sigma_list, eps_ptr, 'b--', linewidth=2)
    plt.fill_between(sigma_list, eps_ptr - ptr_std, eps_ptr + ptr_std,
                     color='y', alpha=0.5)
    plt.fill_between(sigma_list[:-11], eps_ptr[:-11] + ptr_std[:-11], eps_ind[:-11],
                    color='m', alpha=0.1)
    plt.plot(sigma_list, eps_dep, 'D-', color='pink', linewidth=2)
    # plt.plot(sigma_list, eps_ptr, 'b--', linewidth=2)
    plt.legend(
        [r' Gaussian mechanism', 'PATE-PTR ($\hat{\epsilon}+\epsilon_{\sigma_1}$)', 'data-dependent DP (non-private)'], loc='best', fontsize=20)
    plt.grid(True)
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel(r'Noise scale $\sigma_1$', fontsize=23)
    plt.ylabel(r'$\epsilon$', fontsize=23)
    plt.savefig("pate_low.pdf", bbox_inches='tight')

plot('pate_data/pate_low.pkl')

def data(votes, baseline, orders, num_teachers, num_classes, sigma2):
    order = _compute_rdp(votes, baseline, FLAGS.threshold, FLAGS.sigma1,
                         sigma2, FLAGS.delta, orders,
                         FLAGS.data_independent)

    ind_step1 = _is_data_ind_step1(num_teachers, FLAGS.threshold, FLAGS.sigma1,
                                   order)

    ind_step2 = _is_data_ind_step2(num_teachers, num_classes, sigma2, order)
    # best choice is order = 22.
    #order = 22.for high, 15 for low
    order = 22.
    if FLAGS.data_independent or (ind_step1 and ind_step2):
        print('Nothing to do here, all analyses are data-independent.')
        return

    if not _check_conditions(FLAGS.sigma2, num_classes, [order]):
        return  # Quit early: sufficient conditions for correctness fail to hold.

    beta_opt, ss_opt, sigma_ss_opt, order, eps_before_ss, eps_with_ss = _find_optimal_smooth_sensitivity_parameters(
        votes, baseline, num_teachers, FLAGS.threshold, FLAGS.sigma1,
       sigma2, FLAGS.delta, ind_step1, ind_step2, order)

    # beta_opt is beta, ss_opt denotes the smooth sensitivity
    print('Optimal beta = {:.4f}, E[SS_beta] = {:.4}, sigma_ss = {:.2f}'.format(
        beta_opt, ss_opt, sigma_ss_opt))
    eps_private, ptr_std = ptr(beta_opt, ss_opt, sigma_ss_opt, order, eps_before_ss, eps_with_ss)
    return eps_before_ss, eps_private, ptr_std

def ptr(beta, ss, sigma_ss, order, eps_before_ss, eps_with_ss):

    sigma2 = 0.5*sigma_ss

    # mu = log(ss) + beta*N(0,sigma2^2) + np.sqrt(log(2/delta_2))*sigma_2*beta
    delta2 = FLAGS.delta/2
    mu = np.log(ss) + beta * np.random.normal(scale = sigma2) + np.sqrt(2*np.log(2./delta2)) * sigma2 *beta
    loss_sigma2 = order/(2*sigma2**2)
    eps_priv_list = []
    print('mu', np.exp(mu),'exp_term', sigma_ss*np.sqrt(np.log(2/delta2))*np.exp(mu))
    for t in range(500):
        eps_private = eps_with_ss + loss_sigma2 +ss*np.random.normal(scale=sigma_ss) + sigma_ss*np.sqrt(np.log(2/delta2))*np.exp(mu)
        eps_priv_list.append(eps_private)
    eps_priv_list = np.array(eps_priv_list)
    print('eps_private', eps_private, 'loss_sigma2', loss_sigma2)
    print('estimated alpha', 2*np.log(2/delta2)/eps_before_ss +1)
    return  np.mean(eps_priv_list), np.std(eps_priv_list)


"""
if __name__ == '__main__':
    app.run(main)
    
"""