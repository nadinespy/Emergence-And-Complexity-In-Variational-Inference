# %%
import numpy as np
import scipy.linalg as la
from itertools import product
from matplotlib import pyplot as plt
import os
import pandas as pd
import pickle

basedir = '/media/nadinespy/NewVolume1/work/phd/projects/mec_var_inf/mec_var_inf/src'
main_path = os.path.join(basedir, 'scripts')

import sys
sys.path.append(basedir)
import mec_var_inf as mec

path_out1 = '/media/nadinespy/NewVolume1/work/phd/projects/mec_var_inf/mec_var_inf/results/analyses/'
path_out2 = '/media/nadinespy/NewVolume1/work/phd/projects/mec_var_inf/mec_var_inf/results/plots/'

# %%
# -------------------------------------------------------------------------------
# ADJUST PARAMETERS
# -------------------------------------------------------------------------------

# FIXED PARAMETERS
gamma = 0.01                # integration step for discrete case
errvar = 0.01               # sampling error
spread_factor = 0           # variance around mean noise correlation

# create arrays of rho, weights, time-lags, mean noise correlations, and system sizes to test
n_rho = 100
all_rho = np.arange(0, n_rho)/n_rho                          # creates array [0, 0.01, ..., 0.99]

# -------------------------------------------------------------------------------
# CHANGE PAIR OF PARAMETERS (TRUE CORR & SOMETHING ELSE)

filename = 'results_mec_var_inf_corr_time_lag'

n_weights = 1
#all_weights = np.linspace(0.000001, 1, n_weights)            # creates array [0, 0.1, 0.2, ..., 1.0]
all_weights = [1]

n_time_lags = 11
all_time_lags = np.linspace(1, 20, n_time_lags, dtype=int)    
#all_time_lags = [1]

n_mean_noise_corr = 1
#all_mean_noise_corr = np.linspace(0, 0.9, n_mean_noise_corr) # average correlation of noise
all_mean_noise_corr = [0.0]

n_system_sizes = 1
#all_n_var = np.linspace(2, 10, n_system_sizes, dtype=int)    # system size
all_n_var = [2]

# %%
# %%

def get_results_for_all_params(rho, weight, time_lag, mean_noise_corr, n_var, gamma, \
                               errvar, spread_factor):
    """docstring"""

    # get true means, and true (weighted) & mean-field covariance
    np.random.seed(10)
    true_means = np.random.randn(n_var)                 # means of true distribution
    var_means = true_means                              # in the limit, variational and true means 
                                                        # will be the same

    # covariance of true distribution
    true_cov = mec.get_true_cov(rho, n_var)
    
    # inverse of covariance, weighted inverse of covariance & inverse of mean field covariance
    inv_true_cov, weighted_inv_true_cov, mean_field_inv_true_cov = \
        mec.get_approx_cov(true_cov, weight) 

    # 'c' stands for weighted_inv_true_cov (it's called 'C' in the paper), 
    # so 'inv_c_noise_cov_c' is equal to the inverse of (C times the noise covariance times C)
    inv_c_noise_cov_c = mec.get_inv_c_noise_cov_c(weighted_inv_true_cov, gamma, errvar, \
                                  mean_noise_corr=mean_noise_corr, \
                                    spread_factor=spread_factor, seed=None)

    # same-time covariance
    identity = np.eye(n_var)
    steady_state_same_time_COV = np.linalg.inv(inv_c_noise_cov_c - 
                                               (identity - gamma * weighted_inv_true_cov) 
                                               @ inv_c_noise_cov_c 
                                               @ (identity - gamma * weighted_inv_true_cov))
    # time-lagged covariance
    time_lagged_COV = (identity - gamma * weighted_inv_true_cov) @ steady_state_same_time_COV

    time_lagged_COV = steady_state_same_time_COV.copy()
    for time_lag in range(time_lag):
        time_lagged_COV = (identity - gamma * weighted_inv_true_cov) @ time_lagged_COV

    # compute phi measures for minimum bipartition
    min_bipartition, phi, phi_corrected, double_red_mmi = \
        mec.get_phi_for_min_bipartition(steady_state_same_time_COV, time_lagged_COV)

    part1_indices, part2_indices = min_bipartition

    # FOR DIAGNOSTICS 
    # compute conditional covariances
    time_lagged_COND_COV_FULL, time_lagged_COND_COV_PART1, time_lagged_COND_COV_PART2 = \
        mec.get_cond_covs(steady_state_same_time_COV, time_lagged_COV, part1_indices, part2_indices)

    # compute entropy measures
    entropy_PRESENT_PART1, entropy_PRESENT_PART2, entropy_PRESENT_FULL, mi_PAST_PRESENT_PART1, \
        mi_PAST_PRESENT_PART2, mi_PAST_PRESENT_FULL, mi_SAME_TIME_FULL = \
            mec.get_entropies(steady_state_same_time_COV, time_lagged_COND_COV_FULL, \
                          time_lagged_COND_COV_PART1, time_lagged_COND_COV_PART2, \
                            part1_indices, part2_indices)
    
    # compute KL divergence
    kl_div = mec.get_kl_div(weighted_inv_true_cov, mean_field_inv_true_cov,
                           true_means, var_means, steady_state_same_time_COV)
    
    full_time_lagged_COV = np.block([[steady_state_same_time_COV, time_lagged_COV],
                                 [time_lagged_COV.T, steady_state_same_time_COV]])

    [phiid,
     emergence_capacity_phiid,
     downward_causation_phiid,
     synergy_phiid,
     transfer_phiid,
     phi_phiid,
     phi_corrected_phiid] = mec.get_phiid_analytical(full_time_lagged_COV, 'mmi')


    df_temp = pd.DataFrame({
                            'gamma': [gamma],
                            'error_variance': [errvar],
                            'spread_factor': [spread_factor],
                            'correlation': [rho],
                            'time_lag': [time_lag],
                            'weight': [weight],
                            'mean_noise_corr': [mean_noise_corr],
                            'n_var': [n_var],
                            'phi': [phi],
                            'phi_corrected': [phi_corrected],
                            'kldiv': [kl_div],
                            'double_red_mmi': [double_red_mmi],
                            'rtr': [phiid['rtr']],
                            'rtx': [phiid['rtx']],
                            'rty': [phiid['rty']],
                            'rts': [phiid['rts']],
                            'xtr': [phiid['xtr']],
                            'xtx': [phiid['xtx']],
                            'xty': [phiid['xty']],
                            'xts': [phiid['xts']],
                            'ytr': [phiid['ytr']],
                            'ytx': [phiid['ytx']],
                            'yty': [phiid['yty']],
                            'yts': [phiid['yts']],
                            'str': [phiid['str']],
                            'stx': [phiid['stx']],
                            'sty': [phiid['sty']],
                            'sts': [phiid['sts']],
                            'synergy_phiid': [synergy_phiid],
                            'transfer_phiid': [transfer_phiid],
                            'emergence_capacity_phiid': [emergence_capacity_phiid],
                            'downward_causation_phiid': [downward_causation_phiid],
                            'phi_phiid': [phi_phiid],
                            'phi_corrected_phiid': [phi_corrected_phiid],
                            'entropy_PRESENT_PART1': [entropy_PRESENT_PART1],
                            'entropy_PRESENT_PART2': [entropy_PRESENT_PART2],
                            'entropy_PRESENT_FULL' : [entropy_PRESENT_FULL],
                            'mi_PAST_PRESENT_PART1': [mi_PAST_PRESENT_PART1],
                            'mi_PAST_PRESENT_PART2': [mi_PAST_PRESENT_PART2],
                            'mi_PAST_PRESENT_FULL' : [mi_PAST_PRESENT_FULL],
                            'mi_SAME_TIME_FULL': [mi_SAME_TIME_FULL]
                            })

    print('rho: ', rho)
    return df_temp

def mec_var_inf(all_rho, all_weights, all_time_lags, all_mean_noise_corr, \
                                 all_n_var, gamma, errvar, spread_factor):
    results_df = []

    # storing each dataframe in a list
    results = [get_results_for_all_params(rho, weight, time_lag, mean_noise_corr, n_var, \
                                          gamma, errvar, spread_factor)
               for rho, weight, time_lag, mean_noise_corr, n_var, in
               product(all_rho, all_weights, all_time_lags, all_mean_noise_corr, all_n_var)]

    # putting dataframe rows into one a single dataframe
    results_df = pd.concat(results, ignore_index=True)

    return results_df


# variable name: results_df_[correlation]_[error_variance]_[time-lag]
results_df = mec_var_inf(all_rho, all_weights, all_time_lags, all_mean_noise_corr, \
                         all_n_var, gamma, errvar, spread_factor)

results_df.to_pickle(os.path.join(path_out1, filename + '.pkl'))

# results_df = open(path_out1+r'mec_var_inf_discrete_steady_state_df.pkl', 'rb')
# results_df = pickle.load(results_df)






