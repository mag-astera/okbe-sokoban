import numpy as np
import warnings
import queue
import itertools
import scipy as sp
import scipy.sparse
import sparse
import timeit
import cProfile
from line_profiler import LineProfiler
from scipy.sparse import csr_matrix as csr
from scipy import signal
from itertools import product

import TGMDP_methods
import config
import utilities
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import csc_matrix as csc
import math
profiler = LineProfiler()

# import torch  # removed: never used (the `f.` refs here are a local var, not torch.nn.functional); cost ~1.2s at import
# import tensorflow as tf

def time_to_event_multiply(eta, TEC_mat):
    # The TEC_mat is Time X State and the time dimension should be greater than or equal to the eta time dimension. Though this function will work if that is not true.
    internal_max_time = TEC_mat.shape[0]
    eta_max_time = eta.shape[2]
    min_max_time = min(internal_max_time, eta_max_time)
    multed_mat = np.dot(eta[:, :, range(min_max_time)], TEC_mat[range(min_max_time), :])
    no_violation = False
    if np.all(multed_mat == eta):
        no_violation = True

    return(multed_mat, no_violation)

def restrict_to_state(f, x_g, goal_var_ind):
    # Restrict the availability function to goal state x_g for goal variable goal_var_ind
    if len(f.shape) == 4:
        f_g = np.zeros(f.shape[1:])
        f_g[:, x_g, :] = f[goal_var_ind, :, x_g, :]
    if len(f.shape) == 3:
        f_g = np.zeros(f.shape[1:])
        f_g[:, x_g] = f[goal_var_ind, :, x_g]
    return f_g

def restrict_to_state_time(f, x_g, goal_var_ind, t):
    # Restrict the availability function to goal state x_g and time t.
    f_gt = np.zeros(f.shape[1:])
    f_gt[:, x_g, t] = f[goal_var_ind, :, x_g, t]
    return f_gt

def evolve_state_vecs(xt_mat, internal_state_vecs, eta, p_ax_x, secondary_omegas, p_v_list, pi, pi_input, g_to_action_map = []):
    # Written for deterministic operators.  Can be modified for stochastic by representing t_d as probability vector.
    # The first vector in state_vecs must be the low-level state-vector in matrix form, [X,T]
    # g_to_secondary_map maps the goal variable to a task action on the secondary state-space
    # P_x is the low-level operator
    # P_y is an internal state transition operator
    # T_f = len(t_vec)
    # xt_mat = np.outer(x_vec, t_vec)
    t_vec_s = xt_mat.sum(axis=0)

    xt_mat_tf = np.tensordot(xt_mat, eta[pi_input, :, :, :, :], axes=([0, 1], [0, 1]))
    t_vec_tf = xt_mat_tf.sum(axis=0)
    x_vec_tf = xt_mat_tf.sum(axis=1)
    t_s = np.where(t_vec == 1)[0].item()
    xt_inds = np.where(xt_mat_tf == 1)
    x_tf = xt_inds[0][0]
    t_f = xt_inds[1][0]

    x_tf_vec = xt_mat_tf[:, t_f].squeeze()
    next_x_vec = np.dot(x_tf_vec, p_ax_x[pi[x_tf, t_f], :, :].squeeze())
    next_x = np.where(next_x_vec==1)[0].item()
    next_t_vec = onehot(t_f + 1, len(t_vec))
    # xt_mat_prime = np.outer(onehot(x_prime, xt_mat.shape[0]), onehot(t_prime, xt_mat.shape[1]))
    internal_update = []

    t_d = t_f-t_s
    for i in range(len(secondary_omegas)):
        # internal_state_vec = np.where(state_vecs[i+1] == 1)[0]
        internal_state_vec = internal_state_vecs[i]
        internal_state_tf = np.dot(internal_state_vec, secondary_omegas[i][t_d,:,:])
        # One-step update
        next_internal_state = np.dot(internal_state_tf, p_v_list[i][g_to_action_map[next_x][i], :, :])
        internal_update.append(next_internal_state)

    updated_states = [next_x_vec, next_t_vec]
    updated_states.extend(internal_update)
    return updated_states

def evolve_state_vecs_deterministic(xt_mat, internal_state_vecs, internal_ss_list, eta, p_axt_x, secondary_omegas, pi_ind, pi, f, g_to_secondary_map = []):
    # Assumes Deterministic T-Ops.  This state_vector update doesn't use tensordot, which is expensive rather it just indexes the
    # correct 'row' and searches for the 1 entry in the next-state indicies and returns that index.
    # Written for deterministic operators.  Can be modified for stochastic by representing t_d as probability vector.
    # The first vector in state_vecs must be the low-level state-vector in matrix form, [X,T]
    # g_to_secondary_map maps the goal variable to a task action on the secondary state-space

    internal_states = [np.where(isv == 1)[0][0] for isv in internal_state_vecs]
    t_vec = xt_mat.sum(axis=0)
    T_f = len(t_vec)
    t_s = np.where(t_vec == 1)[0].item()
    x_vec = xt_mat.sum(axis=1)
    x_s = np.where(x_vec == 1)[0].item()
    xt_mat_tf = np.tensordot(xt_mat, eta[pi_ind, :, :, :, :], axes=([0, 1], [0, 1]))
    t_vec_tf = xt_mat_tf.sum(axis=0)
    x_vec_tf = xt_mat_tf.sum(axis=1)

    xt_inds = np.where(xt_mat_tf == 1)
    x_tf = xt_inds[0][0]
    t_f = xt_inds[1][0]

    mode_fail = False

    # xt_mat = xt_mat
    # x,t = np.where(xt_mat == 1)
    x_tf, t_f = np.where(eta[pi_ind,x_s,t_s,:,:])
    internal_update = []
    t_d = t_f[0]-t_s
    modes = np.array([int(ss.states_to_mode_dict[internal_states[i]]) for i, ss in enumerate(internal_ss_list)], dtype=int)
    mode_switch_times = [ss.time_to_mode_switch_dict[modes[i]][internal_states[i]] for i, ss in enumerate(internal_ss_list)]
    if not np.any(mode_switch_times < t_d):
        # Perform One-step Update
        x_tfp_vec = np.dot(x_vec_tf, p_axt_x[pi[x_tf, t_f], :, t_f, :])
        x_tfp = np.where(x_tfp_vec == 1)[1]  # One-step Update of Low-Level State
        t_fp = t_f + 1  # One-step Update of Time
        t_fp_vec = onehot(t_fp, T_f)

        xt_mat_tfp = np.outer(x_tfp_vec, t_fp_vec)

        for i in range(len(internal_ss_list)):
            # internal_state_vec = np.where(state_vecs[i+1] == 1)[0]
            ss = internal_ss_list[i]
            alpha_g = np.where(f[:, pi[x_tf, t_f], x_tf, t_f])[0][0]
            action_ind = alpha_g - 1 # This is due to the fact that
            if ss.action_type_ind == action_ind:
                alpha_g_val = ss.task_action
            else:
                alpha_g_val = ss.null_action

            # alpha_g = np.min([alpha_g[0], 1]) # This is a temporary hack which sets the action to 1 if greater than or equal to 1 because the operator P_a_s_s only has dimensions for null and "active" actions, not a specific index for one of many possible "active" actions.
            v = np.where(internal_state_vecs[i] == 1)
            v_tf_vec = secondary_omegas[i][t_d, v, :].squeeze()
            v_tf_onestep = np.dot(v_tf_vec, internal_ss_list[i].P_a_s_s[alpha_g_val,...])
            # v_new =
            internal_update.append(v_tf_onestep)
    else:
        xt_mat_tfp = xt_mat,
        internal_update = internal_state_vecs
        mode_fail = True

    # updated_states = [xt_mat_updated]
    # updated_states.extend(internal_update)
    return xt_mat_tfp, internal_update, mode_fail


def evolve_state_vecs_deterministic_fast(x_vec, t_vec, internal_state_vecs, internal_ss_list, eta, p_axt_x, secondary_omegas, pi_ind, pi, f, g_to_secondary_map = []):
    # This is called "fast" because instead of computing the tensor product, it simply finds variables corresponding to the determinsic state-time and produces the tensor slice corresponding to the final state-time.
    # Assumes Deterministic T-Ops.  This state_vector update doesn't use tensordot, which is expensive rather it just indexes the
    # correct 'row' and searches for the 1 entry in the next-state indicies and returns that index.
    # Written for deterministic operators.  Can be modified for stochastic by representing t_d as probability vector.
    # The first vector in state_vecs must be the low-level state-vector in matrix form, [X,T]
    # g_to_secondary_map maps the goal variable to a task action on the secondary state-space

    internal_states = [np.where(isv == 1)[0][0] for isv in internal_state_vecs]
    # t_vec = xt_mat.sum(axis=0)
    T_f = len(t_vec)
    t_s = np.where(t_vec == 1)[0].item()
    # x_vec = xt_mat.sum(axis=1)
    x_s = np.where(x_vec == 1)[0].item()
    xt_mat_tf = eta[pi_ind, x_s, t_s, :, :]
    t_vec_tf = xt_mat_tf.sum(axis=0)
    x_vec_tf = xt_mat_tf.sum(axis=1)

    xt_inds = np.where(xt_mat_tf == 1)
    x_tf = xt_inds[0][0]
    t_f = xt_inds[1][0]

    mode_fail = False

    # xt_mat = xt_mat
    # x,t = np.where(xt_mat == 1)
    x_tf, t_f = np.where(eta[pi_ind, x_s, t_s, :, :])
    internal_update = []
    t_d = t_f[0]-t_s
    partial_modes = np.array([int(ss.states_to_mode_dict[internal_states[i]]) for i, ss in enumerate(internal_ss_list)], dtype=int)  # A partial mode is the mode e_y mapped by a single element y of a state-vector s=(w,y,...,z)

    # This if-else statement is the zeta function specific to SPA's ontology so that death states map r=(w,y,z,...) to the death mode. It will need to be generalized in the future for arbitrary mode-functions.
    if 0 in partial_modes:
        # Death mode (perhaps switch this to 0)
        mode = 0
        low_level_action = 0
    else:
        # Normal mode
        mode = 1
        low_level_action = pi[x_tf, t_f]
    mode_switch_times = [ss.time_to_mode_switch_dict[partial_modes[i]][internal_states[i]] for i, ss in enumerate(internal_ss_list)]

    # This is technically correct for SPA, but it should be generalized (above) so that the death mode is selected to update with eta_{death}.
    if not np.any(mode_switch_times < t_d) and mode != 0:
        # Policy call is successful and we perform a one-step Update
        x_tfp_vec = np.dot(x_vec_tf, p_axt_x[low_level_action, :, t_f, :])
        x_tfp = np.where(x_tfp_vec == 1)[1]  # One-step Update of Low-Level State
        t_fp = t_f + 1  # One-step Update of Time
        t_fp_vec = onehot(t_fp, T_f)

        # xt_mat_tfp = np.outer(x_tfp_vec, t_fp_vec)

        for i in range(len(internal_ss_list)):
            # internal_state_vec = np.where(state_vecs[i+1] == 1)[0]
            ss = internal_ss_list[i]
            alpha_g = np.where(f[:, low_level_action, x_tf, t_f])[0][0]
            action_ind = alpha_g
            if ss.action_type_ind == action_ind:
                alpha_g_val = ss.task_action
            else:
                alpha_g_val = ss.null_action

            # alpha_g = np.min([alpha_g[0], 1]) # This is a temporary hack which sets the action to 1 if greater than or equal to 1 because the operator P_a_s_s only has dimensions for null and "active" actions, not a specific index for one of many possible "active" actions.
            v = np.where(internal_state_vecs[i] == 1)
            v_tf_vec = secondary_omegas[i][t_d, v, :].squeeze()
            v_tf_onestep = np.dot(v_tf_vec, internal_ss_list[i].P_a_s_s[alpha_g_val, ...])
            # v_new =
            internal_update.append(v_tf_onestep.squeeze())
    elif mode == 0:
        # Policy call fails and we return the input states and time.
        x_tfp_vec = np.dot(x_vec, p_axt_x[low_level_action, :, t_f, :])
        t_fp_vec = onehot(t_s + 1, T_f)
        # xt_mat_tfp = np.outer(x_tfp_vec, t_fp_vec)
        # internal_update = internal_state_vecs
        for i in range(len(internal_ss_list)):
            # internal_state_vec = np.where(state_vecs[i+1] == 1)[0]
            ss = internal_ss_list[i]
            alpha_g = np.where(f[:, low_level_action, x_tf, t_f])[0][0]
            action_ind = alpha_g - 1 # This is due to the fact that
            if ss.action_type_ind == action_ind:
                alpha_g_val = ss.task_action
            else:
                alpha_g_val = ss.null_action

            # alpha_g = np.min([alpha_g[0], 1]) # This is a temporary hack which sets the action to 1 if greater than or equal to 1 because the operator P_a_s_s only has dimensions for null and "active" actions, not a specific index for one of many possible "active" actions.
            v = np.where(internal_state_vecs[i] == 1)
            v_tf_vec = secondary_omegas[i][t_d, v, :].squeeze()
            v_tf_onestep = np.dot(v_tf_vec, internal_ss_list[i].P_a_s_s[alpha_g_val, ...])
            # v_new =
            internal_update.append(v_tf_onestep.squeeze())
    else:
        x_tfp_vec = x_vec
        t_fp_vec = t_vec
        internal_update = internal_state_vecs
        mode_fail = True

    # updated_states = [xt_mat_updated]
    # updated_states.extend(internal_update)
    return x_tfp_vec, t_fp_vec, internal_update, mode_fail


def evolve_state_vecs_deterministic_fast(x_vec, t_vec, internal_state_vecs, internal_ss_list, eta, p_axt_x, secondary_omegas, pi_ind, pi, f, g_to_secondary_map = []):
    # This is called "fast" because instead of computing the tensor product, it simply finds variables corresponding to the determinsic state-time and produces the tensor slice corresponding to the final state-time.
    # Assumes Deterministic T-Ops.  This state_vector update doesn't use tensordot, which is expensive rather it just indexes the
    # correct 'row' and searches for the 1 entry in the next-state indicies and returns that index.
    # Written for deterministic operators.  Can be modified for stochastic by representing t_d as probability vector.
    # The first vector in state_vecs must be the low-level state-vector in matrix form, [X,T]
    # g_to_secondary_map maps the goal variable to a task action on the secondary state-space

    internal_states = [np.where(isv.squeeze() == 1)[0][0] for isv in internal_state_vecs]
    # t_vec = xt_mat.sum(axis=0)
    T_f = len(t_vec)
    t_s = np.where(t_vec == 1)[0].item()
    # x_vec = xt_mat.sum(axis=1)
    x_s = np.where(x_vec == 1)[0].item()
    xt_mat_tf = eta[pi_ind, x_s, t_s, :, :]
    t_vec_tf = xt_mat_tf.sum(axis=0)
    x_vec_tf = xt_mat_tf.sum(axis=1)

    xt_inds = np.where(xt_mat_tf == 1)
    x_tf = xt_inds[0][0]
    t_f = xt_inds[1][0]

    mode_fail = False

    # xt_mat = xt_mat
    # x,t = np.where(xt_mat == 1)
    x_tf, t_f = np.where(eta[pi_ind, x_s, t_s, :, :])
    internal_update = []
    t_d = t_f[0]-t_s
    partial_modes = np.array([int(ss.states_to_mode_dict[internal_states[i]]) for i, ss in enumerate(internal_ss_list)], dtype=int)  # A partial mode is the mode e_y mapped by a single element y of a state-vector s=(w,y,...,z)

    # This if-else statement is the zeta function specific to SPA's ontology so that death states map r=(w,y,z,...) to the death mode. It will need to be generalized in the future for arbitrary mode-functions.
    if 0 in partial_modes:
        # Death mode (perhaps switch this to 0)
        mode = 0
        low_level_action = 0
    else:
        # Normal mode
        mode = 1
        low_level_action = pi[x_tf, t_f]
    mode_switch_times = [ss.time_to_mode_switch_dict[partial_modes[i]][internal_states[i]] for i, ss in enumerate(internal_ss_list)]

    # This is technically correct for SPA, but it should be generalized (above) so that the death mode is selected to update with eta_{death}.
    if not np.any(mode_switch_times < t_d) and mode != 0:
        # Policy call is successful and we perform a one-step Update
        x_tfp_vec = np.dot(x_vec_tf, p_axt_x[low_level_action, :, t_f, :])
        x_tfp = np.where(x_tfp_vec == 1)[1]  # One-step Update of Low-Level State
        t_fp = t_f + 1  # One-step Update of Time
        t_fp_vec = onehot(t_fp, T_f)

        # xt_mat_tfp = np.outer(x_tfp_vec, t_fp_vec)

        for i in range(len(internal_ss_list)):
            # internal_state_vec = np.where(state_vecs[i+1] == 1)[0]
            ss = internal_ss_list[i]
            alpha_g = np.where(f[:, low_level_action, x_tf, t_f])[0][0]
            action_ind = alpha_g
            if ss.action_type_ind == action_ind:
                alpha_g_val = ss.task_action
            else:
                alpha_g_val = ss.null_action

            # alpha_g = np.min([alpha_g[0], 1]) # This is a temporary hack which sets the action to 1 if greater than or equal to 1 because the operator P_a_s_s only has dimensions for null and "active" actions, not a specific index for one of many possible "active" actions.
            v = np.where(internal_state_vecs[i].squeeze() == 1)[0][0]
            v_tf_vec = secondary_omegas[i][t_d][v, :]
            # v_tf_onestep = np.dot(v_tf_vec, internal_ss_list[i].P_a_s_s[alpha_g_val,...])
            v_tf_onestep = sp.sparse.csr_matrix.dot(v_tf_vec, internal_ss_list[i].P_a_s_s[alpha_g_val, ...])
            # v_new =
            internal_update.append(v_tf_onestep)
    elif mode == 0:
        # Policy call fails and we return the input states and time.
        x_tfp_vec = np.dot(x_vec, p_axt_x[low_level_action, :, t_f, :])
        t_fp_vec = onehot(t_s + 1, T_f)
        # xt_mat_tfp = np.outer(x_tfp_vec, t_fp_vec)
        # internal_update = internal_state_vecs
        for i in range(len(internal_ss_list)):
            # internal_state_vec = np.where(state_vecs[i+1] == 1)[0]
            ss = internal_ss_list[i]
            alpha_g = np.where(f[:, low_level_action, x_tf, t_f])[0][0]
            action_ind = alpha_g - 1 # This is due to the fact that
            if ss.action_type_ind == action_ind:
                alpha_g_val = ss.task_action
            else:
                alpha_g_val = ss.null_action

            # alpha_g = np.min([alpha_g[0], 1]) # This is a temporary hack which sets the action to 1 if greater than or equal to 1 because the operator P_a_s_s only has dimensions for null and "active" actions, not a specific index for one of many possible "active" actions.
            v = np.where(internal_state_vecs[i].squeeze() == 1)[0][0]
            v_tf_vec = secondary_omegas[i][t_d][v, :].todense()
            v_tf_onestep = np.dot(v_tf_vec, internal_ss_list[i].P_a_s_s[alpha_g_val, ...])
            # v_new =
            internal_update.append(v_tf_onestep)
    else:
        x_tfp_vec = x_vec
        t_fp_vec = t_vec
        internal_update = internal_state_vecs
        mode_fail = True

    # updated_states = [xt_mat_updated]
    # updated_states.extend(internal_update)
    return x_tfp_vec, t_fp_vec, internal_update, mode_fail


def evolve_state_vecs_deterministic_stationary_fast(x_s, t_s, internal_states, internal_ss_list, eta_ens_dict, p_ax_x, secondary_omegas, pi_ind, pi, f, g_to_secondary_map = []):
    # This is called "fast" because instead of computing the tensor dot product, it simply finds variables corresponding to the determinsic state-time and produces the tensor slice corresponding to the final state-time.
    # Assumes Deterministic T-Ops.  This state_vector update doesn't use tensordot, which is expensive rather it just indexes the
    # correct 'row' and searches for the 1 entry in the next-state indicies and returns that index.
    # Written for deterministic operators.  Can be modified for stochastic by representing t_d as probability vector.
    # The first vector in state_vecs must be the low-level state-vector in matrix form, [X,T]
    # g_to_secondary_map maps the goal variable to a task action on the secondary state-space
    nx = p_ax_x.shape[1]
    na = p_ax_x.shape[0]//nx
    T_f = eta_ens_dict[0].shape[0]//nx
    # internal_states = [np.where(isv == 1)[0][0] for isv in internal_states]
    # t_s = np.where(t_vec == 1)[0].item()
    # x_vec = xt_mat.sum(axis=1)
    # x_s = np.where(x_vec == 1)[0].item()
    xt_s = nx * t_s + x_s
    xt_tf = eta_ens_dict[pi_ind][xt_s, :].nonzero()[1][0]
    x_tf = xt_tf % nx
    t_f = xt_tf // nx
    # xt_mat_tf = xt_mat_tf.reshape(nx, T_f)
    # t_vec_tf = xt_mat_tf.sum(axis=0)
    # x_vec_tf = xt_mat_tf.sum(axis=1)

    # xt_inds = np.where(xt_mat_tf == 1)
    # x_tf = xt_inds[0][0]
    # t_f = xt_inds[1][0]

    mode_fail = False

    # xt_mat = xt_mat
    # x,t = np.where(xt_mat == 1)
    # x_tf, t_f = eta_ens_dict[pi_ind][xt_s, :].nonzeros()
    internal_update = []
    t_d = t_f-t_s
    partial_modes = np.array([int(ss.states_to_mode_dict[internal_states[i]]) for i, ss in enumerate(internal_ss_list)], dtype=int)  # A partial mode is the mode e_y mapped by a single element y of a state-vector s=(w,y,...,z)
    time_block_1 = timeit.default_timer()
    # This if-else statement is the zeta function specific to SPA's ontology so that death states map r=(w,y,z,...) to the death mode. It will need to be generalized in the future for arbitrary mode-functions.
    if 0 in partial_modes:
        # Death mode (perhaps switch this to 0)
        mode = 0
        low_level_action = 0
    else:
        # Normal mode
        mode = 1
        low_level_action = pi[x_tf, t_f]
    mode_switch_times = [ss.time_to_mode_switch_dict[partial_modes[i]][internal_states[i]] for i, ss in enumerate(internal_ss_list)]
    #print("block 1: ", timeit.default_timer() - time_block_1)
    time_block_2 = timeit.default_timer()
    # This is technically correct for SPA, but it should be generalized (above) so that the death mode is selected to update with eta_{death}.
    if not np.any(mode_switch_times < t_d) and mode != 0:
        time_inner_evolve = timeit.default_timer()
        start_time = timeit.default_timer()
        # Policy call is successful and we perform a one-step update
        # x_tfp_vec = np.dot(x_vec_tf, p_ax_x[low_level_action, :, :])
        xa_ind = x_tf * na + low_level_action
        x_tfp = p_ax_x[xa_ind, :].nonzero()[1][0]
        #print("block 1.0: ", timeit.default_timer() - start_time)
        # x_tfp = np.where(x_tfp_vec == 1)[1]  # One-step Update of Low-Level State
        t_fp = t_f + 1  # One-step Update of Time
        for i in range(len(internal_ss_list)):
            start_time = timeit.default_timer()
            # internal_state_vec = np.where(state_vecs[i+1] == 1)[0]
            ss = internal_ss_list[i]
            alpha_g = np.where(f[:, low_level_action, x_tf])[0][0]
            # print("where_time: ", timeit.default_timer() - where_time)
            action_ind = alpha_g
            #print("block 1.1: ", timeit.default_timer() - start_time)
            start_time = timeit.default_timer()
            if ss.action_type_ind == action_ind:
                alpha_g_val = ss.task_action
            else:
                alpha_g_val = ss.null_action
            #print("block 1.2 ", timeit.default_timer() - start_time)
            start_time = timeit.default_timer()
            # alpha_g = np.min([alpha_g[0], 1]) # This is a temporary hack which sets the action to 1 if greater than or equal to 1 because the operator P_a_s_s only has dimensions for null and "active" actions, not a specific index for one of many possible "active" actions.
            indexing_time = timeit.default_timer()
            v = internal_states[i]
            v_tf = secondary_omegas[i][t_d][v, :].nonzero()[1][0]
            #print("block 1.3: ", timeit.default_timer() - start_time)
            start_time = timeit.default_timer()
            v_tf_onestep = internal_ss_list[i].P_s_s_given_g_dict[alpha_g_val][v_tf, :].nonzero()[1][0]
            #print("block 1.4: ", timeit.default_timer() - start_time)
            # print("indexing time: ", timeit.default_timer() - indexing_time)
            # v_new =
            internal_update.append(v_tf_onestep)
        # print("evolve inner 1: ", timeit.default_timer() - time_inner_evolve)
    elif mode == 0:
        # Policy call fails and we return the input states and time.
        xa_ind = x_tf * na + low_level_action
        x_tfp = p_ax_x[xa_ind, :].nonzero()[1][0]
        t_fp = t_f + 1
        # t_fp_vec = onehot(t_s + 1, T_f)
        # xt_mat_tfp = np.outer(x_tfp_vec, t_fp_vec)
        # internal_update = internal_state_vecs
        for i in range(len(internal_ss_list)):
            # internal_state_vec = np.where(state_vecs[i+1] == 1)[0]
            start_time = timeit.default_timer()
            ss = internal_ss_list[i]
            alpha_g = np.where(f[:, low_level_action, x_tf])[0][0]
            action_ind = alpha_g - 1 # This is due to the fact that
            if ss.action_type_ind == action_ind:
                alpha_g_val = ss.task_action
            else:
                alpha_g_val = ss.null_action
            #print("block 2.1: ", timeit.default_timer() - start_time)
            start_time = timeit.default_timer()
            v = internal_states[i]
            v_tf = secondary_omegas[i][t_d][v, :].nonzero()[1][0]
            start_time = timeit.default_timer()
            #print("block 2.2: ", timeit.default_timer() - start_time)
            start_time = timeit.default_timer()
            v_tf_onestep = internal_ss_list[i].P_s_s_given_g_dict[alpha_g_val][v_tf, :].nonzero()[1][0]
            #print("block 2.3: ", timeit.default_timer() - start_time)
            start_time = timeit.default_timer()
            # v_new =
            internal_update.append(v_tf_onestep)
            #print("block 2.4: ", timeit.default_timer() - start_time)
    else:
        x_tfp = x_s
        t_fp = t_s
        internal_update = internal_states
        mode_fail = True
    #print("block 2 total: ", timeit.default_timer() - time_block_2)
    # updated_states = [xt_mat_updated]
    # updated_states.extend(internal_update)
    return x_tfp, t_fp, internal_update, mode_fail


def evolve_state_vecs_deterministic_stationary_fast_dense_tensor(x_s, t_s, internal_states, internal_ss_list, eta_ens_dict, p_ax_x, secondary_omegas, pi_ind, pi, f, g_to_secondary_map = []):
    # This is called "fast" because instead of computing the tensor product, it simply finds variables corresponding to the determinsic state-time and produces the tensor slice corresponding to the final state-time.
    # Assumes Deterministic T-Ops.  This state_vector update doesn't use tensordot, which is expensive rather it just indexes the
    # correct 'row' and searches for the 1 entry in the next-state indicies and returns that index.
    # Written for deterministic operators.  Can be modified for stochastic by representing t_d as probability vector.
    # The first vector in state_vecs must be the low-level state-vector in matrix form, [X,T]
    # g_to_secondary_map maps the goal variable to a task action on the secondary state-space
    nx = p_ax_x.shape[1]
    na = p_ax_x.shape[0]//nx
    T_f = eta_ens_dict[0].shape[0]//nx
    # internal_states = [np.where(isv == 1)[0][0] for isv in internal_states]
    # t_s = np.where(t_vec == 1)[0].item()
    # x_vec = xt_mat.sum(axis=1)
    # x_s = np.where(x_vec == 1)[0].item()
    xt_s = nx * t_s + x_s
    xt_tf = eta_ens_dict[pi_ind][xt_s, :].nonzero()[1][0]
    x_tf = xt_tf % nx
    t_f = xt_tf // nx
    # xt_mat_tf = xt_mat_tf.reshape(nx, T_f)
    # t_vec_tf = xt_mat_tf.sum(axis=0)
    # x_vec_tf = xt_mat_tf.sum(axis=1)

    # xt_inds = np.where(xt_mat_tf == 1)
    # x_tf = xt_inds[0][0]
    # t_f = xt_inds[1][0]

    mode_fail = False

    # xt_mat = xt_mat
    # x,t = np.where(xt_mat == 1)
    # x_tf, t_f = eta_ens_dict[pi_ind][xt_s, :].nonzeros()
    internal_update = []
    t_d = t_f-t_s
    partial_modes = np.array([int(ss.states_to_mode_dict[internal_states[i]]) for i, ss in enumerate(internal_ss_list)], dtype=int)  # A partial mode is the mode e_y mapped by a single element y of a state-vector s=(w,y,...,z)
    time_block_1 = timeit.default_timer()
    # This if-else statement is the zeta function specific to SPA's ontology so that death states map r=(w,y,z,...) to the death mode. It will need to be generalized in the future for arbitrary mode-functions.
    if 0 in partial_modes:
        # Death mode (perhaps switch this to 0)
        mode = 0
        low_level_action = 0
    else:
        # Normal mode
        mode = 1
        low_level_action = pi[x_tf, t_f]
    mode_switch_times = [ss.time_to_mode_switch_dict[partial_modes[i]][internal_states[i]] for i, ss in enumerate(internal_ss_list)]
    #print("block 1: ", timeit.default_timer() - time_block_1)
    time_block_2 = timeit.default_timer()
    # This is technically correct for SPA, but it should be generalized (above) so that the death mode is selected to update with eta_{death}.
    if not np.any(mode_switch_times < t_d) and mode != 0:
        time_inner_evolve = timeit.default_timer()
        start_time = timeit.default_timer()
        # Policy call is successful and we perform a one-step update
        # x_tfp_vec = np.dot(x_vec_tf, p_ax_x[low_level_action, :, :])
        xa_ind = x_tf * na + low_level_action
        x_tfp = p_ax_x[xa_ind, :].nonzero()[1][0]
        #print("block 1.0: ", timeit.default_timer() - start_time)
        # x_tfp = np.where(x_tfp_vec == 1)[1]  # One-step Update of Low-Level State
        t_fp = t_f + 1  # One-step Update of Time
        for i in range(len(internal_ss_list)):
            start_time = timeit.default_timer()
            # internal_state_vec = np.where(state_vecs[i+1] == 1)[0]
            ss = internal_ss_list[i]
            alpha_g = np.where(f[:, low_level_action, x_tf])[0][0]
            # print("where_time: ", timeit.default_timer() - where_time)
            action_ind = alpha_g
            #print("block 1.1: ", timeit.default_timer() - start_time)
            start_time = timeit.default_timer()
            if ss.action_type_ind == action_ind:
                alpha_g_val = ss.task_action
            else:
                alpha_g_val = ss.null_action
            #print("block 1.2 ", timeit.default_timer() - start_time)
            start_time = timeit.default_timer()
            # alpha_g = np.min([alpha_g[0], 1]) # This is a temporary hack which sets the action to 1 if greater than or equal to 1 because the operator P_a_s_s only has dimensions for null and "active" actions, not a specific index for one of many possible "active" actions.
            indexing_time = timeit.default_timer()
            v = internal_states[i]
            v_tf = secondary_omegas[i][t_d][v, :].nonzero()[1][0]
            #print("block 1.3: ", timeit.default_timer() - start_time)
            start_time = timeit.default_timer()
            v_tf_onestep = internal_ss_list[i].P_s_s_given_g_dict[alpha_g_val][v_tf, :].nonzero()[1][0]
            #print("block 1.4: ", timeit.default_timer() - start_time)
            # print("indexing time: ", timeit.default_timer() - indexing_time)
            # v_new =
            internal_update.append(v_tf_onestep)
        # print("evolve inner 1: ", timeit.default_timer() - time_inner_evolve)
    elif mode == 0:
        # Policy call fails and we return the input states and time.
        xa_ind = x_tf * na + low_level_action
        x_tfp = p_ax_x[xa_ind, :].nonzero()[1][0]
        t_fp = t_f + 1
        # t_fp_vec = onehot(t_s + 1, T_f)
        # xt_mat_tfp = np.outer(x_tfp_vec, t_fp_vec)
        # internal_update = internal_state_vecs
        for i in range(len(internal_ss_list)):
            # internal_state_vec = np.where(state_vecs[i+1] == 1)[0]
            start_time = timeit.default_timer()
            ss = internal_ss_list[i]
            alpha_g = np.where(f[:, low_level_action, x_tf])[0][0]
            action_ind = alpha_g - 1 # This is due to the fact that
            if ss.action_type_ind == action_ind:
                alpha_g_val = ss.task_action
            else:
                alpha_g_val = ss.null_action
            #print("block 2.1: ", timeit.default_timer() - start_time)
            start_time = timeit.default_timer()
            v = internal_states[i]
            v_tf = secondary_omegas[i][t_d][v, :].nonzero()[1][0]
            start_time = timeit.default_timer()
            #print("block 2.2: ", timeit.default_timer() - start_time)
            start_time = timeit.default_timer()
            v_tf_onestep = internal_ss_list[i].P_s_s_given_g_dict[alpha_g_val][v_tf, :].nonzero()[1][0]
            #print("block 2.3: ", timeit.default_timer() - start_time)
            start_time = timeit.default_timer()
            # v_new =
            internal_update.append(v_tf_onestep)
            #print("block 2.4: ", timeit.default_timer() - start_time)
    else:
        x_tfp = x_s
        t_fp = t_s
        internal_update = internal_states
        mode_fail = True
    #print("block 2 total: ", timeit.default_timer() - time_block_2)
    # updated_states = [xt_mat_updated]
    # updated_states.extend(internal_update)
    return x_tfp, t_fp, internal_update, mode_fail


class Node:
    def __init__(self, x, t, ho_state_vecs, depth, x_vec=None, t_vec=None, avail_eval = None, trajectory_kappa = None, pi_seq = [], parent = None, is_leaf = False):
        self.x = x
        self.t = t
        self.x_vec = x_vec
        self.t_vec = t_vec
        self.ho_vecs = ho_state_vecs  # Higher-order state-vecs (states of hierarchical spaces)
        self.depth = depth
        self.parent = parent
        self.children = []
        self.pi_seq = pi_seq
        self.avail_eval = avail_eval
        self.trajectory_kappa = trajectory_kappa
        self.empowerment_n = np.nan
        self.empowerment_gain_n = np.nan

    def compute_empowerment_gain(self, initial_empowerment):
        # Assumes Determinism
        self.empowerment_gain_n = self.empowerment_n - initial_empowerment

class Operators:
    def __init__(self, p_xa_x, p_axt_t, eta_ens, internal_ops, p_v_list):
        self.p_xa_x = p_xa_x
        self.p_axt_t = p_axt_t
        self.eta_agg = eta_ens
        self.internal_ops = internal_ops
        self.p_v_list = p_v_list


class State_Vectors:
    def __init__(self, x_vec, t_vec, internal_vecs, e_vec, sigma_vec):
        self.x_vec = x_vec
        self.t_vec = t_vec
        self.internal_vecs = internal_vecs
        self.e_vec = e_vec
        self.sigma_vec = sigma_vec


def tgmdp_bredth_first_plan_search_nsnm_ud(x_vec_init, t_vec_init, ho_vecs_init, eta_ens, kappa_ens, p_ax_x, omegas, p_v_list, pi_ens, pi_ind_list, g_to_action_map, m_depth, sublimated_ho_kappas, f_bar):
    # This BFS search assumes that the dynamics model P_x and availability f used to create eta is deterministic.
    # To incorporate stochastic models, one must rewrite the part on state-evaluation with the availability
    # function because when we propagate state-vectors forward, we need to weight elements differently since f() and
    # 1-f() will be different evaluations, it's unclear how to do this with decomposed marginal state-vectors. We might
    # have to take the marginal vectors and compute the full joint distribution for evaluation, which is not ideal,
    # since we'd have to represent sparsly to avoid exponential state-vectors, and since there *might* be conditional
    # dependencies between state-space vectors. ud = unidirectional

    # Operators is a list [x_operator, internal_op_1,...,internal_op_n] where the first operator takes in X-actions
    # f_bar is a *sparse* hierarchical availability function of dimensions [nx,nt,nw,...,nz] where the trailing
    # dimensions are higher level spaces

    na, nx, _ = p_ax_x.shape
    T_f = len(t_vec_init)
    npi = kappa_ens.shape[0]
    full_state_coord = []
    for s in ([x_vec_init, t_vec_init] + ho_vecs_init):
        full_state_coord.append(np.where(s)[0][0])  # This assumes determinism!
    init_avail_eval = f_bar[tuple(full_state_coord)]
    root_node = Node(x_vec_init, t_vec_init, ho_vecs_init, 0, avail_eval=init_avail_eval, trajectory_kappa=init_avail_eval)
    bfs_queue = queue.Queue()
    bfs_queue.put(root_node)
    leaf_nodes = []
    kappa_ens_xt_flat = kappa_ens.reshape([npi,nx*T_f])  # Preflatten
    while not bfs_queue.empty():
        cur_node = bfs_queue.get()
        x_vec = cur_node.x_vec
        t_vec = cur_node.t_vec
        ho_vecs = cur_node.ho_vecs
        xt_vec = np.outer(x_vec, t_vec).flatten()
        kappa_feas_check = np.dot(kappa_ens_xt_flat, xt_vec) # Low-level Feasibility Check (Only use policies which can complete a goal with non-zero probability)
        valid_pi_inds = np.arange(len(pi_ind_list))
        # for pi_ind, _ in enumerate(kappa_feas_check[pi_ind] > 0):
        for pi_ind in valid_pi_inds[kappa_feas_check]: # Low-level Feasibility Check
            # if kappa_feas_check[pi_ind] > 0:
            new_states = evolve_state_vecs(x_vec, t_vec, cur_node.internal_vecs, eta_ens, p_ax_x, omegas, p_v_list,
                                           pi_ens[pi_ind,:,:], pi_ind, g_to_action_map)
            new_ho_vecs = new_states[2:]
            sublimation_check_list = []
            for i in range(len(new_ho_vecs)):
                new_zt_vec = np.outer(new_ho_vecs[i], new_states[1]).flatten()
                feasability_check = np.any(np.dot(sublimated_ho_kappas[i].flatten(), new_zt_vec)) #Boolean returning True if any state-times in the state-time distribution are feasibile (and thus, the expectation over current state-times will be feasible)
                sublimation_check_list.append(feasability_check)
            if np.all(sublimation_check_list):  # Sublimated Feasibility Check
                full_state_coord = []
                for s in new_states:
                    full_state_coord.append(np.where(s)[0][0])  # This assumes determinism!
                new_avail_eval = f_bar[tuple(full_state_coord)]
                kappa_update = (1 - cur_node.aval_eval) * new_avail_eval
                if (cur_node.depth + 1 == m_depth) or kappa_update == 1:
                    new_node = Node(new_states[0], new_states[1], new_states[2:], cur_node.depth + 1,
                                    avail_eval=new_avail_eval, trajectory_kappa=kappa_update,
                                    pi_seq=cur_node.pi_seq + [pi_ind], parent=cur_node, is_leaf=True)
                    leaf_nodes.append(new_node)
                else:
                    new_node = Node(new_states[0], new_states[1], new_states[2:], cur_node.depth + 1,
                                    avail_eval=new_avail_eval, trajectory_kappa=kappa_update,
                                    pi_seq=cur_node.pi_seq + [pi_ind], parent=cur_node, is_leaf=False)
                    bfs_queue.put(new_node)
    return leaf_nodes

def tgmdp_bredth_first_plan_eval(xt_mat, internal_vecs_init, internal_ss_list, eta_ens, p_axt_x, omegas, p_v_list, pi_ens, pi_ind_list, f, m_depth, f_bar):
    # Still needs testing!

    # Operators is a list [x_operator, internal_op_1,...,internal_op_n] where the first operator takes in X-actions
    # f_bar here is the function for evaluating whether a terminal state is acceptable.
    # Assumes Determinism
    t_vec_init = xt_mat.sum(axis=0)
    x_vec_init = xt_mat.sum(axis=1)
    root_node = Node(x_vec_init, t_vec_init, internal_vecs_init, 0)
    bfs_queue = queue.Queue()
    bfs_queue.put(root_node)
    leaf_nodes = []
    internal_vecs = internal_vecs_init
    while not bfs_queue.empty():
        cur_node = bfs_queue.get()
        x_vec = cur_node.x_vec
        x = np.where(x_vec==1)[0][0]
        t_vec = cur_node.t_vec
        t = x = np.where(t_vec==1)[0][0]
        xt_mat = np.outer(x_vec, t_vec)
        internal_vecs = cur_node.ho_vecs
        for pi_ind in pi_ind_list:
            xt_mat_tf, internal_update, mode_fail = evolve_state_vecs_deterministic(xt_mat, internal_vecs, internal_ss_list, eta_ens, p_axt_x, omegas, pi_ind, pi_ens[pi_ind],f, g_to_secondary_map = [])
            internal_states = [np.where(v == 1)[0][0] for v in internal_update]
            all_states = [x,t] + internal_states
            p_succ = f_bar[tuple(all_states)]
            if not mode_fail:  # Policy not invalidated by mode-switch
                t_vec_new = xt_mat_tf.sum(axis=0)
                x_vec_new = xt_mat_tf.sum(axis=1)
                if cur_node.depth+1 == m_depth:
                    new_node = Node(x_vec_new, t_vec_new, internal_update, cur_node.depth+1,
                                    pi_seq=cur_node.pi_seq + [pi_ind], parent=cur_node, is_leaf=True)
                    new_node.trajectory_kappa = new_node.trajectory_kappa + (1-new_node.trajectory_kappa) * p_succ
                    leaf_nodes.append(new_node)
                else:
                    new_node = Node(x_vec_new, t_vec_new, internal_update, cur_node.depth+1,
                                    pi_seq=cur_node.pi_seq + [pi_ind], parent=cur_node, is_leaf=False)
                    new_node.trajectory_kappa = new_node.trajectory_kappa + (1 - new_node.trajectory_kappa) * p_succ
                    bfs_queue.put(new_node)

    return leaf_nodes
def tgmdp_bredth_first_plan_search_spa(xt_mat, internal_vecs_init, internal_ss_list, eta_ens, p_axt_x, omegas, p_v_list, pi_ens, pi_ind_list, f, m_depth):
    # Operatos is a list [x_operator, internal_op_1,...,internal_op_n] where the first operator takes in X-actions
    t_vec_init = xt_mat.sum(axis=0)
    x_vec_init = xt_mat.sum(axis=1)
    root_node = Node(x_vec_init, t_vec_init, internal_vecs_init, 0)
    bfs_queue = queue.Queue()
    bfs_queue.put(root_node)
    leaf_nodes = []
    internal_vecs = internal_vecs_init
    while not bfs_queue.empty():
        cur_node = bfs_queue.get()
        x_vec = cur_node.x_vec
        t_vec = cur_node.t_vec
        xt_mat = np.outer(x_vec, t_vec)
        internal_vecs = cur_node.ho_vecs
        for pi_ind in pi_ind_list:
            xt_mat_tf, internal_update, mode_fail = evolve_state_vecs_deterministic_fast(xt_mat, internal_vecs, internal_ss_list, eta_ens, p_axt_x, omegas, pi_ind, pi_ens[pi_ind],f, g_to_secondary_map = [])
            if not mode_fail:  # Policy not invalidated by mode-switch
                t_vec_new = xt_mat_tf.sum(axis=0)
                x_vec_new = xt_mat_tf.sum(axis=1)
                if cur_node.depth+1 == m_depth:
                    new_node = Node(x_vec_new, t_vec_new, internal_update, cur_node.depth+1,
                                    pi_seq=cur_node.pi_seq + [pi_ind], parent=cur_node, is_leaf=True)
                    leaf_nodes.append(new_node)
                else:
                    new_node = Node(x_vec_new, t_vec_new, internal_update, cur_node.depth+1,
                                    pi_seq=cur_node.pi_seq + [pi_ind], parent=cur_node, is_leaf=False)
                    bfs_queue.put(new_node)

    return leaf_nodes


def tgmdp_bredth_first_plan_search_spa_deterministic_fast(x_vec_init, t_vec_init, internal_vecs_init, internal_ss_list, eta_ens, p_axt_x, omegas, p_v_list, pi_ens, pi_ind_list, f, m_depth):
    # This is called "fast" because instead of computing the tensor product, it simply finds variables corresponding to the determinsic state-time and produces the tensor slice corresponding to the final state-time.
    # Operatos is a list [x_operator, internal_op_1,...,internal_op_n] where the first operator takes in X-actions
    # t_vec_init = xt_mat.sum(axis=0)
    # x_vec_init = xt_mat.sum(axis=1)
    x = np.where(x_vec_init == 1)[0][0]
    t = np.where(t_vec_init == 1)[0][0]
    root_node = Node(x, t, internal_vecs_init, 0, x_vec=x_vec_init, t_vec=t_vec_init)
    bfs_queue = queue.Queue()
    bfs_queue.put(root_node)
    leaf_nodes = []
    internal_vecs = internal_vecs_init
    max_node_num = len(pi_ind_list)**m_depth
    node_num = 0
    num_added_to_q = 0
    num_not_added_to_q = 0
    num_added_to_leafs = 0
    pi_count = 0
    while not bfs_queue.empty():
        node_num += 1
        if node_num % 100 == 0:
            print("Compute ", m_depth, "-depth BFS: ", node_num, '/', max_node_num)
        cur_node = bfs_queue.get()
        x_vec = cur_node.x_vec
        t_vec = cur_node.t_vec
        # xt_mat = np.outer(x_vec, t_vec)
        internal_vecs = cur_node.ho_vecs
        for pi_ind in pi_ind_list:
            # pi_count += 1
            x_vec_new, t_vec_new, internal_update, mode_fail = evolve_state_vecs_deterministic_fast(x_vec, t_vec, internal_vecs, internal_ss_list, eta_ens, p_axt_x, omegas, pi_ind, pi_ens[pi_ind], f, g_to_secondary_map = [])
            x_vec_new = x_vec_new.squeeze()
            t_vec_new = t_vec_new.squeeze()
            if not mode_fail:  # Policy not invalidated by mode-switch
                # t_vec_new = xt_mat_tf.sum(axis=0)
                # x_vec_new = xt_mat_tf.sum(axis=1)
                new_x = np.where(x_vec_new == 1)[0][0]
                new_t = np.where(t_vec_new == 1)[0][0]
                if cur_node.depth+1 == m_depth:
                    new_node = Node(new_x, new_t, internal_update, cur_node.depth+1, x_vec=x_vec_new, t_vec=t_vec_new,
                                    pi_seq=cur_node.pi_seq + [pi_ind], parent=cur_node, is_leaf=True)
                    # num_added_to_leafs += 1
                    leaf_nodes.append(new_node)
                else:
                    new_node = Node(new_x, new_t, internal_update, cur_node.depth+1, x_vec=x_vec_new, t_vec=t_vec_new,
                                    pi_seq=cur_node.pi_seq + [pi_ind], parent=cur_node, is_leaf=False)
                    bfs_queue.put(new_node)
                    # num_added_to_q += 1

    # # if t % (T_f-2) == 0: print("num_added_to_leafs: ", num_added_to_leafs)
    # # if t % (T_f-2) == 0: print("num_added_to_q: ", num_added_to_q)
    # # if t % (T_f-2) == 0: print("pi_count: ", pi_count)
    return leaf_nodes


def tgmdp_bredth_first_plan_search_stationary_eta(x_vec_init, t_vec_init, internal_vecs_init, internal_ss_list, eta_pos_envs, eta_neg_envs, p_ax_x, omegas, p_v_list, pi_ens, pi_ind_list, big_F, m_depth):
    # This is called "fast" because instead of computing the tensor product, it simply finds variables corresponding to the determinsic state-time and produces the tensor slice corresponding to the final state-time.
    # Operatos is a list [x_operator, internal_op_1,...,internal_op_n] where the first operator takes in X-actions
    # t_vec_init = xt_mat.sum(axis=0)
    # x_vec_init = xt_mat.sum(axis=1)
    x = np.where(x_vec_init == 1)[0][0]
    t = np.where(t_vec_init == 1)[0][0]
    root_node = Node(x, t, internal_vecs_init, 0, x_vec=x_vec_init, t_vec=t_vec_init)
    bfs_queue = queue.Queue()
    bfs_queue.put(root_node)
    leaf_nodes = []
    internal_vecs = internal_vecs_init
    max_node_num = len(pi_ind_list)**m_depth
    node_num = 0
    num_added_to_q = 0
    num_not_added_to_q = 0
    num_added_to_leafs = 0
    pi_count = 0
    env_vec = utilities.onehot(0, len(eta_pos_envs))
    env_ind = 0
    while not bfs_queue.empty():
        node_num += 1
        if node_num % 100 == 0:
            print("Compute ", m_depth, "-depth BFS: ", node_num, '/', max_node_num)
        cur_node = bfs_queue.get()
        x_vec = cur_node.x_vec
        t_vec = cur_node.t_vec
        # xt_mat = np.outer(x_vec, t_vec)
        internal_vecs = cur_node.ho_vecs
        for pi_ind in pi_ind_list:
            # pi_count += 1
            pos_etas = eta_pos_envs[env_ind]
            eta_pos = pos_etas[pi_ind]
            neg_etas = eta_neg_envs[env_ind]
            eta_neg = neg_etas[pi_ind]
            st_pos = np.tensordot(x_vec, eta_pos, axes=[0, 0])
            st_neg = np.tensordot(x_vec, eta_neg, axes=[0, 0])
            x_vec = st_pos.sum(axis=0) + st_neg.sum(axis=0)
            internal_ss_list_new = []
            for y_i, int_ss in enumerate(internal_ss_list):
                y_vec = internal_vecs[y_i]
                ty = np.tensordot(y_vec, int_ss.null_omega_dense, axes=[0, 1])
                yp_new = np.dot(st_pos.sum(axis=0), ty)
                yn_new = np.dot(st_neg.sum(axis=0), ty)
                internal_ss_list_new.append(y_new)

            x_vec_new, t_vec_new, internal_update, mode_fail = evolve_state_vecs_deterministic_fast(x_vec, t_vec, internal_vecs, internal_ss_list, etas, p_ax_x, omegas, pi_ind, pi_ens[pi_ind], big_F, g_to_secondary_map = [])
            x_vec_new = x_vec_new.squeeze()
            t_vec_new = t_vec_new.squeeze()
            if not mode_fail:  # Policy not invalidated by mode-switch
                # t_vec_new = xt_mat_tf.sum(axis=0)
                # x_vec_new = xt_mat_tf.sum(axis=1)
                new_x = np.where(x_vec_new == 1)[0][0]
                new_t = np.where(t_vec_new == 1)[0][0]
                if cur_node.depth+1 == m_depth:
                    new_node = Node(new_x, new_t, internal_update, cur_node.depth+1, x_vec=x_vec_new, t_vec=t_vec_new,
                                    pi_seq=cur_node.pi_seq + [pi_ind], parent=cur_node, is_leaf=True)
                    # num_added_to_leafs += 1
                    leaf_nodes.append(new_node)
                else:
                    new_node = Node(new_x, new_t, internal_update, cur_node.depth+1, x_vec=x_vec_new, t_vec=t_vec_new,
                                    pi_seq=cur_node.pi_seq + [pi_ind], parent=cur_node, is_leaf=False)
                    bfs_queue.put(new_node)
                    # num_added_to_q += 1

    # # if t % (T_f-2) == 0: print("num_added_to_leafs: ", num_added_to_leafs)
    # # if t % (T_f-2) == 0: print("num_added_to_q: ", num_added_to_q)
    # # if t % (T_f-2) == 0: print("pi_count: ", pi_count)
    return leaf_nodes


def tgmdp_BFS_stationary_eta_sp_tensor(x_vec_init, t_vec_init, internal_vecs_init, internal_ss_list, eta_pos_envs, eta_neg_envs, p_ax_x, omegas, p_v_list, pi_ens, pi_ind_list, big_F, m_depth):
    # This is needs to be implemented using the sparse package.
    # t_vec_init = xt_mat.sum(axis=0)
    # x_vec_init = xt_mat.sum(axis=1)
    x = np.where(x_vec_init == 1)[0][0]
    t = np.where(t_vec_init == 1)[0][0]
    root_node = Node(x, t, internal_vecs_init, 0, x_vec=x_vec_init, t_vec=t_vec_init)
    bfs_queue = queue.Queue()
    bfs_queue.put(root_node)
    leaf_nodes = []
    internal_vecs = internal_vecs_init
    max_node_num = len(pi_ind_list)**m_depth
    node_num = 0
    num_added_to_q = 0
    num_not_added_to_q = 0
    num_added_to_leafs = 0
    pi_count = 0
    env_vec = utilities.onehot(0, len(eta_pos_envs))
    env_ind = 0
    while not bfs_queue.empty():
        node_num += 1
        if node_num % 100 == 0:
            print("Compute ", m_depth, "-depth BFS: ", node_num, '/', max_node_num)
        cur_node = bfs_queue.get()
        x_vec = cur_node.x_vec
        t_vec = cur_node.t_vec
        # xt_mat = np.outer(x_vec, t_vec)
        internal_vecs = cur_node.ho_vecs
        for pi_ind in pi_ind_list:
            # pi_count += 1
            pos_etas = eta_pos_envs[env_ind]
            eta_pos = pos_etas[pi_ind]
            neg_etas = eta_neg_envs[env_ind]
            eta_neg = neg_etas[pi_ind]
            st_pos = np.tensordot(x_vec, eta_pos, axes=[0, 0])
            st_neg = np.tensordot(x_vec, eta_neg, axes=[0, 0])
            x_vec = st_pos.sum(axis=0) + st_neg.sum(axis=0)
            internal_ss_list_new = []
            for y_i, int_ss in enumerate(internal_ss_list):
                y_vec = internal_vecs[y_i]
                ty = np.tensordot(y_vec, int_ss.null_omega_dense, axes=[0, 1])
                yp_new = np.dot(st_pos.sum(axis=0), ty)
                yn_new = np.dot(st_neg.sum(axis=0), ty)
                internal_ss_list_new.append(y_new)

            x_vec_new, t_vec_new, internal_update, mode_fail = evolve_state_vecs_deterministic_fast(x_vec, t_vec, internal_vecs, internal_ss_list, etas, p_ax_x, omegas, pi_ind, pi_ens[pi_ind], big_F, g_to_secondary_map = [])
            x_vec_new = x_vec_new.squeeze()
            t_vec_new = t_vec_new.squeeze()
            if not mode_fail:  # Policy not invalidated by mode-switch
                # t_vec_new = xt_mat_tf.sum(axis=0)
                # x_vec_new = xt_mat_tf.sum(axis=1)
                new_x = np.where(x_vec_new == 1)[0][0]
                new_t = np.where(t_vec_new == 1)[0][0]
                if cur_node.depth+1 == m_depth:
                    new_node = Node(new_x, new_t, internal_update, cur_node.depth+1, x_vec=x_vec_new, t_vec=t_vec_new,
                                    pi_seq=cur_node.pi_seq + [pi_ind], parent=cur_node, is_leaf=True)
                    # num_added_to_leafs += 1
                    leaf_nodes.append(new_node)
                else:
                    new_node = Node(new_x, new_t, internal_update, cur_node.depth+1, x_vec=x_vec_new, t_vec=t_vec_new,
                                    pi_seq=cur_node.pi_seq + [pi_ind], parent=cur_node, is_leaf=False)
                    bfs_queue.put(new_node)
                    # num_added_to_q += 1

    # # if t % (T_f-2) == 0: print("num_added_to_leafs: ", num_added_to_leafs)
    # # if t % (T_f-2) == 0: print("num_added_to_q: ", num_added_to_q)
    # # if t % (T_f-2) == 0: print("pi_count: ", pi_count)
    return leaf_nodes


def dfs(current_state, action_sequence, depth, max_depth, p_ax_x, score_fn=None):
    """
    Recursively explores the action tree up to a fixed depth and returns a list of results.

    Each result is a tuple (action_sequence, final_state, score). Debug print statements
    trace execution without printing the complete state vectors.

    Parameters:
      current_state : np.ndarray
          The current state vector.
      action_sequence : list
          List of action indices taken so far.
      depth : int
          Current depth in the DFS tree.
      max_depth : int
          The fixed maximum depth for the DFS.
      p_ax_x : np.ndarray
          The transition kernel, a 3D numpy array of shape (na, nx, nx) where
          na is the number of actions and nx is the state vector dimension.
      score_fn : function or None, optional
          A function that takes a state vector and returns a scalar score.
          If not callable or None, the score is set to None.

    Returns:
      results : list of tuples
          Each tuple is (action_sequence, final_state, score) for a leaf node.
    """
    # Print debug message without printing the state vector.
    # test = dfs(onehot(0, p_ax_x.shape[0]), [], 0, 3, p_ax_x, [])

    print(f"Entering node: depth={depth}, action_sequence={action_sequence}")

    # Base case: maximum depth reached.
    if depth == max_depth:
        score = score_fn(current_state) if (score_fn is not None and callable(score_fn)) else None
        print(f"Leaf node reached: action_sequence={action_sequence}, score={score}")
        return [(action_sequence, current_state, score)]

    results = []
    for a in range(p_ax_x.shape[0]):
        print(f"At depth {depth}: processing action {a}")

        # Propagate the state using the transition kernel.
        new_state = np.dot(current_state, p_ax_x[a, :, :])
        new_action_sequence = action_sequence + [a]
        child_results = dfs(new_state, new_action_sequence, depth + 1, max_depth, p_ax_x, score_fn)
        results.extend(child_results)

    print(f"Exiting node: depth={depth}, action_sequence={action_sequence}")
    return results


def get_sparse_entries(arr):
    """
    Given an array (dense or a sparse.COO), return:
      - indices: a list of tuples representing the multi-index of each nonzero element.
      - values: a numpy array of corresponding nonzero values.
      - shape: the shape of the array.
    """
    if isinstance(arr, sparse.COO):
        # For a sparse COO tensor, assume arr.coords is of shape (ndim, nnz)
        coords = arr.coords.T  # now shape (nnz, ndim)
        indices = [tuple(coord) for coord in coords]
        values = arr.data
        shape = arr.shape
    else:
        # Assume a dense numpy array.
        nz = np.nonzero(arr)  # returns a tuple of arrays, one per dimension.
        indices = list(zip(*nz))
        values = arr[arr != 0]
        shape = arr.shape
    return indices, values, shape


def generalized_sparse_outer_product(*arrays):
    """
    Compute the sparse outer product of arbitrary arrays (vectors or higher-order tensors).
    The resulting tensor has shape equal to the concatenation of the shapes of the inputs.

    For each input, only the nonzero entries are used, and the resulting nonzero entries
    are the products of the corresponding nonzero values from each input.
    """
    # For each input array, get its nonzero entries (indices, values, shape)
    entries = [get_sparse_entries(arr) for arr in arrays]
    # Each entry is a tuple: (list of index tuples, values, shape)
    indices_lists = [entry[0] for entry in entries]  # list of lists of index tuples
    values_lists = [entry[1] for entry in entries]
    shapes = [entry[2] for entry in entries]

    # Compute the Cartesian product over the nonzero indices from each input.
    # Each element in cartesian_indices is a tuple of index tuples, one per input.
    cartesian_indices = list(product(*indices_lists))

    out_indices = []
    out_values = []
    for combo in cartesian_indices:
        # combo is something like ( (i0, i1, ...), (j0, j1, ...), ... )
        # The final index is the concatenation of these tuples.
        combined_index = tuple(idx for sub_idx in combo for idx in sub_idx)

        # Multiply the corresponding nonzero values.
        prod_val = 1
        for dim, idx in enumerate(combo):
            # Find the position of idx in indices_lists[dim]
            pos = indices_lists[dim].index(idx)
            prod_val *= values_lists[dim][pos]
        out_indices.append(combined_index)
        out_values.append(prod_val)

    # Convert the list of output indices to a NumPy array of shape (total_ndim, nnz)
    if out_indices:
        coords = np.array(out_indices).T  # each column is one coordinate
    else:
        # No nonzero entries, so create an empty coordinate array.
        total_ndim = sum(len(shape) for shape in shapes)
        coords = np.empty((total_ndim, 0), dtype=int)

    # The final tensor's shape is the concatenation of the input shapes.
    final_shape = tuple(dim for shape in shapes for dim in shape)

    # Build and return the sparse tensor (COO format)
    return sparse.COO(coords, np.array(out_values), shape=final_shape)


def sparse_outer_product(*vectors):
    """
    Compute the outer product of multiple sparse vectors and return a sparse COO tensor.

    Parameters:
        *vectors: np.array
            A variable number of 1D NumPy arrays representing the sparse vectors.
            The vectors can be dense (with zeros) but only their nonzero elements are used.

    Returns:
        sparse.COO: The sparse outer product tensor.
    """
    # Extract nonzero indices and corresponding nonzero values for each vector.
    nonzero_indices = [np.nonzero(vec)[0] for vec in vectors]
    # Convert each numpy array of indices to a list so we can use .index() later.
    nonzero_indices_lists = [list(idxs) for idxs in nonzero_indices]
    nonzero_values = [vec[idxs] for vec, idxs in zip(vectors, nonzero_indices)]

    # Generate the Cartesian product of nonzero indices.
    # Each element of `indices` is a tuple like (i, j, k, ...) representing one coordinate.
    indices = list(product(*nonzero_indices))

    # For each tuple of indices, determine the corresponding values by finding the position
    # in the nonzero-values array. For example, if vec1 has nonzero indices [0, 2] and we get 2,
    # we need to use position 1 in the corresponding nonzero_values.
    values = np.array([
        np.prod([
            nonzero_values[i][nonzero_indices_lists[i].index(idx)]
            for i, idx in enumerate(idx_tuple)
        ])
        for idx_tuple in indices
    ])

    # Convert the list of index tuples into an array with shape (ndim, nnz).
    indices = np.array(indices).T

    # Determine the shape of the output tensor from the original vectors.
    shape = tuple(vec.shape[0] for vec in vectors)

    # Create and return the sparse COO tensor.
    return sparse.COO(indices, values, shape=shape)


import sparse
import string


def contract_over_time(xi_joint, SPK_lst):
    """
    xi_joint: sparse.COO of shape (n1, n2, ..., nk, nt)
    SPK_lst: list of sparse.COO of shape (ni, ni', nt)

    Returns:
        sparse.COO of shape (n1, ..., nk, n1', ..., nk')
    """
    ndim = len(SPK_lst)  # number of state variables
    assert xi_joint.ndim == ndim + 1  # last dim of xi_joint is time

    # Prepare einsum labels
    base_letters = list(string.ascii_lowercase)
    lhs_inds = base_letters[:ndim]  # e.g. ['a', 'b', 'c', ...]
    time_ind = base_letters[ndim]  # e.g. 'e'
    primed_inds = base_letters[ndim + 1: 2 * ndim + 1]  # e.g. ['p', 'q', 'r', ...]

    # Build einsum string
    einsum_terms = []

    # Term for xi_joint
    xi_term = ''.join(lhs_inds) + time_ind
    einsum_terms.append(xi_term)

    # Terms for each SPK
    for i in range(ndim):
        einsum_terms.append(lhs_inds[i] + primed_inds[i] + time_ind)

    # Output term
    output_term = ''.join(lhs_inds + primed_inds)

    # Final einsum string
    einsum_str = ','.join(einsum_terms) + '->' + output_term

    # Call sparse.einsum
    return sparse.einsum(einsum_str, xi_joint, *SPK_lst)


def dfs_sparse_tensor(current_tensor, action_sequence, depth, max_depth, p_ax_x, eta_x_dict, hl_kernel_lst, hl_SPK_lst, BL_SPK_env_dict, F_lst, pi_ind_list, pi_lst, HL_state_spaces, state_time_kappa_comp_list, T_f, score_fn=None):
    """
    Recursively explores the action tree up to a fixed depth and returns a list of results.

    Each result is a tuple (action_sequence, final_state, score). Debug print statements
    trace execution without printing the complete state vectors.

    Parameters:
      current_tensor : np.ndarray
          The current state vector.
      action_sequence : list
          List of action indices taken so far.
      depth : int
          Current depth in the DFS tree.
      max_depth : int
          The fixed maximum depth for the DFS.
      p_ax_x : np.ndarray
          The transition kernel, a 3D numpy array of shape (na, nx, nx) where
          na is the number of actions and nx is the state vector dimension.
      score_fn : function or None, optional
          A function that takes a state vector and returns a scalar score.
          If not callable or None, the score is set to None.

    Returns:
      results : list of tuples
          Each tuple is (action_sequence, final_state, score) for a leaf node.
    """
    # Print debug message without printing the state vector.
    # test = dfs(onehot(0, p_ax_x.shape[0]), [], 0, 3, p_ax_x, [])
    print(f"Entering node: depth={depth}, action_sequence={action_sequence}")
    full_kernel_list = [p_ax_x] + hl_kernel_lst
    env = 0
    HL_xi_list = [HL_state_spaces[i].eta_event_0.sum(axis=1) for i in range(len(HL_state_spaces))]
    HL_spk_event = [np.divide(HL_state_spaces[i].eta_event_0, HL_xi_list[i]) for i in range(len(HL_state_spaces))]
    # Base case: maximum depth reached.
    if depth == max_depth:
        score = score_fn(current_tensor) if (score_fn is not None and callable(score_fn)) else None
        print(f"Leaf node reached: action_sequence={action_sequence}, score={score}")
        return [(action_sequence, current_tensor, score)]

    results = []
    for pi_ind in pi_ind_list:
        print(f"At depth {depth}: processing action {pi_ind}")
        # Check that the current state's dimension matches the transition matrix dimension.
        # if current_tensor.shape[0] != p_ax_x[pi_ind, :, :].shape[0]:
        #     print(
        #         f"Dimension mismatch at action {pi_ind}: current_state dimension {current_tensor.shape[0]} != transition matrix dimension {p_ax_x[pi_ind, :, :].shape[0]}")
        #     continue
        # Propagate the state using the transition kernel.
        # marginal_list = [current_tensor.sum(axis=tuple(i for i in range(current_tensor.ndim) if i != k)) for k in range(len(current_tensor.shape))]
        xi_joint = TGMDP_methods.compute_TEF_given_joint_dist(state_time_kappa_comp_list, current_tensor, env, pi_ind, T_f)
        x_vec = current_tensor.sum(axis=tuple(range(1, current_tensor.ndim))).todense()
        h_vecs = []
        bl_spk = BL_SPK_env_dict[env][pi_ind]
        spk_list = [bl_spk] + hl_SPK_lst

        # new_joint_inter = sparse.tensordot(current_tensor, eta_x_dict[env][pi_ind], axes=((0,), (0,)))
        # new_joint_inter_new = new_joint_inter
        pred_funs_marged_lst = []
        for j, pred_fun in enumerate(hl_SPK_lst):
            # F = sparse.COO(F_lst[j])
            F = F_lst[j]
            # hl_action_dist = sparse.tensordot(x_vec, F[:, 0, :], axes=(0, 1))  # Get distribution over high-level actions in order to evolve hl space (for each space)
            hl_action_dist = np.dot(F[:, 0, :], x_vec)  # Get distribution over high-level actions in order to evolve hl space (for each space)
            hl_predict = sparse.tensordot(sparse.COO(hl_action_dist), pred_fun, axes=(0, 0))  # Marginalize over high-level actions to get the prediction function \rho
            pred_funs_marged_lst.append(hl_predict)
            # new_joint_inter_new = sparse.tensordot(new_joint_inter_new, hl_predict, axes=((0,), (0,)))

        pred_fun_marged_full = [bl_spk] + pred_funs_marged_lst
        contracted = contract_over_time(xi_joint, pred_fun_marged_full)
        next_state_tensor = contracted.sum(axis=(0,1,2,3))
        # composite_spk = build_composite_operator_sparse(pred_funs_marged_lst)

        new_state = np.dot(current_tensor, p_ax_x[pi_ind, :, :])
        new_action_sequence = action_sequence + [pi_ind]
        child_results = dfs(new_state, new_action_sequence, depth + 1, max_depth, p_ax_x, score_fn)
        results.extend(child_results)

    print(f"Exiting node: depth={depth}, action_sequence={action_sequence}")
    return results


def dfs_sparse_tensor_deterministic(current_tensor, action_sequence, depth, max_depth, p_ax_x, eta_x_dict, hl_kernel_lst, hl_SPK_lst, BL_SPK_env_dict, F_lst, pi_ind_list, pi_lst, HL_state_spaces, state_time_kappa_comp_list, T_f, score_fn=None):
    """
    Recursively explores the action tree up to a fixed depth and returns a list of results.

    Each result is a tuple (action_sequence, final_state, score). Debug print statements
    trace execution without printing the complete state vectors.

    Parameters:
      current_tensor : np.ndarray
          The current state vector.
      action_sequence : list
          List of action indices taken so far.
      depth : int
          Current depth in the DFS tree.
      max_depth : int
          The fixed maximum depth for the DFS.
      p_ax_x : np.ndarray
          The transition kernel, a 3D numpy array of shape (na, nx, nx) where
          na is the number of actions and nx is the state vector dimension.
      score_fn : function or None, optional
          A function that takes a state vector and returns a scalar score.
          If not callable or None, the score is set to None.

    Returns:
      results : list of tuples
          Each tuple is (action_sequence, final_state, score) for a leaf node.
    """
    # Print debug message without printing the state vector.
    # test = dfs(onehot(0, p_ax_x.shape[0]), [], 0, 3, p_ax_x, [])
    print(f"Entering node: depth={depth}, action_sequence={action_sequence}")
    full_kernel_list = [p_ax_x] + hl_kernel_lst
    env = 0
    HL_xi_list = [HL_state_spaces[i].eta_event_0.sum(axis=1) for i in range(len(HL_state_spaces))]
    HL_spk_event = [np.divide(HL_state_spaces[i].eta_event_0, HL_xi_list[i]) for i in range(len(HL_state_spaces))]
    # Base case: maximum depth reached.
    if depth == max_depth:
        score = score_fn(current_tensor) if (score_fn is not None and callable(score_fn)) else None
        print(f"Leaf node reached: action_sequence={action_sequence}, score={score}")
        return [(action_sequence, current_tensor, score)]

    results = []
    for pi_ind in pi_ind_list:
        print(f"At depth {depth}: processing action {pi_ind}")
        # Check that the current state's dimension matches the transition matrix dimension.
        # if current_tensor.shape[0] != p_ax_x[pi_ind, :, :].shape[0]:
        #     print(
        #         f"Dimension mismatch at action {pi_ind}: current_state dimension {current_tensor.shape[0]} != transition matrix dimension {p_ax_x[pi_ind, :, :].shape[0]}")
        #     continue
        # Propagate the state using the transition kernel.
        # marginal_list = [current_tensor.sum(axis=tuple(i for i in range(current_tensor.ndim) if i != k)) for k in range(len(current_tensor.shape))]
        xi_joint = TGMDP_methods.compute_TEF_given_joint_dist(state_time_kappa_comp_list, current_tensor, env, pi_ind, T_f)
        x_vec = current_tensor.sum(axis=tuple(range(1, current_tensor.ndim))).todense()
        h_vecs = []
        bl_spk = BL_SPK_env_dict[env][pi_ind]
        spk_list = [bl_spk] + hl_SPK_lst

        # new_joint_inter = sparse.tensordot(current_tensor, eta_x_dict[env][pi_ind], axes=((0,), (0,)))
        # new_joint_inter_new = new_joint_inter
        pred_funs_marged_lst = []
        for j, pred_fun in enumerate(hl_SPK_lst):
            # F = sparse.COO(F_lst[j])
            F = F_lst[j]
            # hl_action_dist = sparse.tensordot(x_vec, F[:, 0, :], axes=(0, 1))  # Get distribution over high-level actions in order to evolve hl space (for each space)
            hl_action_dist = np.dot(F[:, 0, :], x_vec)  # Get distribution over high-level actions in order to evolve hl space (for each space)
            hl_predict = sparse.tensordot(sparse.COO(hl_action_dist), pred_fun, axes=(0, 0))  # Marginalize over high-level actions to get the prediction function \rho
            pred_funs_marged_lst.append(hl_predict)
            # new_joint_inter_new = sparse.tensordot(new_joint_inter_new, hl_predict, axes=((0,), (0,)))

        pred_fun_marged_full = [bl_spk] + pred_funs_marged_lst
        contracted = contract_over_time(xi_joint, pred_fun_marged_full)
        next_state_tensor = contracted.sum(axis=(0,1,2,3))
        # composite_spk = build_composite_operator_sparse(pred_funs_marged_lst)

        new_state = np.dot(current_tensor, p_ax_x[pi_ind, :, :])
        new_action_sequence = action_sequence + [pi_ind]
        child_results = dfs(new_state, new_action_sequence, depth + 1, max_depth, p_ax_x, score_fn)
        results.extend(child_results)

    print(f"Exiting node: depth={depth}, action_sequence={action_sequence}")
    return results


def dfs_monte_carlo(current_tensor, action_sequence, depth, max_depth, p_ax_x, eta_x_dict, hl_kernel_lst, hl_SPK_lst, BL_SPK_env_dict, F_lst, pi_ind_list, pi_lst, HL_state_spaces, state_time_kappa_comp_list, T_f, score_fn=None):
    """
    This DFS tree samples from the joint product space factorization at every step.

    Parameters:
      current_tensor : np.ndarray
          The current state vector.
      action_sequence : list
          List of action indices taken so far.
      depth : int
          Current depth in the DFS tree.
      max_depth : int
          The fixed maximum depth for the DFS.
      p_ax_x : np.ndarray
          The transition kernel, a 3D numpy array of shape (na, nx, nx) where
          na is the number of actions and nx is the state vector dimension.
      score_fn : function or None, optional
          A function that takes a state vector and returns a scalar score.
          If not callable or None, the score is set to None.

    Returns:
      results : list of tuples
          Each tuple is (action_sequence, final_state, score) for a leaf node.
    """
    # Print debug message without printing the state vector.
    # test = dfs(onehot(0, p_ax_x.shape[0]), [], 0, 3, p_ax_x, [])
    print(f"Entering node: depth={depth}, action_sequence={action_sequence}")
    full_kernel_list = [p_ax_x] + hl_kernel_lst
    env = 0
    # Base case: maximum depth reached.
    if depth == max_depth:
        score = score_fn(current_tensor) if (score_fn is not None and callable(score_fn)) else None
        print(f"Leaf node reached: action_sequence={action_sequence}, score={score}")
        return [(action_sequence, current_tensor, score)]

    results = []
    for pi_ind in pi_ind_list:
        print(f"At depth {depth}: processing action {pi_ind}")

        new_state = np.dot(current_tensor, p_ax_x[pi_ind, :, :])
        new_action_sequence = action_sequence + [pi_ind]
        child_results = dfs(new_state, new_action_sequence, depth + 1, max_depth, p_ax_x, score_fn)
        results.extend(child_results)

    print(f"Exiting node: depth={depth}, action_sequence={action_sequence}")
    return results


def build_composite_operator_sparse(pred_funs_lst):
    """
    pred_funs_lst: list of sparse.COO tensors, each of shape (n_in_i, n_out_i, nt)

    Returns:
        composite_op: sparse.COO tensor with shape:
            (n_in_1, ..., n_in_k, n_out_1, ..., n_out_k, nt)
    """
    num_vars = len(pred_funs_lst)
    time_dim = pred_funs_lst[0].shape[-1]

    composite_op = None

    for i, P in enumerate(pred_funs_lst):
        n_in, n_out, nt = P.shape
        assert nt == time_dim, f"Time mismatch on tensor {i}"

        full_shape = [1] * (2 * num_vars + 1)
        full_shape[i] = n_in
        full_shape[num_vars + i] = n_out
        full_shape[-1] = nt

        P_reshaped = P.reshape(full_shape)

        if composite_op is None:
            composite_op = P_reshaped
        else:
            composite_op = composite_op * P_reshaped  # sparse-safe broadcasting

    return composite_op


def tgmdp_bredth_first_plan_search_spa_deterministic_stationary_fast(x_s, t_s, internal_states_init, internal_ss_list, eta_ens_dict, p_ax_x, omegas, p_v_list, pi_ens, pi_ind_list, f, m_depth):
    # This is called "fast" because instead of computing the tensor product, it simply finds variables corresponding to the determinsic state-time and produces the tensor slice corresponding to the final state-time.
    # Operatos is a list [x_operator, internal_op_1,...,internal_op_n] where the first operator takes in X-actions
    # t_vec_init = xt_mat.sum(axis=0)
    # x_vec_init = xt_mat.sum(axis=1)
    root_node = Node(x_s, t_s, internal_states_init, 0)
    bfs_queue = queue.Queue()
    bfs_queue.put(root_node)
    leaf_nodes = []
    internal_states = internal_states_init
    max_node_num = len(pi_ind_list)**m_depth
    node_num = 0
    num_added_to_q = 0
    num_not_added_to_q = 0
    num_added_to_leafs = 0
    pi_count = 0
    # print("Computing BFS for ", max_node_num, "Nodes")
    while not bfs_queue.empty():
        # if node_num%(max_node_num-2): print("Node: ", node_num, "/", max_node_num)
        cur_node = bfs_queue.get()

        x_s = cur_node.x
        t_s = cur_node.t
        # xt_mat = np.outer(x_vec, t_vec)
        internal_vecs = cur_node.ho_vecs
        for pi_ind in pi_ind_list:
            # pi_count += 1
            state_evolve_time = timeit.default_timer()
            x_fp, t_fp, internal_update, mode_fail = evolve_state_vecs_deterministic_stationary_fast(x_s, t_s,
                                                                                                     internal_states,
                                                                                                     internal_ss_list,
                                                                                                     eta_ens_dict,
                                                                                                     p_ax_x, omegas,
                                                                                                     pi_ind, pi_ens[
                                                                                                         pi_ind], f,
                                                                                                     g_to_secondary_map=[])
            # print("evolve time: ", timeit.default_timer() - state_evolve_time)
            if not mode_fail:  # Policy not invalidated by mode-switch
                # t_new = xt_mat_tf.sum(axis=0)
                # x_fp = xt_mat_tf.sum(axis=1)
                if cur_node.depth+1 == m_depth:
                    new_node = Node(x_fp, t_fp, internal_update, cur_node.depth + 1,
                                    pi_seq=cur_node.pi_seq + [pi_ind], parent=cur_node, is_leaf=True)
                    # num_added_to_leafs += 1
                    leaf_nodes.append(new_node)
                    node_num += 1
                    # if len(leaf_nodes) % 100 == 0:
                    #     print("Compute ", m_depth, "-depth BFS: ", node_num, '/', max_node_num)
                else:
                    new_node = Node(x_fp, t_fp, internal_update, cur_node.depth + 1,
                                    pi_seq=cur_node.pi_seq + [pi_ind], parent=cur_node, is_leaf=False)
                    bfs_queue.put(new_node)
                    node_num += 1
                    # if len(leaf_nodes) % 100 == 0:
                    #     print("Compute ", m_depth, "-depth BFS: ", node_num, '/', max_node_num)
                    # num_added_to_q += 1

    # # if t % (T_f-2) == 0: print("num_added_to_leafs: ", num_added_to_leafs)
    # # if t % (T_f-2) == 0: print("num_added_to_q: ", num_added_to_q)
    # # if t % (T_f-2) == 0: print("pi_count: ", pi_count)
    return leaf_nodes


def deterministic_empowerment_bfs(x_vec, t_vec, internal_vecs_init, internal_op_list, eta_ens, p_axt_x, omega_list, p_v_list, pi_ens, pi_ind_list,
                                x_g_to_action_map, f, n_depth):
    # This empowerment solver assumes determinism in all representations
    xt_mat = np.outer(x_vec, t_vec)
    leaves = tgmdp_bredth_first_plan_search_spa(xt_mat, internal_vecs_init, internal_op_list, eta_ens, p_axt_x, omega_list, p_v_list, pi_ens, pi_ind_list, f, n_depth)
    state_vector_list = []
    for leaf in leaves:
        x = np.where(leaf.x_vec)[0].item()
        t = np.where(leaf.t_vec)[0].item()
        int_state_index_vec = [x, t]
        for i in range(len(leaf.ho_vecs)):
            int_state_index_vec.append(np.where(leaf.ho_vecs[i])[0][0])

        state_vector_list.append(int_state_index_vec)
    state_vector_array = np.array(state_vector_list)
    unique_rows = np.unique(state_vector_array, axis=0)
    empowerment = np.log2(len(unique_rows))

    return empowerment


def deterministic_empowerment_bfs_fast(x_vec, t_vec, internal_vecs_init, internal_op_list, eta_ens, p_axt_x, omega_list, p_v_list, pi_ens, pi_ind_list,
                                x_g_to_action_map, f, n_depth):
    # This empowerment solver assumes determinism in all representations
    # xt_mat = np.outer(x_vec, t_vec)
    leaves = tgmdp_bredth_first_plan_search_spa_deterministic_fast(x_vec, t_vec, internal_vecs_init, internal_op_list, eta_ens, p_axt_x, omega_list, p_v_list, pi_ens, pi_ind_list, f, n_depth)
    state_vector_list = []
    for leaf in leaves:
        int_state_index_vec = []
        x = np.where(leaf.x_vec)[0].item()
        t = np.where(leaf.t_vec)[0].item()
        int_state_index_vec = [x, t]
        for i in range(len(leaf.ho_vecs)):
            int_state_index_vec.append(np.where(leaf.ho_vecs[i])[0][0])

        state_vector_list.append(np.array(int_state_index_vec))
    state_vector_array = np.array(state_vector_list)
    unique_rows = np.unique(state_vector_array, axis=0)
    empowerment = np.log2(len(unique_rows))

    return empowerment


def deterministic_empowerment_bfs_stationary_fast(x_vec, t_vec, internal_vecs_init, internal_op_list, eta_ens, p_ax_x, omega_list, p_v_list, pi_ens, pi_ind_list,
                                x_g_to_action_map, f, n_depth):
    # This empowerment solver assumes determinism in all representations
    # xt_mat = np.outer(x_vec, t_vec)
    n_bfs_start_time = timeit.default_timer()
    leaves = tgmdp_bredth_first_plan_search_spa_deterministic_stationary_fast(x_vec, t_vec, internal_vecs_init,
                                                                              internal_op_list, eta_ens, p_ax_x,
                                                                              omega_list, p_v_list, pi_ens, pi_ind_list,
                                                                              f, n_depth)
    # print("n-depth bfs time: ", timeit.default_timer() - n_bfs_start_time)
    number_of_leaves = len(leaves)
    state_vector_list = []
    unique_leaves_start_time = timeit.default_timer()
    for ind, leaf in enumerate(leaves):
        # if ind % 10 == 0: print('emp_leaf: ', ind, '/', number_of_leaves)
        int_state_index_vec = []
        # x = np.where(leaf.x)[0].item()
        # t = np.where(leaf.t_vec)[0].item()
        int_state_index_vec = [leaf.x, leaf.t]
        int_state_index_vec.extend(leaf.ho_vecs)
        # for i in range(len(leaf.ho_vecs)):
        #     pass

        state_vector_list.append(np.array(int_state_index_vec))
    state_vector_array = np.array(state_vector_list)
    unique_rows = np.unique(state_vector_array, axis=0)
    empowerment = np.log2(len(unique_rows))
    # print("unique leaves time: ", timeit.default_timer() - unique_leaves_start_time)
    # print('Final_emp: ', empowerment)

    return empowerment



def get_identity_state_actions(p_ax_x):
    # Returns [na,nx] matrix of boolean values which says which state-actions are identity operations and a vector of
    # length nx which says that a given state has an identity state-action
    na = p_ax_x.shape[0]
    nx = p_ax_x.shape[1]

    three_d_id = np.dstack([np.identity(nx)] * na).swapaxes(0,2)
    id_ax = np.sum(p_ax_x * three_d_id, axis=2)
    id_x = np.any(id_ax, axis=0).squeeze()

    return id_x, id_ax


def fill_in():
    return []

# Indicator Function for feasibility.  Returns 1 if feasible from state x.
def feas(kappa):
    return np.ceil(kappa)

# Indicator Function for infeasibility.  Returns 1 if infeasible from state x.
def infeas(kappa):
    return np.abs(1-np.ceil(kappa))

def risk_threshold(kappa_c, theta):
     vec = kappa_c <= theta
     return vec

def e_ttg_mat(eta):
    times = np.arange(eta.shape[2])
    etime_mat = np.dot(eta, times)
    return etime_mat

def e_ttg_vec(eta):
    times = np.arange(eta.shape[2])
    etime = np.dot(eta, times).sum(axis=1)
    return

def e_ttg_vec_rs(eta, ny, nx):
    times = np.arange(eta.shape[2])
    etime_vec = np.dot(eta, times).sum(axis=1)
    return np.flip(etime_vec.reshape(ny, nx), axis=0)



# def chapman_kolmo_forward(xt_mat, eta):
#     # Chapman Kolmogorov Kernel Composition.
#     # Takes in two etas and an initial state-time probability matrix v.  Outputs a state-time probability vector
#     C = chapman_kolmo_forward(xt_mat, eta)
#     xt_mat_new = C.sum(axis=0)
#     return xt_mat_new


# def chapman_kolmo_forward(xt_mat, eta):
#     # Takes in a state-time matrix and convolves each slice of eta. Outputs a time-extended eta.
#
#     x_vec = xt_mat.sum(axis=1)
#     mat_lst = []
#     for i in range(len(x_vec)):
#         mat_lst.append(np.array([signal.convolve(xt_mat[i], eta[i, j, :], mode='full') for j in range(eta.shape[1])]))
#     summed_mats = sum(mat_lst)
#     return summed_mats

def chapman_kolmo_forward_inefficient(xt_mat, eta):
    """TOM'S ORIGINAL implementation, kept verbatim as a reference. Renamed and preserved here by
    Claude when the faster version below was written; the body is unchanged.

    Superseded by chapman_kolmo_forward, which skips the (i, j) pairs of eta that are entirely
    zero -- typically ~99% of them, since eta is stored dense but is very sparse in content. The
    two are numerically IDENTICAL, because convolve(x, 0) is 0 and adding it changes nothing;
    test_chapman_sparse.py asserts exact equality between them.

    Kept so the change is auditable and reversible, and so the faster version always has an oracle
    to be checked against. Not used by anything.
    """
    # Takes in a state-time matrix and convolves each slice of eta.
    # Outputs a time-extended eta.

    x_vec = xt_mat.sum(axis=1)
    output_length = xt_mat.shape[1] + eta.shape[2] - 1  # Length after convolution
    xt_final = np.zeros((eta.shape[1], output_length))  # Initialize the output matrix

    non_zero_indices = np.nonzero(x_vec)[0]  # Indices where x_vec is non-zero

    for i in non_zero_indices:
        # Perform convolution for non-zero entries only
        conv_results = np.array([
            signal.convolve(xt_mat[i], eta[i, j, :], mode='full')
            for j in range(eta.shape[1])
        ])
        xt_final += conv_results  # Accumulate the convolution results

    return xt_final


def chapman_kolmo_forward(xt_mat, eta, nz_pairs=None, method='auto'):
    # Takes in a state-time matrix and convolves each slice of eta.
    # Outputs a time-extended eta.
    #
    # MODIFIED BY CLAUDE (the body below; the signature gained two optional arguments).
    # SPARSITY: eta is stored dense but is ~99% empty -- for a 9x9 grid only about 81 of the 6561
    # (i, j) pairs carry any mass at all. The inner loop used to convolve every j for each nonzero
    # row i, so almost all of that work was convolving against all-zero series and adding zeros.
    # Skipping those pairs is numerically IDENTICAL -- convolve(x, 0) is 0, and adding it changes
    # nothing -- and test_chapman_sparse.py asserts exact equality against the old implementation.
    #
    # `nz_pairs` is that mask, optional and purely for speed. Finding it costs an O(nx^2 * T) scan,
    # which once the wasted convolutions are gone is itself the dominant cost -- so a caller that
    # propagates MANY distributions through the SAME eta (composing kernels does exactly that,
    # once per start state) should compute it once and pass it in. Omitted, it is derived here and
    # behaviour is unchanged.
    #
    # `method` is passed straight to scipy.signal.convolve. Left at the scipy default ('auto') for
    # every EXISTING caller and for test_chapman_sparse.py's own bit-for-bit comparison against
    # chapman_kolmo_forward_inefficient, which stays on 'auto' too and would fail that comparison
    # under any other method -- FFT convolution is numerically close to direct, not bit-identical.
    # compose_stok_pair (tutorial_server.py) passes method='fft' explicitly: scipy's own 'auto'
    # heuristic picks 'direct' for the ~600-3000-sample arrays a long policy string produces, but
    # measured FFT is 2-3.5x faster there and the gap WIDENS as the string (and so the time axis)
    # grows, direct being O(T1*T2) against FFT's O((T1+T2)log(T1+T2)) -- exactly the mechanism
    # behind composing a long string getting progressively, noticeably slower per additional step.

    x_vec = xt_mat.sum(axis=1)
    output_length = xt_mat.shape[1] + eta.shape[2] - 1  # Length after convolution
    xt_final = np.zeros((eta.shape[1], output_length))  # Initialize the output matrix

    non_zero_indices = np.nonzero(x_vec)[0]  # Indices where x_vec is non-zero
    if nz_pairs is None:
        nz_pairs = np.any(eta != 0, axis=2)  # (i, j) pairs with any mass across time

    for i in non_zero_indices:
        for j in np.nonzero(nz_pairs[i])[0]:
            xt_final[j] += signal.convolve(xt_mat[i], eta[i, j, :], mode='full', method=method)

    return xt_final


def compute_cvar(reward_dist, bins, alpha=0.05):
    """
    Compute CVaR for a discrete probability distribution

    Args:
        reward_dist: probability mass function (your array)
        bins: corresponding reward values
        alpha: confidence level (e.g., 0.05 for 95% CVaR)
    """
    # Create cumulative distribution function
    cdf = np.cumsum(reward_dist)

    # Find VaR (alpha-quantile)
    var_index = np.searchsorted(cdf, alpha)
    var = bins[var_index] if var_index < len(bins) else bins[-1]

    # Compute CVaR as expected value of tail beyond VaR
    tail_mask = bins <= var
    tail_probs = reward_dist[tail_mask]
    tail_values = bins[tail_mask]

    # Normalize tail probabilities
    tail_prob_sum = np.sum(tail_probs)
    if tail_prob_sum > 0:
        cvar = np.sum(tail_values * tail_probs) / tail_prob_sum
    else:
        cvar = var

    return var, cvar


def plot_reward_dist_with_cvar(reward_dist, bins, alpha=0.05, show_var=True, show_cvar=True):
    """
    Plot reward distribution with optional VaR and CVaR indicators

    Args:
        reward_dist: probability mass function
        bins: corresponding reward values
        alpha: confidence level for VaR computation
        show_var: whether to show VaR threshold and color coding
        show_cvar: whether to show CVaR line
    """
    # Compute VaR and CVaR
    cdf = np.cumsum(reward_dist)
    var_index = np.searchsorted(cdf, alpha)
    var = bins[var_index] if var_index < len(bins) else bins[-1]

    tail_mask = bins <= var
    tail_probs = reward_dist[tail_mask]
    tail_values = bins[tail_mask]
    tail_prob_sum = np.sum(tail_probs)

    if tail_prob_sum > 0:
        cvar = np.sum(tail_values * tail_probs) / tail_prob_sum
    else:
        cvar = var

    # Set colors based on show_var flag
    if show_var:
        colors = ['red' if bin_val <= var else 'blue' for bin_val in bins]
    else:
        colors = 'blue'  # All bars blue

    # Plot
    plt.figure(figsize=(10, 4))
    plt.bar(bins, reward_dist, width=0.008, color=colors)

    # Add VaR line if requested
    if show_var:
        boundary = var + 0.005  # Half of bin width (0.01/2)
        plt.axvline(x=boundary, color='grey', linestyle='--', linewidth=2, alpha=0.7,
                    label=f'VaR ({(1 - alpha) * 100:.0f}%): {var:.3f}')

    # Add CVaR line if requested
    if show_cvar:
        plt.axvline(x=cvar, color='orange', linestyle=':', linewidth=2, alpha=0.8,
                    label=f'CVaR ({(1 - alpha) * 100:.0f}%): {cvar:.3f}')

    plt.xlabel('Reward Value')
    plt.ylabel('Probability Mass')
    plt.title('Reward Distribution')
    # plt.xlim(0, 1.5)

    # Only show legend if we have lines to show
    if show_var or show_cvar:
        plt.legend()

    plt.show()

    # Print values
    print(f"VaR ({(1 - alpha) * 100:.0f}%): {var:.4f}")
    print(f"CVaR ({(1 - alpha) * 100:.0f}%): {cvar:.4f}")

    return var, cvar


def chapman_kolmo_forward_on_rewards(xtr_tensor, eta_r):
    # Takes in a state-time matrix and convolves each slice of eta.
    # Outputs a time-extended eta.

    x_vec = xtr_tensor.sum(axis=(1, 2))
    output_length_time = xtr_tensor.shape[1] + eta_r.shape[2] - 1  # Length after convolution
    output_length_reward = xtr_tensor.shape[2] + eta_r.shape[3] - 1  # Length after convolution
    xtr_final = np.zeros((eta_r.shape[1], output_length_time, output_length_reward))  # Initialize the output matrix

    non_zero_indices = np.nonzero(x_vec)[0]  # Indices where x_vec is non-zero

    for i in non_zero_indices:
        # Perform convolution for non-zero entries only
        conv_results = np.array([
            scipy.signal.convolve2d(xtr_tensor[i], eta_r[i, j, :, :], mode='full')
            for j in range(eta_r.shape[1])
        ])
        xtr_final += conv_results  # Accumulate the convolution results

    return xtr_final


def feas_iter_stationary_flat_dense(p_ax_x, g, c, max_time = 100, fill_in = False, wallmat = [], use_ET = False, xspace=[]):
    # p_axt_x must be [na,nx,T_f,nx']:  "p_axt_x(x'|x,a,t)" in the theory
    # f_g must be [na, nx]
    na, nx, _ = p_ax_x.shape
    # x_g = np.where(f_g > 0)[1][0]  # Not actually needed, just for diagnostics
    # a_g = np.where(f_c[:, x_g] > 0)[1][0]  # Not actually needed, just for diagnostics
    # NOTE: the `max_time` argument is intentionally overridden here with a diameter-based
    # iteration cap (~4*sqrt(nx)); the caller-supplied value is ignored.
    max_time = int(np.sqrt(nx)) * 2 * 2
    T_f = max_time
    kappa = np.zeros([nx])
    kappa_c = np.zeros([nx])  # Cumulative constraint violation, denoted by _c
    eta_pos = np.zeros([nx, nx, T_f])  # eta^+(t_f,x_f|x)
    eta_neg = np.zeros([nx, nx, T_f])  # eta^-(t_f,x_f|x)
    pi = np.zeros([nx]).astype('int')
    ET = np.zeros([nx])

    # Caution: This assumes p_axt_x is stationary (even though it is indexed by time).  If it is non-stationary, one needs to compute time-dependent identity state-actions.
    # id_x, id_ax = get_identity_state_actions(p_axt_x[:,:,0,:])


    f = g * c   # Achievement Function
    h = (1 - g) * c   # Continuation Function

    time_func = np.ones([na, nx])  # The time-function is essentially a cost function over X for accumulating time.
    time_func[np.where(f > 0)] = 0

    # beta = feas * (f + (1 - f_c)) + (1 - feas)  # termination function

    # Boundary Conditions
    kappa[:] = np.amax(f[:, :], axis=0)
    pi[:] = np.argmax(g[:, :], axis=0).astype('int')
    ET[:] = f[pi[:], range(nx)]
    eta_pos[range(nx), range(nx), 0] = f[pi[:], range(nx)]
    eta_neg[range(nx), range(nx), 0] = feas(kappa) * (1 - c[pi[:], range(nx)]) + infeas(kappa)
    q = np.ones_like(g) * (1 - g)  # cost function of one for expected time computation
    # For computing average time minimization
    times_arr = np.arange(T_f) + 1

    x_inds = np.arange(nx)
    delta = np.inf
    epsilon = 0.005
    t = 0  # absolute time
    tau = 0  # time relative to last eta-extension
    while t < max_time:
        st = timeit.default_timer()
        eta_pos_old = eta_pos.copy()
        eta_neg_old = eta_neg.copy()
        st = timeit.default_timer()
        old_kappa = kappa.copy()
        feas_indic = feas(kappa)
        eta_contraction_ax_xt = np.tensordot(p_ax_x, eta_pos, axes=([2, 0]))
        future_kappa_w_actions_ax = np.sum(eta_contraction_ax_xt, axis=(2, 3))
        print("time1: ", timeit.default_timer() - st)
        kappa_w_actions_ax = f + np.multiply(h, future_kappa_w_actions_ax)
        # NOTE: rounding to 4 decimals imposes an IMPLICIT feasibility threshold of 1e-4
        # (values below flush to 0), so feas()=ceil(kappa) treats ϱ<1e-4 as infeasible.
        kappa_w_actions_ax = np.round(kappa_w_actions_ax, 4)
        kappa = np.max(kappa_w_actions_ax, axis=0)
        # kappa_nz_inds = np.where(kappa > 0)[0]
        # kappa_ax_nz_inds = np.where(kappa_w_actions_ax > 0)
        # best_inds = np.where(kappa_w_actions_ax == kappa)  # A^*_x set in the OBEs
        not_best_inds = np.where(kappa_w_actions_ax != kappa)  # Used to set suboptimal inds to inf
        pi_kappa = np.argmax(kappa_w_actions_ax, axis=0)
        print("time2: ", timeit.default_timer() - st)

        st = timeit.default_timer()
        # Conditional Time Minimization
        if use_ET:
            st = timeit.default_timer()
            ax_time = q + np.tensordot(p_ax_x, ET, axes=([2, 0]))
            # ax_time = h * np.tensordot(p_ax_x, ET, axes=([2, 0]))
            ax_time[not_best_inds[0], not_best_inds[1]] = np.infty  # set suboptimal inds to inf
            pi = np.argmin(ax_time, axis=0).astype('int')  # compute minimum expected time
            ET = (feas(kappa) * q[pi[:], range(nx)]) + np.dot(p_ax_x[pi[:], range(nx), :], ET)  # update expected time vector
            # print("time3: ", timeit.default_timer() - st)
        else:
            st = timeit.default_timer()
            eta_contraction_ax_avt = np.dot(eta_contraction_ax_xt, times_arr).sum(axis=-1)
            eta_contraction_ax_avt[not_best_inds[0], not_best_inds[1]] = np.infty
            pi = np.argmin(eta_contraction_ax_avt, axis=0).astype('int')
            # print("time3.5: ", timeit.default_timer() - st)

        # Read policy dynamics AFTER pi has been updated this iteration (was previously
        # computed with the stale, previous-iteration pi). Used for the quiver diagnostic.
        pol_dynamics = p_ax_x[pi[:], range(nx), :]

        st = timeit.default_timer()

        # Test: Since pi was optimized with conditional minimization, then:
        if np.any(kappa_w_actions_ax[pi_kappa, x_inds] != kappa_w_actions_ax[pi, x_inds]):
            print("whoa nelly!")


        # Fail Eta Expectation
        eta_fail_contraction_ax_xt = np.tensordot(p_ax_x, eta_neg, axes=([2, 0]))

        new_eta_pos = np.multiply(h[pi, x_inds, np.newaxis, np.newaxis], eta_contraction_ax_xt[pi, x_inds, :, :])
        eta_pos[:, :, 1:] = new_eta_pos[:, :, :-1].copy()
        eta_pos[range(nx), range(nx), 0] = feas_indic*f[pi[:], x_inds].copy()
        print("time4: ", timeit.default_timer() - st)
        st = timeit.default_timer()

        print("Time Step: ", t)

        new_eta_neg = feas_indic[:, np.newaxis, np.newaxis] * np.multiply(h[pi, x_inds, np.newaxis, np.newaxis], eta_fail_contraction_ax_xt[pi, x_inds, :, :])
        eta_neg[:, :, 1:] = new_eta_neg[:, :, :-1].copy()
        eta_neg[range(nx), range(nx), 0] = (feas(kappa) * (1 - c[pi, x_inds]) + infeas(kappa)).copy()
        # utilities.plotgrid2(kappa, 0.2)
        # utilities.plotgrid2(kappa_c, 0.2)
        print("time5: ", timeit.default_timer() - st)
        st = timeit.default_timer()
        if not np.all(eta_neg.sum(axis=1).sum(axis=1)+eta_pos.sum(axis=1).sum(axis=1)):
            warnings.warn("Does not sum to one!")

        # Extend eta time dimension if needed.
        if tau >= T_f-2:
            eta_pos = np.pad(eta_pos, ((0, 0), (0, 0), (0, T_f)), 'constant', constant_values=0)
            eta_neg = np.pad(eta_neg, ((0, 0), (0, 0), (0, T_f)), 'constant', constant_values=0)
            times_arr = np.arange(eta_pos.shape[2]) + 1
            tau = 0

        # pos_exp_time_old = eta_pos_old.dot(times_arr).sum(axis=1)
        # pos_exp_time = eta_pos.dot(times_arr).sum(axis=1)
        # neg_exp_time_old = eta_neg_old.dot(times_arr).sum(axis=1)
        # neg_exp_time = eta_neg.dot(times_arr).sum(axis=1)
        # delta_eta_pos = np.sum(np.abs(pos_exp_time - pos_exp_time_old))
        # utilities.plotgrid(kappa - old_kappa, 1, wallmat)
        # print("Eta_pos Delta: ", delta_eta_pos)
        delta = np.sum(np.abs(kappa - old_kappa))
        print("Delta = ", delta)
        if delta <= epsilon:
            print("time6: ", timeit.default_timer() - st)
            break

        t = t + 1
        tau = tau + 1
        print("time6: ", timeit.default_timer() - st)

    if np.all(np.round(eta_pos.sum(axis=1).sum(axis=1), 4) == np.round(kappa, 4)) == False:
        warnings.warn("Warning: Kappa not equal to summed Eta")

    # Kappa and eta should be equal when final states and times are summed over in eta
    # np.all(eta[:,:,:].sum(axis=2)==kappa[:,:]) should return True
    # utilities.polquiver(pol_dynamics, xspace.liToCoord, xspace.wall_list, xspace.wallmat)
    print("Total Stationary Feasibility Steps: ", t)

    beta = feas_indic * (f + (1 - c)) + (1 - feas_indic)  # Termination Function ("beta" by convention).

    return kappa, pi, eta_pos, eta_neg, beta


def feas_iter_stationary_flat_dense_thresh(p_ax_x, p_ax_x_det, g, c, kappa_c, max_time = 100, risk_thresh = 1, fill_in = False, wallmat = [], use_ET = False, xspace=[]):
    # p_axt_x must be [na,nx,T_f,nx']:  "p_axt_x(x'|x,a,t)" in the theory
    # f_g must be [na, nx]
    na, nx, _ = p_ax_x.shape
    # x_g = np.where(f_g > 0)[1][0]  # Not actually needed, just for diagnostics
    # a_g = np.where(f_c[:, x_g] > 0)[1][0]  # Not actually needed, just for diagnostics
    T_const = int(np.sqrt(nx)) * 2 * 2
    kappa = np.zeros([nx])
    eta_pos = np.zeros([nx, nx, T_const])  # eta^+(t_f,x_f|x)
    eta_neg = np.zeros([nx, nx, T_const])  # eta^-(t_f,x_f|x)
    pi = np.zeros([nx]).astype('int')
    ET = np.zeros([nx])

    # Caution: This assumes p_axt_x is stationary (even though it is indexed by time).  If it is non-stationary, one needs to compute time-dependent identity state-actions.
    # id_x, id_ax = get_identity_state_actions(p_axt_x[:,:,0,:])
    r_thresh = risk_threshold(kappa_c, risk_thresh).astype(int)
    c_thresh = np.maximum(c, r_thresh)
    f_thresh = g * r_thresh  # Achievement Function Risk-Thresholded
    h_thresh = (1 - g) * r_thresh # Continuation Function

    f = g * c   # Achievement Function
    h = (1 - g) * c   # Continuation Function

    time_func = np.ones([na, nx])  # The time-function is essentially a cost function over X for accumulating time.
    time_func[np.where(f_thresh > 0)] = 0

    # beta = feas * (f + (1 - f_c)) + (1 - feas)  # termination function

    # Boundary Conditions
    kappa[:] = np.amax(f_thresh[:, :], axis=0)
    pi[:] = np.argmax(f_thresh[:, :], axis=0).astype('int')
    ET[:] = f_thresh[pi[:], range(nx)]
    eta_pos[range(nx), range(nx), 0] = f_thresh[pi[:], range(nx)]
    eta_neg[range(nx), range(nx), 0] = feas(kappa) * (1 - c_thresh[pi[:], range(nx)]) + infeas(kappa)
    q = np.ones_like(g) * (1 - g)  # cost function of one for expected time computation
    # For computing average time minimization
    times_arr = np.arange(T_const) + 1

    x_inds = np.arange(nx)
    delta = np.inf
    epsilon = 0.005
    t = 0  # absolute time
    tau = 0  # time relative to last eta-extension
    while t < max_time:
        st = timeit.default_timer()
        eta_pos_old = eta_pos.copy()
        eta_neg_old = eta_neg.copy()
        st = timeit.default_timer()
        old_kappa = kappa.copy()
        future_kappa_w_actions_ax = np.tensordot(p_ax_x_det, kappa, axes=([2, 0]))
        print("time1: ", timeit.default_timer() - st)
        kappa_w_actions_ax = f_thresh + np.multiply(h_thresh, future_kappa_w_actions_ax)
        kappa_w_actions_ax = np.round(kappa_w_actions_ax, 4)  # Rounding pervents numerical instability problems when optimizing kappa
        kappa = np.max(kappa_w_actions_ax, axis=0)
        # kappa_nz_inds = np.where(kappa > 0)[0]
        # kappa_ax_nz_inds = np.where(kappa_w_actions_ax > 0)
        # best_inds = np.where(kappa_w_actions_ax == kappa)  # A^*_x set in the OBEs
        not_best_inds = np.where(kappa_w_actions_ax != kappa)  # Used to set suboptimal inds to inf
        pi_kappa = np.argmax(kappa_w_actions_ax, axis=0)
        print("time2: ", timeit.default_timer() - st)

        st = timeit.default_timer()
        # Conditional Time Minimization
        if use_ET:
            st = timeit.default_timer()
            ax_time = q + np.tensordot(p_ax_x_det, ET, axes=([2, 0]))
            # ax_time = h * np.tensordot(p_ax_x, ET, axes=([2, 0]))
            ax_time[not_best_inds[0], not_best_inds[1]] = np.infty  # set suboptimal inds to inf
            pi = np.argmin(ax_time, axis=0).astype('int')  # compute minimum expected time
            ET = (feas(kappa) * q[pi[:], range(nx)]) + np.dot(p_ax_x_det[pi[:], range(nx), :], ET)  # update expected time vector
            # print("time3: ", timeit.default_timer() - st)
        else:
            warnings.warn("Not implemented")

        # Read policy dynamics AFTER pi has been updated this iteration (was previously
        # computed with the stale, previous-iteration pi). Used for the quiver diagnostic.
        pol_dynamics = p_ax_x_det[pi[:], range(nx), :]
        # utilities.polquiver(pol_dynamics, xspace.liToCoord, xspace.wall_list, xspace.wallmat)

        st = timeit.default_timer()

        # Test: Since pi was optimized with conditional minimization, then:
        if np.any(kappa_w_actions_ax[pi_kappa, x_inds] != kappa_w_actions_ax[pi, x_inds]):
            print("whoa nelly!")


        # Fail Eta Expectation
        eta_fail_contraction_ax_xt = np.tensordot(p_ax_x, eta_neg, axes=([2, 0]))
        eta_pos_contraction_ax_xt = np.tensordot(p_ax_x, eta_pos, axes=([2, 0]))

        new_eta_pos = np.multiply(h_thresh[pi, x_inds, np.newaxis, np.newaxis], eta_pos_contraction_ax_xt[pi, x_inds, :, :])
        eta_pos[:, :, 1:] = new_eta_pos[:, :, :-1].copy()
        eta_pos[range(nx), range(nx), 0] = f_thresh[pi[:], x_inds].copy()
        print("time4: ", timeit.default_timer() - st)
        st = timeit.default_timer()

        print("Time Step: ", t)

        new_eta_neg = feas(kappa)[:, np.newaxis, np.newaxis] * np.multiply(h_thresh[pi, x_inds, np.newaxis, np.newaxis], eta_fail_contraction_ax_xt[pi, x_inds, :, :])
        eta_neg[:, :, 1:] = new_eta_neg[:, :, :-1].copy()
        eta_neg[range(nx), range(nx), 0] = (feas(kappa) * (1 - c[pi, x_inds]) + infeas(kappa)).copy()
        # utilities.plotgrid2(kappa, 0.2)
        # utilities.plotgrid2(kappa_c, 0.2)
        print("time5: ", timeit.default_timer() - st)
        st = timeit.default_timer()
        if not np.all(eta_neg.sum(axis=1).sum(axis=1) + eta_pos.sum(axis=1).sum(axis=1)):
            warnings.warn("Does not sum to one!")

        # Extend eta time dimension if needed.
        if tau >= T_const-2:
            eta_pos = np.pad(eta_pos, ((0, 0), (0, 0), (0, T_const)), 'constant', constant_values=0)
            eta_neg = np.pad(eta_neg, ((0, 0), (0, 0), (0, T_const)), 'constant', constant_values=0)
            times_arr = np.arange(eta_pos.shape[2]) + 1
            tau = 0
        delta = np.sum(np.abs(kappa - old_kappa))
        # BUGFIX: eta_*_old was snapshotted BEFORE the np.pad above, so after the horizon grows it
        # still has the OLD number of time slots while times_arr has been regrown to the new length.
        # Dotting them then raised "shapes (nx,nx,T) and (2T,) not aligned". times_arr is just
        # arange+1, so its prefix is the correct time vector for the shorter array.
        # Only fires once a solve exceeds T_const-2 iterations, which is why it went unnoticed --
        # the equivalent lines in feas_iter_stationary_flat_dense are commented out.
        t_old = np.arange(eta_pos_old.shape[2]) + 1
        pos_exp_time_old = eta_pos_old.dot(t_old).sum(axis=1)
        pos_exp_time = eta_pos.dot(times_arr).sum(axis=1)
        neg_exp_time_old = eta_neg_old.dot(np.arange(eta_neg_old.shape[2]) + 1).sum(axis=1)
        neg_exp_time = eta_neg.dot(times_arr).sum(axis=1)
        delta_eta_pos = np.sum(np.abs(pos_exp_time - pos_exp_time_old))
        # utilities.plotgrid(kappa - old_kappa, 1, wallmat)
        print("Eta_pos Delta: ", delta_eta_pos)
        delta = np.sum(np.abs(kappa - old_kappa))
        print("Delta = ", delta)
        if delta <= epsilon:
            print("time6: ", timeit.default_timer() - st)
            break

        t = t + 1
        tau = tau + 1
        print("time6: ", timeit.default_timer() - st)

    if np.all(np.round(eta_pos.sum(axis=1).sum(axis=1), 4) == np.round(kappa, 4)) == False:
        warnings.warn("Warning: Kappa not equal to summed Eta")

    # Kappa and eta should be equal when final states and times are summed over in eta
    # np.all(eta[:,:,:].sum(axis=2)==kappa[:,:]) should return True
    # utilities.plotgrid2(kappa, 0.2)
    # utilities.plotgrid2(kappa_c, 0.2)
    # -- left commented like the identical pair inside the loop above (line ~2084): debug-only,
    # and unconditional here where the loop's copy at least ran inside a print-heavy debug section.
    # plotgrid2 only reshapes its 1D input to 2D when data.size == utilities.ROW_COUNT *
    # utilities.COLUMN_COUNT (hardcoded to 9x9 in config.py) -- on any other grid size it hands
    # plt.imshow a flat array and crashes with "Invalid shape (N,) for image data", which is what
    # this function did on every call for a non-9x9 world (e.g. the risk-thresholded algorithm
    # picked in the tutorial app on a resized grid). Nothing downstream of this function uses a
    # plot; it exists only to eyeball kappa/kappa_c interactively, which a caller who wants that
    # can still do with the returned kappa themselves.
    # beta = feas * (f + (1 - f_c)) + (1 - feas)  # termination function
    print("Total Stationary Feasibility Steps: ", t)
    return kappa, pi, eta_pos, eta_neg


# TODO: Figure out if replacing tensordot with (np.dot with a tall matrix tallmat_p_ax_x) makes code run faster.
def feasibility_iteration_flat_dense(p_axt_x, fg_axt, p_ax_x = None, fill_in = False):
    # p_axt_x must be [na,nx,T_f,nx']:  "p_axt_x(x'|x,a,t)" in the theory
    # f must be [nx,T_f]
    na, nx, T_f, _ = p_axt_x.shape
    x_g = np.where(fg_axt > 0)[1][0]
    a_g = np.where(fg_axt[:, x_g, :] > 0)[1][0]
    kappa = np.zeros([nx, T_f])
    eta = np.zeros([nx, T_f, T_f]) # Typicially eta(x_f,t_f|x,t) should be [nx,T_f,nx,T_f] or [ng,T_f,nx,T_f], however assuming f encodes a single goal, we can drop the index
    pi = np.zeros([nx, T_f]).astype('int')

    # Caution: This assumes p_axt_x is stationary (even though it is indexed by time).  If it is non-stationary, one needs to compute time-dependent identity state-actions.
    # id_x, id_ax = get_identity_state_actions(p_axt_x[:,:,0,:])

    # Boundary Conditions
    kappa[:, -1] = np.amax(fg_axt[:, :, -1], axis=0)
    pi[:, -1] = np.argmax(fg_axt[:, :, -1], axis=0).astype('int')
    eta[:, -1, -1] = fg_axt[pi[:, -1], range(nx), -1]

    # For computing average time minimization
    times_arr = np.arange(T_f) + 1  # +1 is for the needed time shift to avoid an error introduced from setting zeros to inf

    # Fill-in
    if fill_in:
        id_x, id_ax = get_identity_state_actions(p_ax_x)
        max_kappa_states = np.array([np.argmax(kappa[:, -1])])
        if np.any(kappa[max_kappa_states,-1]>0):
            min_time_states = np.array([max_kappa_states[np.argmin(np.dot(eta[max_kappa_states,-1,:], times_arr))]])
            # Fill-in states are the states that are jointly cumulative maximizing, time-minimizing and stable
            fill_in_states = np.intersect1d(np.intersect1d(np.arange(nx)[id_x], max_kappa_states), min_time_states)
            fill_in_actions = np.array([np.argmax(id_ax[:,fill_in_states], axis=0)])
            for x, a in np.array([fill_in_states, fill_in_actions]).transpose():
                for t in np.flip(np.arange(T_f - 1)):
                    future_kappa = kappa[x, t+1]
                    kappa[x, t] = fg_axt[a, x, t] + np.multiply((1 - fg_axt[a, x, t]), future_kappa)
                    pi[x, t] = a
                    eta[x, t, :] = (1 - fg_axt[a, x, t]) * np.tensordot(p_axt_x[a, x, t, :], eta[:, t + 1, :], axes=([0, 0]))

    max_time = np.max(np.where(fg_axt>0)[2]) # This sets the time to be the largest time for which there is positive availability. Do not change!
    x_inds = np.arange(nx)

    for t in reversed(np.arange(max_time)):
        eta_contraction_ax_t = np.tensordot(p_axt_x[:, :, t + 1, :], eta[:, t + 1, :], axes=([2, 0])) # condition on t
        future_kappa_w_actions_axt = np.sum(eta_contraction_ax_t, axis=2)
        kappa_w_actions_axt = fg_axt[:, :, t] + np.multiply((1 - fg_axt[:, :, t]), future_kappa_w_actions_axt)
        kappa_update = np.max(kappa_w_actions_axt, axis=0)
        kappa[:, t] = kappa_update
        best_inds = np.where(kappa_w_actions_axt == kappa_update)
        non_zero_inds = np.where(kappa_update>0)[0]

        # idx = np.ix_(best_inds[0], best_inds[1])
        best_eta_contraction_axt_t = np.zeros(eta_contraction_ax_t.shape)
        best_eta_contraction_axt_t[best_inds[0], best_inds[1], :] = eta_contraction_ax_t[best_inds[0], best_inds[1], :]
        eta_time_expec_2 = np.dot(best_eta_contraction_axt_t, times_arr)
        inner_eq = (fg_axt[:, :, t] * (t + 1)) + np.multiply(1 - fg_axt[:, :, t], eta_time_expec_2)  # "+1" is for the needed time shift to avoid an error introduced from setting zeros to inf. Do not change!
        inner_eq[inner_eq == 0] = np.inf
        pi[:, t] = np.argmin(inner_eq, axis=0).astype('int')
        new_eta = np.multiply((1 - fg_axt[pi[:, t], np.arange(nx), :]), eta_contraction_ax_t[pi[:, t], x_inds, :])
        eta[:, t, :] = new_eta
        eta[:, t, t] = fg_axt[pi[:, t], x_inds, t]

    if np.all(eta.sum(axis=2) == kappa) == False:
        warnings.warn("Warning: Kappa not equal to summed Eta")

    # Kappa and eta should be equal when final states and times are summed over in eta
    # np.all(eta[:,:,:].sum(axis=2)==kappa[:,:]) should return True

    return kappa, pi, eta


def feasibility_iteration_flat_dense_w_failure(p_axt_x, fg_axt, p_ax_x = None, fill_in = False):
    # TODO: I verified it works with determinisitic conditions. Check Stochastic to make sure that works too.
    # p_axt_x must be [na, nx, T_f, nx']:  "p_axt_x(x'|x,a,t)" in the theory
    # f must be [nx, T_f]
    na, nx, T_f, _ = p_axt_x.shape
    x_g = np.where(fg_axt > 0)[1][0]
    kappa = np.zeros([nx, T_f])
    eta = np.zeros([nx, T_f, T_f]) # Typicially eta(x_f,t_f|x,t) should be [nx,T_f,nx,T_f] or [ng,T_f,nx,T_f], however assuming f encodes a single goal, we can drop the index
    eta_fail = np.zeros([nx, T_f, T_f])
    pi = np.zeros([nx, T_f]).astype('int')

    # Caution: This assumes p_axt_x is stationary (even though it is indexed by time).  If it is non-stationary, one needs to compute time-dependent identity state-actions.
    # id_x, id_ax = get_identity_state_actions(p_axt_x[:,:,0,:])

    # Boundary Conditions
    kappa[:,-1] = np.amax(fg_axt[:, :, -1], axis=0)
    pi[:,-1] = np.argmax(fg_axt[:, :, -1], axis=0).astype('int')
    eta[:,-1,-1] = fg_axt[pi[:, -1], range(nx), -1]
    eta_fail[:, -1, -1] = 1-fg_axt[pi[:, -1], range(nx), -1]

    # For computing average time minimization
    times_arr = np.arange(T_f) + 1  # +1 is for the needed time shift to avoid an error introduced from setting zeros to inf

    # Fill-in. Set feasibility of known policy values for stable-states. (Not required and disabled by default.)
    if fill_in:
        id_x, id_ax = get_identity_state_actions(p_ax_x)
        max_kappa_states = np.array([np.argmax(kappa[:, -1])])
        if np.any(kappa[max_kappa_states,-1]>0):
            min_time_states = np.array([max_kappa_states[np.argmin(np.dot(eta[max_kappa_states,-1,:], times_arr))]])
            # Fill-in states are the states that are jointly cumulative maximizing, time-minimizing and stable
            fill_in_states = np.intersect1d(np.intersect1d(np.arange(nx)[id_x], max_kappa_states), min_time_states)
            fill_in_actions = np.array([np.argmax(id_ax[:,fill_in_states], axis=0)])
            for x, a in np.array([fill_in_states, fill_in_actions]).transpose():
                for t in np.flip(np.arange(T_f - 1)):
                    future_kappa = kappa[x, t+1]
                    kappa[x, t] = fg_axt[a, x, t] + np.multiply((1 - fg_axt[a, x, t]), future_kappa)
                    pi[x, t] = a
                    eta[x, t, :] = (1 - fg_axt[a, x, t]) * np.tensordot(p_axt_x[a, x, t, :], eta[:, t + 1, :], axes=([0, 0]))

    max_feasible_time = np.max(np.where(fg_axt>0)[2]) # This sets the time to be the largest time for which there is positive availability. Do not change!
    x_inds = np.arange(nx)

    final_time = min(max_feasible_time, T_f-2)

    # Main Loop of Feasibility Iteration.  Start at t=T_f and work backwards to t=0
    for t in reversed(np.arange(final_time + 1)):
        eta_contraction_ax_t = np.tensordot(p_axt_x[:, :, t, :], eta[:, t + 1, :], axes=([2, 0])) # condition on t
        future_kappa_w_actions_ax = np.sum(eta_contraction_ax_t, axis=2)
        kappa_w_actions_axt = fg_axt[:, :, t] + np.multiply((1 - fg_axt[:, :, t]), future_kappa_w_actions_ax)
        kappa_update = np.max(kappa_w_actions_axt, axis=0)
        kappa[:, t] = kappa_update
        best_inds = np.where(kappa_w_actions_axt == kappa_update)
        non_zero_inds = np.where(kappa_update>0)[0]

        # idx = np.ix_(best_inds[0], best_inds[1])
        best_eta_contraction_axt_t = np.zeros(eta_contraction_ax_t.shape)
        best_eta_contraction_axt_t[best_inds[0], best_inds[1], :] = eta_contraction_ax_t[best_inds[0], best_inds[1], :]
        eta_time_expec_2 = np.dot(best_eta_contraction_axt_t, times_arr + 1)
        inner_eq = (fg_axt[:, :, t] * (t + 1)) + np.multiply(1 - fg_axt[:, :, t], eta_time_expec_2)  # "+1" is for the needed time shift to avoid an error introduced from setting zeros to inf. Do not change!
        inner_eq[inner_eq == 0] = np.inf
        pi[:, t] = np.argmin(inner_eq, axis=0).astype('int')
        new_eta = np.multiply((1 - fg_axt[pi[:, t], x_inds, :]), eta_contraction_ax_t[pi[:, t], x_inds, :])
        eta[:, t, :] = new_eta
        eta[:, t, t] = fg_axt[pi[:, t], x_inds, t]

        is_gt_zero = kappa[:, t] > 0
        is_zero = np.logical_not(is_gt_zero)
        eta_fail[is_gt_zero, t, t] = 0
        eta_fail[is_gt_zero, t, (t + 1):] = np.tensordot(p_axt_x[pi[is_gt_zero, t], is_gt_zero, t, :], eta_fail[:, t + 1, (t + 1):], axes=([1, 0]))
        eta_fail[is_zero, t, t] = 1
        eta_fail[is_zero, t, (t + 1):] = 0
    
    # Compute eta_fail for the time beyond max_time to T_f that we did not optimze for because we knew it was infeasible
    known_infeasible_times = np.arange(max_feasible_time + 1, T_f)
    eta_fail[:, known_infeasible_times, known_infeasible_times] = 1

    if np.all((eta_fail + eta).sum(axis=2)[:]) == False:
        warnings.warn("Does not sum to one over success and failure.")
    if np.all(eta.sum(axis=2) == kappa) == False:
        warnings.warn("Kappa not equal to summed Eta")

    return kappa, pi, eta, eta_fail


def feasibility_iteration_flat_dense_w_failure_ns(xspace, fg_axt, T_f, p_ax_x=None, fill_in=False):
    # TODO: I verified it works with determinisitic conditions. Check Stochastic to make sure that works too.
    # ns stands for non-stationary.  This version takes in xspace so that we can index the correct dynamics at a given time t
    # p_axt_x must be [na,nx,T_f,nx']:  "p_axt_x(x'|x,a,t)" in the theory
    # f must be [nx,T_f]
    p_ax_x = xspace.get_p_ax_x_at_time_t(0)
    na, nx, _ = p_ax_x.shape
    x_g = np.where(fg_axt > 0)[1][0]
    kappa = np.zeros([nx, T_f])
    eta = np.zeros([nx, T_f,
                    T_f])  # Typicially eta(x_f,t_f|x,t) should be [nx,T_f,nx,T_f] or [ng,T_f,nx,T_f], however assuming f encodes a single goal, we can drop the index
    eta_fail = np.zeros([nx, T_f, T_f])
    pi = np.zeros([nx, T_f]).astype('int')

    # Caution: This assumes p_axt_x is stationary (even though it is indexed by time).  If it is non-stationary, one needs to compute time-dependent identity state-actions.
    # id_x, id_ax = get_identity_state_actions(p_axt_x[:,:,0,:])

    # Boundary Conditions
    kappa[:, -1] = np.amax(fg_axt[:, :, -1], axis=0)
    pi[:, -1] = np.argmax(fg_axt[:, :, -1], axis=0).astype('int')
    eta[:, -1, -1] = fg_axt[pi[:, -1], range(nx), -1]
    eta_fail[:, -1, -1] = 1 - fg_axt[pi[:, -1], range(nx), -1]

    # For computing average time minimization
    times_arr = np.arange(
        T_f) + 1  # +1 is for the needed time shift to avoid an error introduced from setting zeros to inf

    # Fill-in
    if fill_in:
        id_x, id_ax = get_identity_state_actions(p_ax_x)
        max_kappa_states = np.array([np.argmax(kappa[:, -1])])
        if np.any(kappa[max_kappa_states, -1] > 0):
            min_time_states = np.array([max_kappa_states[np.argmin(np.dot(eta[max_kappa_states, -1, :], times_arr))]])
            # Fill-in states are the states that are jointly cumulative maximizing, time-minimizing and stable
            fill_in_states = np.intersect1d(np.intersect1d(np.arange(nx)[id_x], max_kappa_states), min_time_states)
            fill_in_actions = np.array([np.argmax(id_ax[:, fill_in_states], axis=0)])
            for x, a in np.array([fill_in_states, fill_in_actions]).transpose():
                for t in np.flip(np.arange(T_f - 1)):
                    p_ax_x = xspace.get_p_ax_x_at_time_t(t)
                    future_kappa = kappa[x, t + 1]
                    kappa[x, t] = fg_axt[a, x, t] + np.multiply((1 - fg_axt[a, x, t]), future_kappa)
                    pi[x, t] = a
                    eta[x, t, :] = (1 - fg_axt[a, x, t]) * np.tensordot(p_ax_x[a, x, :], eta[:, t + 1, :],
                                                                        axes=([0, 0]))

    max_feasible_time = np.max(np.where(fg_axt > 0)[
                                   2])  # This sets the time to be the largest time for which there is positive availability. Do not change!
    x_inds = np.arange(nx)

    final_time = min(max_feasible_time, T_f - 2)
    for t in reversed(np.arange(final_time + 1)):
        p_ax_x = xspace.get_p_ax_x_at_time_t(t)
        eta_contraction_ax_t = np.tensordot(p_ax_x, eta[:, t + 1, :], axes=([2, 0]))  # condition on t
        future_kappa_w_actions_ax = np.sum(eta_contraction_ax_t, axis=2)
        kappa_w_actions_axt = fg_axt[:, :, t] + np.multiply((1 - fg_axt[:, :, t]), future_kappa_w_actions_ax)
        kappa_update = np.max(kappa_w_actions_axt, axis=0)
        kappa[:, t] = kappa_update
        best_inds = np.where(kappa_w_actions_axt == kappa_update)
        non_zero_inds = np.where(kappa_update > 0)[0]

        # idx = np.ix_(best_inds[0], best_inds[1])
        best_eta_contraction_axt_t = np.zeros(eta_contraction_ax_t.shape)
        best_eta_contraction_axt_t[best_inds[0], best_inds[1], :] = eta_contraction_ax_t[best_inds[0], best_inds[1], :]
        eta_time_expec_2 = np.dot(best_eta_contraction_axt_t, times_arr + 1)
        inner_eq = (fg_axt[:, :, t] * (t + 1)) + np.multiply(1 - fg_axt[:, :, t],
                                                             eta_time_expec_2)  # "+1" is for the needed time shift to avoid an error introduced from setting zeros to inf. Do not change!
        inner_eq[inner_eq == 0] = np.inf
        pi[:, t] = np.argmin(inner_eq, axis=0).astype('int')
        new_eta = np.multiply((1 - fg_axt[pi[:, t], x_inds, :]), eta_contraction_ax_t[pi[:, t], x_inds, :])
        eta[:, t, :] = new_eta
        eta[:, t, t] = fg_axt[pi[:, t], x_inds, t]

        is_gt_zero = kappa[:, t] > 0
        is_zero = np.logical_not(is_gt_zero)
        eta_fail[is_gt_zero, t, t] = 0
        eta_fail[is_gt_zero, t, (t + 1):] = np.tensordot(p_ax_x[pi[is_gt_zero, t], is_gt_zero, :],
                                                         eta_fail[:, t + 1, (t + 1):], axes=([1, 0]))
        eta_fail[is_zero, t, t] = 1
        eta_fail[is_zero, t, (t + 1):] = 0

    # Compute eta_fail for the time beyond max_time to T_f that we did not optimize for because we knew it was infeasible
    known_infeasible_times = np.arange(max_feasible_time + 1, T_f)
    eta_fail[:, known_infeasible_times, known_infeasible_times] = 1

    if np.all((eta_fail[:, :, :] + eta[:, :, :]).sum(axis=2)[:]) == False:
        warnings.warn("Does not sum to one over success and failure.")
    if np.all(eta.sum(axis=2) == kappa) == False:
        warnings.warn("Kappa not equal to summed Eta")

    return kappa, pi, eta, eta_fail


def feasibility_iteration_flat_spmat_w_failure(p_ax_x_sp, fg_ax, T_f, xg, fill_in=False):
    # TODO: I verified it works with determinisitic conditions. Check Stochastic to make sure that works too.
    # p_axt_x must be [na,nx,T_f,nx']:  "p_axt_x(x'|x,a,t)" in the theory
    # f must be [nx,T_f]
    # fg_xat = np.moveaxis(fg_axt, [0], [1]).reshape(nx * na, T_f)
    FI_start_time = timeit.default_timer()
    fg_axt = fg_ax[:, :, np.newaxis]
    fg_xa = fg_ax.transpose()
    na, nx, _ = fg_axt.shape
    x_g = np.where(fg_axt > 0)[1][0]
    kappa = np.zeros([nx, T_f])
    eta = sp.sparse.csr_matrix((nx*T_f, nx*T_f))
    # eta = np.zeros([nx, T_f,
    #                 T_f])  # Typicially eta(x_f,t_f|x,t) should be [nx,T_f,nx,T_f] or [ng,T_f,nx,T_f], however assuming f encodes a single goal, we can drop the index
    eta_fail = sp.sparse.csr_matrix((nx*T_f, nx*T_f))
    # eta_fail = np.zeros([nx, T_f, T_f])
    pi = np.zeros([nx, T_f]).astype('int')

    # Caution: This assumes p_axt_x is stationary (even though it is indexed by time).  If it is non-stationary, one needs to compute time-dependent identity state-actions.
    # id_x, id_ax = get_identity_state_actions(p_axt_x[:,:,0,:])

    # Boundary Conditions
    kappa[:, -1] = np.amax(fg_axt[:, :, -1], axis=0)
    pi[:, -1] = np.argmax(fg_axt[:, :, -1], axis=0).astype('int')
    final_times_arr = np.zeros(nx*T_f)
    final_times_arr = np.arange(nx * (T_f - 1), nx * T_f)
    final_availability = fg_axt[pi[:, -1], range(nx), -1]
    fa_nz = np.where(final_availability)[0]
    fa_z = np.where(final_availability == 0)[0]
    # eta[final_times_arr[fa_nz], final_times_arr[fa_nz]] = fg_axt[pi[fa_nz, -1], fa_nz, -1]
    eta_partial = sp.sparse.csr_matrix((nx, nx * T_f))
    eta_partial[fa_nz, final_times_arr[fa_nz]] = fg_axt[pi[fa_nz, -1], fa_nz, -1]
    # eta[:, -1, -1, :] = np.repeat(fg_axt[pi[:, -1], range(nx), -1], nx).reshape([nx, nx])
    # eta_fail[final_times_arr, final_times_arr] = 1 - fg_axt[pi[:, -1], range(nx), -1]
    eta_fail_partial = sp.sparse.csr_matrix((nx, nx * T_f))
    eta_fail_partial[fa_z, final_times_arr[fa_z]] = 1 - fg_axt[pi[fa_z, -1], fa_z, -1]

    # For computing average time minimization
    future_time_arr = np.arange(T_f) + 1  # +1 is for the needed time shift to avoid an error introduced from setting zeros to inf

    max_feasible_time = np.max(np.where(fg_axt > 0)[2])  # This sets the time to be the largest time for which there is positive availability. Do not change!
    x_inds = np.arange(nx)
    # fg_xat = sp.sparse.csr_matrix(np.moveaxis(fg_axt, [0],[1]))
    final_time = min(max_feasible_time, T_f - 1)
    full_time_arr = np.arange(T_f).repeat(nx) + 1  # +1 is a needed time shift to set all zeros to np.inf when minimizing over final time.
    # full_time_arr_sparse = csr(np.arange(T_f).repeat(nx))
    eta_vstack_list = []
    eta_fail_vstack_list = []
    eta_vstack_list.append(eta_partial)
    eta_fail_vstack_list.append(eta_fail_partial)

    if np.all((eta_fail_partial[:, :] + eta_partial[:, :]).sum(axis=1)) == False:
        warnings.warn("Does not sum to one over success and failure.")
    for t in reversed(np.arange(T_f - 1)):
        starttime_loop = timeit.default_timer()
        cur_time_arr = np.arange(nx * t, nx * (t + 1))
        future_time_arr = np.arange(nx * (t + 1), nx * (t + 2))
        future_time_till_end_arr = np.arange(nx * (t + 1), nx * T_f)
        starttime = timeit.default_timer()
        eta_contraction_ax_xt = p_ax_x_sp.dot(eta_partial)

        # p_csc = sp.sparse.csc_matrix(p_ax_x_sp)
        # eta_p_csc = sp.sparse.csc_matrix(eta_partial)
        # starttime = timeit.default_timer()
        # test2 = p_csc.dot(eta_partial)
        # # test3 = p_ax_x_sp.dot(eta_partial)
        # print("csc dot: ", timeit.default_timer() - starttime)
        # eta_contraction_ax_xt_tall = sp.sparse.csr_matrix.dot(p_ax_x_sp_tall_time, eta[future_time_arr, :])
        if t % (T_f-2) == 0: print("csr dot: ", timeit.default_timer() - starttime)
        starttime = timeit.default_timer()
        future_kappa_w_actions_xa = eta_contraction_ax_xt.sum(axis=1)
        future_kappa_w_actions_xa_rs = future_kappa_w_actions_xa.reshape(nx, na)
        # if t % (T_f-2) == 0: print("1.0 block: ", timeit.default_timer() - starttime)

        starttime = timeit.default_timer()
        kappa_w_actions_axt = np.array(fg_xa + np.multiply((1 - fg_xa), future_kappa_w_actions_xa_rs))
        kappa_update = np.max(kappa_w_actions_axt, axis=1)
        kappa[:, t] = kappa_update
        # if t % (T_f-2) == 0: print("1.1 block: ", timeit.default_timer() - starttime)

        # non_zero_states = np.where(kappa_update > 0)[0]  # States that have positive feasibility
        # starttime = timeit.default_timer()
        # non_zero_states_with_actions = np.array(non_zero_states).repeat(na) + np.array(list(np.arange(na))*len(non_zero_states))
        a_star_set = np.where(kappa_w_actions_axt == kappa_update[:, np.newaxis])  # a_star_set[0] is the state inds and a_star_set[1] is the action inds
        # # if t % (T_f-2) == 0: print("1.22 block: ", timeit.default_timer() - starttime)
        # starttime = timeit.default_timer()
        # list(filter(lambda x: x[1] in non_zero_states, list(enumerate(a_star_set[0]))))
        # a_star_set_feas_filtered = np.where(kappa_w_actions_axt == kappa_update[:, np.newaxis])
        # non_zero_states_inds = [index for index, el in enumerate(a_star_set[0]) if el in non_zero_states]
        # nz_sa_inds = a_star_set[0][non_zero_states_inds] * na + a_star_set[1][non_zero_states_inds]  # The best state-action inds filtered for postitive feasibility.
        # nzi_inds = list(np.concatenate([np.where(a_star_set[0] == nzi)[0] for nzi in non_zero_states]))
        # # if t % (T_f-2) == 0: print("1.23 block: ", timeit.default_timer() - starttime)

        # starttime = timeit.default_timer()
        # best_nz_states = a_star_set[0][non_zero_states_inds]
        # best_nz_actions = a_star_set[1][non_zero_states_inds]
        # best_nz_states_unique = np.unique(best_nz_states)
        best_nz_states = np.unique(a_star_set[0])
        optimal_state_actions = a_star_set[0] * na + a_star_set[1]
        # # if t % (T_f-2) == 0: print("1.3 block: ", timeit.default_timer() - starttime)
        # idx = np.ix_(a_star_set[0], a_star_set[1])
        # best_eta_contraction_ax_xt = sp.sparse.csr_matrix(eta_contraction_ax_xt.shape)
        # best_eta_contraction_ax_xt = eta_contraction_ax_xt[a_star_set[0] * na + a_star_set[1], :]
        # starttime = timeit.default_timer()
        # eta_time_expec_2_v1 = eta_contraction_ax_xt[optimal_state_actions, :].dot(full_time_arr)
        # print("2.01 block: ", timeit.default_timer() - starttime)
        starttime = timeit.default_timer()
        eta_time_expec_2 = eta_contraction_ax_xt.dot(full_time_arr)[optimal_state_actions]
        # eta_time_expec_2[eta_time_expec_2==0] = np.inf
        # if t % (T_f-2) == 0: print("2.02 block: ", timeit.default_timer() - starttime)

        starttime = timeit.default_timer()
        final_time_xa_mat = np.ones([nx, na]) * np.inf  # Not expensive
        availability = fg_xa[a_star_set[0], a_star_set[1]]
        inner_eq = (availability * (t + 1)) + np.multiply(1 - availability, eta_time_expec_2)  # "+ 1" is for the needed time shift to avoid an error introduced from setting zeros to inf. Do not change!
        # if t % (T_f-2) == 0: print("2.1 block: ", timeit.default_timer() - starttime)
        starttime = timeit.default_timer()
        inner_eq[inner_eq == 0] = np.inf
        final_time_xa_mat[a_star_set[0], a_star_set[1]] = inner_eq
        # if t % (T_f-2) == 0: print("2.2 block: ", timeit.default_timer() - starttime)
        starttime = timeit.default_timer()
        # final_time_xa_mat[final_time_xa_mat == 0] = np.inf
        pi[best_nz_states, t] = np.argmin(final_time_xa_mat[best_nz_states, :], axis=1).astype('int')
        # new_eta = sp.sparse.lil_matrix(eta)
        # if t % (T_f-2) == 0: print("2.3 block: ", timeit.default_timer() - starttime)

        # starttime = timeit.default_timer()
        # best_eta_contraction_ax_xt = eta_contraction_ax_xt[best_nz_states * na + best_nz_actions, :]
        # eta_time_expec_2 = sp.sparse.csr_matrix.dot(best_eta_contraction_ax_xt, full_time_arr + 1)
        # final_time_xa_mat = np.ones([nx, na]) * np.inf
        # inner_eq = (fg_xa[best_nz_states, best_nz_actions] * (t + 1)) + np.multiply(
        #     1 - fg_xa[best_nz_states, best_nz_actions],
        #     eta_time_expec_2)  # "+ 1" is for the needed time shift to avoid an error introduced from setting zeros to inf. Do not change!
        # inner_eq[inner_eq == 0] = np.inf
        # final_time_xa_mat[best_nz_states, best_nz_actions] = inner_eq
        # # final_time_xa_mat[final_time_xa_mat == 0] = np.inf
        # pi[best_nz_states_unique, t] = np.argmin(final_time_xa_mat[best_nz_states_unique, :], axis=1).astype('int')
        # # new_eta = sp.sparse.lil_matrix(eta)
        # # if t % (T_f-2) == 0: print("second block: ", timeit.default_timer() - starttime)

        # starttime = timeit.default_timer()
        # new_eta = eta.copy()
        # # if t % (T_f-2) == 0: print("copy: ", timeit.default_timer() - starttime)

        # fg_xa_sparse_mat = sp.sparse.csr_matrix()
        # fg_xa_sparse_mat = np.vstack([fg_xa] * T_f)

        starttime = timeit.default_timer()
        # pi_inds_all_state_times = pi.transpose().flatten()
        # new_eta[x_inds * na + pi[:, t], :] = eta_contraction_ax_xt[x_inds * na + pi[:, t], :] * np.array(1 - fg_xa[x_inds, pi[:, t]])[:, np.newaxis]
        # new_eta = eta_contraction_ax_xt[x_inds * na + pi[:, t], :] * np.array(1 - fg_xa[x_inds, pi[:, t]])[:, np.newaxis]
        unavail_dense = 1 - fg_xa[x_inds, pi[:, t]]
        row = x_inds
        col = np.zeros(len(unavail_dense), dtype=int)
        # if t % (T_f-2) == 0: print("prep 1: ", timeit.default_timer() - starttime)
        starttime = timeit.default_timer()
        unavail_vec = sp.sparse.csc_matrix((unavail_dense, (row, col)))
        # if t % (T_f-2) == 0: print("prep 2: ", timeit.default_timer() - starttime)
        starttime = timeit.default_timer()
        new_eta_partial = unavail_vec.multiply(eta_contraction_ax_xt[x_inds * na + pi[:, t], :])
        # if t % (T_f-2) == 0: print("prep 3: ", timeit.default_timer() - starttime)
        starttime = timeit.default_timer()
        non_zero_data_inds = np.where(fg_xa[x_inds, pi[:, t]] > 0)[0]
        # if t % (T_f-2) == 0: print("prep 4: ", timeit.default_timer() - starttime)
        # fg_xa[non_zero_data_inds, pi[non_zero_data_inds, t]]
        starttime = timeit.default_timer()
        if len(non_zero_data_inds) > 0:
            new_eta_partial[non_zero_data_inds, cur_time_arr[non_zero_data_inds]] = fg_xa[non_zero_data_inds, pi[non_zero_data_inds, t]]
        # if t % (T_f-2) == 0: print("assignment 1: ", timeit.default_timer() - starttime)

        # small_mat = sp.sparse.csr_matrix( ( fg_xa[non_zero_data_inds, pi[non_zero_data_inds, t]], (non_zero_data_inds, non_zero_data_inds)), shape=(nx, nx))
        # new_eta_partial = sp.sparse.hstack([small_mat, new_eta_partial])
        eta_vstack_list.append(new_eta_partial)
        eta_partial = new_eta_partial.copy()

        # eta[nx * t + x_inds, :] = new_eta
        starttime = timeit.default_timer()
        # if t % (T_f-2) == 0: print("current time set: ", timeit.default_timer() - starttime)

        starttime = timeit.default_timer()
        gt_zero_inds = np.where(kappa[:, t] > 0)[0]
        # gt_zero_inds_w_time = nx * t + gt_zero_inds
        # gt_zero_inds_w_actions = gt_zero_inds * na + pi[gt_zero_inds, t]
        states_w_action_inds = np.arange(nx) * na + pi[:, t]
        is_zero_inds = np.where(kappa[:, t] == 0)[0]
        is_zero_inds_w_time = nx * t + is_zero_inds
        # is_zero_inds_w_actions = is_zero_inds * na + pi[is_zero_inds, t]
        if t % (T_f-2) == 0: print("4.0 block: ", timeit.default_timer() - starttime)

        starttime = timeit.default_timer()
        data = np.ones(len(gt_zero_inds), dtype=int)
        row = gt_zero_inds
        col = np.zeros(len(gt_zero_inds), dtype=int)
        gt_zero_spvec = sp.sparse.csr_matrix((data, (row, col)), (nx, 1)).transpose()
        # if t % (T_f-2) == 0: print("4.11 block: ", timeit.default_timer() - starttime)

        # starttime = timeit.default_timer()
        # # future_time_until_end_diag = csr(onehot(future_time_till_end_arr, nx * T_f))
        # # future_time_until_end_vec = sp.sparse.csr_matrix(onehot(future_time_till_end_arr, nx * T_f))  # Doesn't work, takes a long time
        # # if t % (T_f-2) == 0: print("4.12 block: ", timeit.default_timer() - starttime)
        # starttime = timeit.default_timer()
        # # gt_p_ax_x_sp = gt_zero_diag.dot(p_ax_x_sp)
        # # if t % (T_f-2) == 0: print("4.2 block: ", timeit.default_timer() - starttime)

        starttime = timeit.default_timer()
        # col = future_time_till_end_arr
        # data = np.ones(len(future_time_till_end_arr))
        # row = np.zeros(len(future_time_till_end_arr), dtype=int)
        # future_time_till_end_vec_csr = csr((data, (row, col)))
        # eta_fail_future_until_end = eta_fail_partial.multiply(future_time_till_end_vec_csr)  # This is a problem because
        # if t % (T_f-2) == 0: print("4.2 block: ", timeit.default_timer() - starttime)

        starttime = timeit.default_timer()
        # eta_fail_partial_new = sp.sparse.csr_matrix.dot(gt_zero_diag.dot(p_ax_x_sp[states_w_action_inds, :]), eta_fail_future_until_end)
        eta_fail_partial_new = (gt_zero_spvec.multiply(p_ax_x_sp[states_w_action_inds, :])).dot(eta_fail_partial)  # We should expect eta_fail_partial_new to have no positive entries under deterministic dynamics!
        # if t % (T_f-2) == 0: print("4.4 block: ", timeit.default_timer() - starttime)
        # eta_fail_partial_new[gt_zero_inds, cur_time_arr[gt_zero_inds]] = 0  # All entries should be zero here, so we don't need this line.

        starttime = timeit.default_timer()
        data = np.ones(len(is_zero_inds))
        row = is_zero_inds
        col = is_zero_inds_w_time
        eta_fail_time_t = csr((data, (row, col)), eta_fail_partial_new.shape)
        # if t % (T_f-2) == 0: print("create csr: ", timeit.default_timer() - starttime)
        starttime = timeit.default_timer()
        eta_fail_partial_new_with_cur_time = eta_fail_partial_new + eta_fail_time_t
        # if t % (T_f-2) == 0: print("add csr mats: ", timeit.default_timer() - starttime)
        starttime = timeit.default_timer()
        # eta_fail_partial_new[is_zero_inds, is_zero_inds_w_time] = 1
        # eta_fail_partial_new[is_zero_inds, :][:, future_time_till_end_arr] = 0  # This line is commented out because the entries should already be zero, thus it is unnecessary.
        eta_fail_vstack_list.append(eta_fail_partial_new_with_cur_time)
        eta_fail_partial = eta_fail_partial_new_with_cur_time.copy()
        # eta = new_eta.tocsr()
        # if t % (T_f-2) == 0: print("fifth block: ", timeit.default_timer() - starttime)
        if t % (T_f-2) == 0: print("loop total: ", timeit.default_timer() - starttime_loop)
        # profiler.disable()
        # profiler.print_stats(sort='cumulative')
        if np.all((eta_fail_partial + eta_partial).sum(axis=1)) == False:
            warnings.warn("Does not sum to one over success and failure.")
        # if np.all(eta_partial.sum(axis=2) == kappa) == False:
        #     warnings.warn("Kappa not equal to summed Eta")
        if t%100 == 0:
            print('Time step: ',t,'/',T_f)

    # Compute eta_fail for the time beyond max_time to T_f that we did not optimize for because we knew it was infeasible
    # known_infeasible_times = np.arange(max_feasible_time + 1, T_f)
    # eta_fail[:, known_infeasible_times, known_infeasible_times] = 1


    # if np.all(eta.sum(axis=2) == kappa) == False:
    #     warnings.warn("Kappa not equal to summed Eta")
    eta_vstack_list.reverse()
    eta_fail_vstack_list.reverse()
    eta = sp.sparse.vstack(eta_vstack_list)
    eta_fail = sp.sparse.vstack(eta_fail_vstack_list)
    if np.all((eta_fail + eta).sum(axis=1)) == False:
        warnings.warn("Does not sum to one over success and failure.")
    # if t % (T_f-2) == 0: print("FI total: ", timeit.default_timer() - FI_start_time)

    return kappa, pi, eta, eta_fail



def feasibility_iteration_flat_sptensor_w_failure(p_axt_x, fg_axt_dense, xg, p_ax_x=None, fill_in=False):
    # This is hopeless, the sparse_tensor package "sparse" sucks and is WAY slower than np.tensordot on dense matricies
    # TODO: I verified it works with deterministic conditions. Check Stochastic to make sure that works too.
    # p_axt_x must be sparse and  [na,nx,T_f,nx']:  "p_axt_x(x'|x,a,t)" in the theory
    # fg_axt must be sparse
    # p_axt_x = sparse.COO(p_axt_x_dense)
    na, nx, _, _ = p_axt_x.shape
    T_f = p_axt_x.shape[2]
    x_g = np.where(fg_axt_dense > 0)[1][0]
    kappa = np.zeros([nx, T_f])
    eta = np.zeros([nx, T_f, T_f])  # Typicially eta(x_f,t_f|x,t) should be [nx,T_f,nx,T_f] or [ng,T_f,nx,T_f], however assuming f encodes a single goal, we can drop the index
    eta_fail = np.zeros([nx, T_f, T_f])
    pi = np.zeros([nx, T_f]).astype('int')

    fg_axt = sparse.COO(fg_axt_dense)
    fg_axt_DOK = sparse.DOK(fg_axt_dense)

    # Caution: This assumes p_axt_x is stationary (even though it is indexed by time).  If it is non-stationary, one needs to compute time-dependent identity state-actions.
    # id_x, id_ax = get_identity_state_actions(p_axt_x[:, :, 0, :])
    eta = sparse.DOK(shape=(nx, T_f, T_f))
    eta_fail = sparse.DOK(shape=(nx, T_f, T_f))

    # Boundary Conditions
    kappa[:, -1] = np.amax(fg_axt_dense[:, :, -1], axis=0)
    pi[:, -1] = np.argmax(fg_axt_dense[:, :, -1], axis=0).astype('int')
    eta[:, -1, -1] = fg_axt_dense[pi[:, -1], range(nx), -1]
    eta_fail[:, -1, -1] = 1 - fg_axt_dense[pi[:, -1], range(nx), -1]


    # For computing average time minimization
    times_arr = np.arange(
        T_f) + 1  # +1 is for the needed time shift to avoid an error introduced from setting zeros to inf

    # Fill-in
    if fill_in:
        id_x, id_ax = get_identity_state_actions(p_ax_x)
        max_kappa_states = np.array([np.argmax(kappa[:, -1])])
        if np.any(kappa[max_kappa_states, -1] > 0):
            min_time_states = np.array([max_kappa_states[np.argmin(np.dot(eta[max_kappa_states, -1, :], times_arr))]])
            # Fill-in states are the states that are jointly cumulative maximizing, time-minimizing and stable
            fill_in_states = np.intersect1d(np.intersect1d(np.arange(nx)[id_x], max_kappa_states), min_time_states)
            fill_in_actions = np.array([np.argmax(id_ax[:, fill_in_states], axis=0)])
            for x, a in np.array([fill_in_states, fill_in_actions]).transpose():
                for t in np.flip(np.arange(T_f - 1)):
                    future_kappa = kappa[x, t + 1]
                    kappa[x, t] = fg_axt[a, x, t] + np.multiply((1 - fg_axt[a, x, t]), future_kappa)
                    pi[x, t] = a
                    eta[x, t, :] = (1 - fg_axt[a, x, t]) * np.tensordot(p_axt_x[a, x, t, :], eta[:, t + 1, :],
                                                                        axes=([0, 0]))

    max_feasible_time = np.max(np.where(fg_axt > 0)[
                                   2])  # This sets the time to be the largest time for which there is positive availability. Do not change!
    x_inds = np.arange(nx)

    dense_paxtx = p_axt_x.todense()
    dense_eta = eta.todense()

    final_time = min(max_feasible_time, T_f - 2)
    p_axt_x_GCXS = sparse.GCXS(p_axt_x, compressed_axes=[3])

    for t in reversed(np.arange(final_time + 1)):
        starttime_loop = timeit.default_timer()
        print(t)
        starttime = timeit.default_timer()
        eta_contraction_ax_t = sparse.einsum("ijk,kl->ijl", p_axt_x_GCXS[:, :, t, :], eta[:, t + 1, :])
        # if t % (T_f-2) == 0: print("einsum: ", timeit.default_timer()-starttime)
        starttime = timeit.default_timer()
        eta_contraction_ax_t = sparse.tensordot(p_axt_x_GCXS[:, :, t, :], sparse.COO(eta[:, t + 1, :]), axes=[2,0])
        # if t % (T_f-2) == 0: print("tensordot_w_COO_convert: ", timeit.default_timer()-starttime)
        starttime = timeit.default_timer()
        eta_contraction_ax_t_dense = np.tensordot(dense_paxtx[:, :, t, :], dense_eta[:, t + 1, :], axes=([2, 0]))
        # if t % (T_f-2) == 0: print("dense_tensordot: ", timeit.default_timer() - starttime)
        # eta_contraction_ax_t = np.tensordot(p_axt_x[:, :, t, :], eta[:, t + 1, :], axes=([2, 0]))  # condition on t
        starttime = timeit.default_timer()
        future_kappa_w_actions_ax = eta_contraction_ax_t.sum(axis=2)
        # if t % (T_f-2) == 0: print("sum 1: ", timeit.default_timer() - starttime)
        # future_kappa_w_actions_ax = np.sum(eta_contraction_ax_t, axis=2)

        starttime = timeit.default_timer()
        kappa_w_actions_axt = fg_axt_dense[:, :, t] + (1 - fg_axt_dense[:, :, t])*future_kappa_w_actions_ax
        # if t % (T_f-2) == 0: print("add and mult: ", timeit.default_timer() - starttime)
        # kappa_w_actions_axt = fg_axt[:, :, t] + np.multiply((1 - fg_axt[:, :, t]), future_kappa_w_actions_ax)

        starttime = timeit.default_timer()
        kappa_update = kappa_w_actions_axt.max(axis=0)
        # if t % (T_f-2) == 0: print("to dense: ", timeit.default_timer() - starttime)
        # kappa_update = np.max(kappa_w_actions_axt, axis=0)
        starttime = timeit.default_timer()
        kappa[:, t] = kappa_update
        best_inds = np.where(kappa_w_actions_axt == kappa_update)
        # if t % (T_f-2) == 0: #print("best inds: ", timeit.default_timer() - starttime)
        # non_zero_inds = np.where(kappa_update > 0)[0]
        starttime = timeit.default_timer()
        eta_contraction_ax_t_dense = eta_contraction_ax_t.todense()
        # if t % (T_f-2) == 0: print("to dense 2: ", timeit.default_timer() - starttime)
        # idx = np.ix_(best_inds[0], best_inds[1])
        starttime = timeit.default_timer()
        best_eta_contraction_axt_t = np.zeros(eta_contraction_ax_t.shape)
        best_eta_contraction_axt_t[best_inds[0], best_inds[1], :] = eta_contraction_ax_t_dense[best_inds[0], best_inds[1], :]
        eta_time_expec_2 = np.dot(best_eta_contraction_axt_t, times_arr)
        # if t % (T_f-2) == 0: print("weird bs 1: ", timeit.default_timer() - starttime)
        starttime = timeit.default_timer()
        inner_eq = (fg_axt[:, :, t] * (t + 1)) + np.multiply(1 - fg_axt[:, :, t],
                                                             eta_time_expec_2)  # "+1" is for the needed time shift to avoid an error introduced from setting zeros to inf. Do not change!
        # if t % (T_f-2) == 0: print("rest 1: ", timeit.default_timer() - starttime)
        starttime = timeit.default_timer()
        inner_eq[inner_eq == 0] = np.inf
        # if t % (T_f-2) == 0: print("rest 2: ", timeit.default_timer() - starttime)
        starttime = timeit.default_timer()
        pi[:, t] = np.argmin(inner_eq, axis=0).astype('int')
        # if t % (T_f-2) == 0: print("rest 3: ", timeit.default_timer() - starttime)
        starttime = timeit.default_timer()
        new_eta = np.multiply((1 - fg_axt[pi[:, t], x_inds, :]), eta_contraction_ax_t_dense[pi[:, t], x_inds, :])
        # if t % (T_f-2) == 0: print("rest 4: ", timeit.default_timer() - starttime)
        starttime = timeit.default_timer()
        eta[:, t, :] = new_eta  # THIS IS THE BOTTLE-NECK! Need to figure out how to do this efficiently.  Ma
        # if t % (T_f-2) == 0: print("rest 5: ", timeit.default_timer() - starttime)
        starttime = timeit.default_timer()
        new_eta[:, t] = fg_axt_dense[pi[:, t], x_inds, t]
        # eta[:, t, t] = fg_axt_dense[pi[:, t], x_inds, t]
        # if t % (T_f-2) == 0: print("rest 6: ", timeit.default_timer() - starttime)
        # if t % (T_f-2) == 0: print("Loop: ", timeit.default_timer() - starttime_loop)
        # is_gt_zero = kappa[:, t] > 0
        # is_zero = np.logical_not(is_gt_zero)
        # eta_fail[is_gt_zero*1, t, t] = 0
        # eta_fail[is_gt_zero, t, (t + 1):] = np.tensordot(p_axt_x[pi[is_gt_zero, t], is_gt_zero, t, :],
        #                                                  eta_fail[:, t + 1, (t + 1):], axes=([1, 0]))
        # eta_fail[is_zero, t, t] = 1
        # eta_fail[is_zero, t, (t + 1):] = 0

    # Compute eta_fail for the time beyond max_time to T_f that we did not optimze for because we knew it was infeasible
    # known_infeasible_times = np.arange(max_feasible_time + 1, T_f)
    # eta_fail[:, known_infeasible_times, known_infeasible_times] = 1

    # if np.all((eta_fail[:, :, :] + eta[:, :, :]).sum(axis=2)[:]) == False:
    #     warnings.warn("Does not sum to one over success and failure.")
    # if np.all(eta.sum(axis=2) == kappa) == False:
    #     warnings.warn("Kappa not equal to summed Eta")

    return kappa, pi, eta, eta_fail


def feasibility_iteration_flat_dense_stationary_w_failure(fg_axt, p_ax_x, T_f, fill_in=False):
    # TODO: I verified it works with determinisitic conditions. Check Stochastic to make sure that works too.
    # p_axt_x must be [na,nx,T_f,nx']:  "p_axt_x(x'|x,a,t)" in the theory
    # f must be [nx,T_f]
    na, nx, _ = p_ax_x.shape
    x_g = np.where(fg_axt > 0)[1][0]
    kappa = np.zeros([nx, T_f])
    eta = np.zeros([nx, T_f,
                    T_f])  # Typicially eta(x_f,t_f|x,t) should be [nx,T_f,nx,T_f] or [ng,T_f,nx,T_f], however assuming f encodes a single goal, we can drop the index
    eta_fail = np.zeros([nx, T_f, T_f])
    pi = np.zeros([nx, T_f]).astype('int')

    # Caution: This assumes p_axt_x is stationary (even though it is indexed by time).  If it is non-stationary, one needs to compute time-dependent identity state-actions.
    # id_x, id_ax = get_identity_state_actions(p_ax_x[:, :, 0, :])

    # Boundary Conditions
    kappa[:, -1] = np.amax(fg_axt[:, :, -1], axis=0)
    pi[:, -1] = np.argmax(fg_axt[:, :, -1], axis=0).astype('int')
    eta[:, -1, -1] = fg_axt[pi[:, -1], range(nx), -1]
    eta_fail[:, -1, -1] = 1 - fg_axt[pi[:, -1], range(nx), -1]

    # For computing average time minimization
    times_arr = np.arange(T_f) + 1  # +1 is for the needed time shift to avoid an error introduced from setting zeros to inf

    # Fill-in
    if fill_in:
        id_x, id_ax = get_identity_state_actions(p_ax_x)
        max_kappa_states = np.array([np.argmax(kappa[:, -1])])
        if np.any(kappa[max_kappa_states, -1] > 0):
            min_time_states = np.array([max_kappa_states[np.argmin(np.dot(eta[max_kappa_states, -1, :], times_arr))]])
            # Fill-in states are the states that are jointly cumulative maximizing, time-minimizing and stable
            fill_in_states = np.intersect1d(np.intersect1d(np.arange(nx)[id_x], max_kappa_states), min_time_states)
            fill_in_actions = np.array([np.argmax(id_ax[:, fill_in_states], axis=0)])
            for x, a in np.array([fill_in_states, fill_in_actions]).transpose():
                for t in np.flip(np.arange(T_f - 1)):
                    future_kappa = kappa[x, t + 1]
                    kappa[x, t] = fg_axt[a, x, t] + np.multiply((1 - fg_axt[a, x, t]), future_kappa)
                    pi[x, t] = a
                    eta[x, t, :] = (1 - fg_axt[a, x, t]) * np.tensordot(p_ax_x[a, x, :], eta[:, t + 1, :],
                                                                        axes=([0, 0]))

    max_feasible_time = np.max(np.where(fg_axt > 0)[2])  # This sets the time to be the largest time for which there is positive availability. Do not change!
    x_inds = np.arange(nx)

    final_time = min(max_feasible_time, T_f - 2)

    # Main Loop of Feasibility Iteration.
    for t in reversed(np.arange(final_time + 1)):
        eta_contraction_ax_t = np.tensordot(p_ax_x[:, :, :], eta[:, t + 1, :], axes=([2, 0]))  # condition on t
        future_kappa_w_actions_ax = np.sum(eta_contraction_ax_t, axis=2)
        kappa_w_actions_axt = fg_axt[:, :, t] + np.multiply((1 - fg_axt[:, :, t]), future_kappa_w_actions_ax)
        kappa_update = np.max(kappa_w_actions_axt, axis=0)
        kappa[:, t] = kappa_update
        best_inds = np.where(kappa_w_actions_axt == kappa_update)
        non_zero_inds = np.where(kappa_update > 0)[0]

        # idx = np.ix_(best_inds[0], best_inds[1])
        best_eta_contraction_axt_t = np.zeros(eta_contraction_ax_t.shape)
        best_eta_contraction_axt_t[best_inds[0], best_inds[1], :] = eta_contraction_ax_t[best_inds[0], best_inds[1], :]
        eta_time_expec_2 = np.dot(best_eta_contraction_axt_t, times_arr)
        inner_eq = (fg_axt[:, :, t] * (t + 1)) + np.multiply(1 - fg_axt[:, :, t],
                                                             eta_time_expec_2)  # "+1" is for the needed time shift to avoid an error introduced from setting zeros to inf. Do not change!
        inner_eq[inner_eq == 0] = np.inf
        pi[:, t] = np.argmin(inner_eq, axis=0).astype('int')
        new_eta = np.multiply((1 - fg_axt[pi[:, t], x_inds, :]), eta_contraction_ax_t[pi[:, t], x_inds, :])
        eta[:, t, :] = new_eta
        eta[:, t, t] = fg_axt[pi[:, t], x_inds, t]

        is_gt_zero = kappa[:, t] > 0
        is_zero = np.logical_not(is_gt_zero)
        eta_fail[is_gt_zero, t, t] = 0
        eta_fail[is_gt_zero, t, (t + 1):] = np.tensordot(p_ax_x[pi[is_gt_zero, t], is_gt_zero, :],
                                                         eta_fail[:, t + 1, (t + 1):], axes=([1, 0]))
        eta_fail[is_zero, t, t] = 1
        eta_fail[is_zero, t, (t + 1):] = 0

    # Compute eta_fail for the time beyond max_time to T_f that we did not optimze for because we knew it was infeasible
    known_infeasible_times = np.arange(max_feasible_time + 1, T_f)
    eta_fail[:, known_infeasible_times, known_infeasible_times] = 1

    if np.all((eta_fail[:, :, :] + eta[:, :, :]).sum(axis=2)[:]) == False:
        warnings.warn("Does not sum to one over success and failure.")
    if np.all(eta.sum(axis=2) == kappa) == False:
        warnings.warn("Kappa not equal to summed Eta")

    return kappa, pi, eta, eta_fail


def feasibility_iteration_flat_sparse_w_failure(p_ax_x_dict, fg_axt, p_ax_x=None, fill_in=False):
    # TODO: I verified it works with determinisitic conditions. Check Stochastic to make sure that works too.
    # p_axt_x must be [na,nx,T_f,nx']:  "p_axt_x(x'|x,a,t)" in the theory
    # f must be [nx,T_f]
    # p_axt_x must be sparse
    na, nx, T_f = fg_axt.shape
    x_g = np.where(fg_axt > 0)[1][0]
    kappa = np.zeros([nx, T_f])
    eta = np.zeros([nx * T_f, T_f])  # Typicially eta(x_f,t_f|x,t) should be [nx,T_f,nx,T_f] or [ng,T_f,nx,T_f], however assuming f encodes a single goal, we can drop the index
    eta_fail = np.zeros([nx * T_f, T_f])
    pi = np.zeros([nx, T_f]).astype('int')

    # Caution: This assumes p_axt_x is stationary (even though it is indexed by time).  If it is non-stationary, one needs to compute time-dependent identity state-actions.

    # Boundary Conditions
    kappa[:, -1] = np.amax(fg_axt[:, :, -1], axis=0)
    pi[:, -1] = np.argmax(fg_axt[:, :, -1], axis=0).astype('int')
    eta[:, -1, -1] = fg_axt[pi[:, -1], range(nx), -1]
    eta_fail[:, -1, -1] = 1 - fg_axt[pi[:, -1], range(nx), -1]

    # For computing average time minimization
    times_arr = np.arange(
        T_f) + 1  # +1 is for the needed time shift to avoid an error introduced from setting zeros to inf

    max_feasible_time = np.max(np.where(fg_axt > 0)[
                                   2])  # This sets the time to be the largest time for which there is positive availability. Do not change!
    x_inds = np.arange(nx)

    final_time = min(max_feasible_time, T_f - 2)
    for t in reversed(np.arange(final_time + 1)):
        eta_contraction_ax_t = np.tensordot(p_axt_x[:, :, t, :], eta[:, t + 1, :], axes=([2, 0]))  # condition on t
        test = [sp.sparse.csr_matrix.dot(p_ax_x_dict[a], eta[:, t + 1, :]) for a in range(na)]
        eta_contraction_ax_t_2 = np.tensordot(p_axt_x[:, :, t, :], eta[:, t + 1, :], axes=([2, 0]))  # condition on t
        future_kappa_w_actions_ax = np.sum(eta_contraction_ax_t, axis=2)
        kappa_w_actions_axt = fg_axt[:, :, t] + np.multiply((1 - fg_axt[:, :, t]), future_kappa_w_actions_ax)
        kappa_update = np.max(kappa_w_actions_axt, axis=0)
        kappa[:, t] = kappa_update
        best_inds = np.where(kappa_w_actions_axt == kappa_update)
        non_zero_inds = np.where(kappa_update > 0)[0]

        # idx = np.ix_(best_inds[0], best_inds[1])
        best_eta_contraction_axt_t = np.zeros(eta_contraction_ax_t.shape)
        best_eta_contraction_axt_t[best_inds[0], best_inds[1], :] = eta_contraction_ax_t[best_inds[0], best_inds[1], :]
        eta_time_expec_2 = np.dot(best_eta_contraction_axt_t, times_arr)
        inner_eq = (fg_axt[:, :, t] * (t + 1)) + np.multiply(1 - fg_axt[:, :, t],
                                                             eta_time_expec_2)  # "+1" is for the needed time shift to avoid an error introduced from setting zeros to inf. Do not change!
        inner_eq[inner_eq == 0] = np.inf
        pi[:, t] = np.argmin(inner_eq, axis=0).astype('int')
        new_eta = np.multiply((1 - fg_axt[pi[:, t], x_inds, :]), eta_contraction_ax_t[pi[:, t], x_inds, :])
        eta[:, t, :] = new_eta
        eta[:, t, t] = fg_axt[pi[:, t], x_inds, t]

        is_gt_zero = kappa[:, t] > 0
        is_zero = np.logical_not(is_gt_zero)
        eta_fail[is_gt_zero, t, t] = 0
        eta_fail[is_gt_zero, t, (t + 1):] = np.tensordot(p_axt_x[pi[is_gt_zero, t], is_gt_zero, t, :],
                                                         eta_fail[:, t + 1, (t + 1):], axes=([1, 0]))
        eta_fail[is_zero, t, t] = 1
        eta_fail[is_zero, t, (t + 1):] = 0

    # Compute eta_fail for the time beyond max_time to T_f that we did not optimze for because we knew it was infeasible
    known_infeasible_times = np.arange(max_feasible_time + 1, T_f)
    eta_fail[:, known_infeasible_times, known_infeasible_times] = 1

    if np.all((eta_fail[:, :, :] + eta[:, :, :]).sum(axis=2)[:]) == False:
        warnings.warn("Does not sum to one over success and failure.")
    if np.all(eta.sum(axis=2) == kappa) == False:
        warnings.warn("Kappa not equal to summed Eta")

    return kappa, pi, eta, eta_fail


def feasibility_iteration_flat_dense_forloop(p_axt_x, fg_axt, p_ax_x = None, fill_in = False):
    # Older, slower version with for loop over time minimization.  Consider deleting.

    na, nx, T_f, _ = p_axt_x.shape
    x_g = np.where(fg_axt > 0)[1][0]
    a_g = np.where(fg_axt[:, x_g, :] > 0)[1][0]
    kappa = np.zeros([nx, T_f])
    eta = np.zeros([nx, T_f, T_f]) # Typicially eta(x_f,t_f|x,t) should be [nx,T_f,nx,T_f] or [ng,T_f,nx,T_f], however assuming f encodes a single goal, we can drop the index
    pi = np.zeros([nx, T_f]).astype('int')

    # Caution: This assumes p_axt_x is stationary (even though it is indexed by time).  If it is non-stationary, one needs to compute time-dependent identity state-actions.
    id_x, id_ax = get_identity_state_actions(p_axt_x[:,:,0,:])

    # Boundary Conditions
    kappa[:, -1] = np.amax(fg_axt[:, :, -1], axis=0)
    pi[:, -1] = np.argmax(fg_axt[:, :, -1], axis=0).astype('int')
    eta[:, -1, -1] = fg_axt[pi[:, -1], range(nx), -1]
    # for t_f in range(T_f):
    #     eta[:,t_f,t_f] = fg_axt[pi[:, t_f], range(nx), t_f]

    times_arr = np.arange(T_f)

    # Fill-in
    if fill_in:
        id_x, id_ax = get_identity_state_actions(p_ax_x)
        max_kappa_states = np.array([np.argmax(kappa[:, -1])])
        if np.any(kappa[max_kappa_states,-1]>0):
            min_time_states = np.array([max_kappa_states[np.argmin(np.dot(eta[max_kappa_states,-1,:], times_arr))]])
            # Fill-in states are the states that are jointly cumulative maximizing, time-minimizing and stable
            fill_in_states = np.intersect1d(np.intersect1d(np.arange(nx)[id_x], max_kappa_states), min_time_states)
            fill_in_actions = np.array([np.argmax(id_ax[:,fill_in_states], axis=0)])
            for x, a in np.array([fill_in_states, fill_in_actions]).transpose():
                for t in np.flip(np.arange(T_f - 1)):
                    future_kappa = kappa[x, t+1]
                    kappa[x, t] = fg_axt[a, x, t] + np.multiply((1 - fg_axt[a, x, t]), future_kappa)
                    pi[x, t] = a
                    eta[x, t, :] = (1 - fg_axt[a, x, t]) * np.tensordot(p_axt_x[a, x, t, :], eta[:, t + 1, :], axes=([0, 0]))

    max_time = np.max(np.where(fg_axt>0)[2]) # This sets the time to be the largest time for which there is positive availability.

    for t in reversed(np.arange(max_time+1)):
        eta_contraction_axt_t = np.tensordot(p_axt_x[:, :, t, :], eta[:, t + 1, :], axes=([2, 0]))
        future_kappa_w_actions_axt = np.sum(eta_contraction_axt_t, axis=2)
        kappa_w_actions_axt = fg_axt[:, :, t] + np.multiply((1 - fg_axt[:, :, t]), future_kappa_w_actions_axt)
        kappa_update = np.max(kappa_w_actions_axt, axis=0)
        kappa[:,t]=kappa_update
        best_inds = np.where(kappa_w_actions_axt == kappa_update)
        non_zero_inds = np.where(kappa_update>0)[0]

        # We have to loop over individual states because python does not support jagged matrices to optimize only over the A^* actions. Perhaps there is another way by pointwise multiplying a mask of ones for the A* set.
        for x in non_zero_inds:
            a_star_set = best_inds[0][np.where(best_inds[1] == x)[0]].astype('int')
            eta_time_expec = np.dot(eta_contraction_axt_t[a_star_set, x, :], times_arr)
            pi[x, t] = a_star_set[np.argmin((fg_axt[a_star_set, x, t] * t) + np.multiply(1 - fg_axt[a_star_set, x, t], eta_time_expec)).astype('int')]
            new_eta = np.multiply((1 - fg_axt[pi[x, t], x, t]), eta_contraction_axt_t[pi[x, t], x, :])
            eta[x, t, :] = new_eta

            # Enforce Boundary Condition
            eta[x, t, t] = fg_axt[pi[x, t], x, t]
            # eta[x, t, t] = fg_axt[a_g, x, t]

        # if np.all(eta.sum(axis=2)==kappa) == False:
        #     warnings.warn("Kappa not equal to summed Eta")

    if np.all(eta.sum(axis=2)==kappa) == False:
        warnings.warn("Kappa not equal to summed Eta")

    # Kappa and eta should be equal when final states and times are summed over in eta
    # np.all(eta[:,:,:].sum(axis=2)==kappa[:,:]) should return True

    return kappa, pi, eta


def feasibility_iteration_flat_dense_alt_version(p_axt_x, fg_axt, p_ax_x = None, fill_in = False):
    # With this version, the idea is to have a time slot in eta for the infinity time. It runs about as fast as the regular verion, but could be better to move to sparse.


    # p_axt_x must be [na,nx,T_f,nx']:  "p_axt_x(x'|x,a,t)" in the theory
    # f must be [nx,T_f]
    na, nx, T_f, _ = p_axt_x.shape
    x_g = np.where(fg_axt > 0)[1][0]
    a_g = np.where(fg_axt[:, x_g, :] > 0)[1][0]
    kappa = np.zeros([nx, T_f])
    eta = np.zeros([nx, T_f, T_f]) # Typicially eta(x_f,t_f|x,t) should be [nx,T_f,nx,T_f] or [ng,T_f,nx,T_f], however assuming f encodes a single goal, we can drop the index
    pi = np.zeros([nx, T_f]).astype('int')

    # Caution: This assumes p_axt_x is stationary (even though it is indexed by time).  If it is non-stationary, one needs to compute time-dependent identity state-actions.
    id_x, id_ax = get_identity_state_actions(p_axt_x[:,:,0,:])

    # Boundary Conditions
    # kappa[:,-1] = np.amax(fg_axt[:, :, -1], axis=0)
    # pi[:,-1] = np.argmax(fg_axt[:, :, -1], axis=0).astype('int')
    # eta[:,-1,-1] = fg_axt[pi[:, -1], range(nx), -1]

    # For computing average time minimization
    # times_arr = np.arange(T_f) + 1  # +1 is for the needed time shift to avoid an error introduced from setting zeros to inf

    # For alternative (potentially sparse) implementation: This should run close to 5x faster when debugged, I think. And it should work for sparse versions.
    # New version time difference is: 0.011956718000000421
    # Forloop version time difference is: 0.048442724000000936
    #
    big_num = T_f * 10  # Arbitrary large number greater than T_f
    kappa = np.zeros([nx, T_f + 1])
    pi = np.zeros([nx, T_f + 1]).astype('int')
    eta = np.zeros([nx, T_f + 1, T_f + 1])  # Where the +1 is an extra "inf" slot for for the time-min optimization
    eta[:, :, -1] = 1  # Set last entry in axis 2 to 1, meaning the agent take inf time to complete.
    eta[:, -2, -2] = fg_axt[pi[:, -1], range(nx), -1]
    kappa[:, -1] = 0
    pi[:, -1] = 0
    times_arr = np.arange(T_f + 1).astype('float')  # + 1 is for the needed time shift to avoid an error introduced from setting zeros to inf
    times_arr[-1] = big_num  # Arbitrarily high number
    shape = list(fg_axt.shape)
    shape[2] += 1
    # Expand last dimension by one for the inf slot
    new_fg_axt = np.zeros(shape)
    new_fg_axt[:,:,:-1] = fg_axt
    fg_axt = new_fg_axt

    # Fill-in
    if fill_in:
        id_x, id_ax = get_identity_state_actions(p_ax_x)
        max_kappa_states = np.array([np.argmax(kappa[:, -1])])
        if np.any(kappa[max_kappa_states,-1]>0):
            min_time_states = np.array([max_kappa_states[np.argmin(np.dot(eta[max_kappa_states,-1,:], times_arr))]])
            # Fill-in states are the states that are jointly cumulative maximizing, time-minimizing and stable
            fill_in_states = np.intersect1d(np.intersect1d(np.arange(nx)[id_x], max_kappa_states), min_time_states)
            fill_in_actions = np.array([np.argmax(id_ax[:,fill_in_states], axis=0)])
            for x, a in np.array([fill_in_states, fill_in_actions]).transpose():
                for t in np.flip(np.arange(T_f - 1)):
                    future_kappa = kappa[x, t+1]
                    kappa[x, t] = fg_axt[a, x, t] + np.multiply((1 - fg_axt[a, x, t]), future_kappa)
                    pi[x, t] = a
                    eta[x, t, :] = (1 - fg_axt[a, x, t]) * np.tensordot(p_axt_x[a, x, t, :], eta[:, t + 1, :], axes=([0, 0]))

    max_time = np.max(np.where(fg_axt>0)[2]) # This sets the time to be the largest time for which there is positive availability. Do not change!
    x_inds = np.arange(nx)

    for t in reversed(np.arange(max_time+1)):
        eta_contraction_ax_t = np.tensordot(p_axt_x[:, :, t, :], eta[:, t + 1, :], axes=([2, 0]))
        future_kappa_w_actions_axt = np.sum(eta_contraction_ax_t[:,:,:-1], axis=2)
        kappa_w_actions_axt = fg_axt[:, :, t] + np.multiply((1 - fg_axt[:, :, t]), future_kappa_w_actions_axt)
        kappa_update = np.max(kappa_w_actions_axt, axis=0)
        kappa[:,t]=kappa_update
        best_inds = np.where(kappa_w_actions_axt == kappa_update)
        # non_zero_inds = np.where(kappa_update>0)[0]

        # idx = np.ix_(best_inds[0], best_inds[1])
        best_eta_contraction_axt_t = np.ones(eta_contraction_ax_t.shape) * big_num
        best_eta_contraction_axt_t[best_inds[0], best_inds[1], :] = eta_contraction_ax_t[best_inds[0], best_inds[1], :]
        eta_time_expec_2 = np.dot(best_eta_contraction_axt_t, times_arr)
        inner_eq = (fg_axt[:, :, t] * (t + 1)) + np.multiply(1 - fg_axt[:, :, t], eta_time_expec_2)  # "+1" is for the needed time shift to avoid an error introduced from setting zeros to inf. Do not change!
        # inner_eq[inner_eq == 0] = big_num
        pi[:, t] = np.argmin(inner_eq, axis=0).astype('int')
        new_eta = np.multiply((1 - fg_axt[pi[:, t], np.arange(nx), t])[:, np.newaxis], eta_contraction_ax_t[pi[:, t], x_inds, :])
        eta[:, t, :] = new_eta
        eta[:, t, t] = fg_axt[pi[:, t], x_inds, t]

        if np.all(eta[:,:-1,:-1].sum(axis=2) == kappa[:,:-1]) == False:
            warnings.warn("Kappa not equal to summed Eta")

        # We have to loop over individual states because python does not support jagged matrices to optimize only over the A^* actions. Perhaps there is another way by pointwise multiplying a mask of ones for the A* set.
        # for x in non_zero_inds:
        #     a_star_set = best_inds[0][np.where(best_inds[1] == x)[0]].astype('int')
        #     eta_time_expec = np.dot(eta_contraction_axt_t[a_star_set, x, :], times_arr)
        #     pi[x, t] = a_star_set[np.argmin((fg_axt[a_star_set, x, t] * t)  + np.multiply(1 - fg_axt[a_star_set, x, t], eta_time_expec)).astype('int')]
        #     new_eta = np.multiply((1 - fg_axt[pi[x, t], x, t]), eta_contraction_axt_t[pi[x, t], x, :])
        #     eta[x, t, :] = new_eta
        #
        #     # Enforce current time condition
        #     eta[x, t, t] = fg_axt[pi[x, t], x, t]
        #     # eta[x, t, t] = fg_axt[a_g, x, t]

        # if np.all(eta.sum(axis=2)==kappa) == False:
        #     warnings.warn("Kappa not equal to summed Eta")

    if np.all(eta[:,:-1,:-1].sum(axis=2) == kappa[:,:-1]) == False:
        warnings.warn("Kappa not equal to summed Eta")

    # Kappa and eta should be equal when final states and times are summed over in eta
    # np.all(eta[:,:,:].sum(axis=2)==kappa[:,:]) should return True

    return kappa[:,:-1], pi[:,:-1], eta[:,:-1,:-1]


def non_stationary_non_markovian_BFS_solver_ud(x_vec_init, t_vec_init, ho_vecs_init, p_axt_x, ho_op_list, f, goallist, T_f):
    # ud = unidirectional, meaning that there is no influence from the higher-order state-spaces on the low-level
    # This algorithm computes solutions to hierarchical TGMDPs via forward sampling.  It uses an operator factorization
    # to form a basis of low-level control solutions, and BFS to for the forward sampler, pruning leaves with
    # sublimated solutions to higher-level state-spaces.
    # ho_op_list is a list of higher-order transition operators, such as physiological or logical states-spaces.

    X_kappas = dict()
    X_pis = dict()
    X_etas = dict()

    # Check to see which goal states for which you need a spatiotemporal basis
    non_identity_states = []
    temporal_range_dict = dict()
    for i, xg in enumerate(goallist):
        if not np.any(np.where(p_axt_x[:, xg, :, :])[2] == xg):
            non_identity_states.append(xg)
            temporal_range_dict[xg] = np.where(p_axt_x[:, xg, :, :])[1]

    # First compute TGMDP solutions for states that only need shortest-paths
    goal_states = np.setdiff1d(goallist, non_identity_states)
    for i, xg in enumerate(goal_states):
        fg_axt = restrict_to_state(f, xg, i+1),
        kappa, pi, eta = feasibility_iteration_flat_dense(p_axt_x, fg_axt)
        X_kappas[xg], X_pis[xg], X_etas[xg] = (kappa, pi, eta)

    # Now compute TGMDP solutions for states that need a temporal basis
    for xg in temporal_range_dict.keys():
        for t_target in temporal_range_dict[xg]:
            goal_var_ind = np.where(f[:, :, xg, :])[0]
            fg_axt = restrict_to_state_time(f, xg, goal_var_ind, t_target)
            kappa, pi, eta = feasibility_iteration_flat_dense(p_axt_x, fg_axt)
            X_kappas[(xg, t_target)], X_pis[(xg, t_target)], X_etas[(xg, t_target)] = (kappa, pi, eta)

    # Compute the lookahead operators for each Higher-Order Transition Operator in ho_op_list
    omega_list = []
    for op in enumerate(ho_op_list):
        omega_list.append([])

    # Sample polices via BFS with sublimation inequality pruning and infeasibility pruning
    # Sublimated pruning terminates nodes when the feasibility of the higher-order spaces coresponding to terminal states is zero
    # Infeasibility pruning will use the low-level feasibility information (kappa_g = 0) in order to terminate nodes

    best_kappa, act_seq, best_eta, leaf_nodes = tgmdp_bredth_first_plan_search_nsnm_ud(x_vec_init, t_vec_init, ho_vecs_init, X_etas, X_kappas,)


def compute_feas_func_state_time_cover(p_axt_x, f, p_ax_x, goalstates):
    _, nx, T_f, _ = p_axt_x.shape
    kappa_dict = dict()
    pi_dict = dict()
    sparse_eta_dict = dict()
    for i, xg in enumerate(goalstates):
        valid_times = np.unique(np.where(f[1:, :, xg, :]>0)[2])
        for t in valid_times:
            fgt_axt = restrict_to_state_time(f, xg, i + 1, t)
            kappa, pi, eta = feasibility_iteration_flat_dense(p_axt_x, fgt_axt, p_ax_x = None, fill_in = False)
            kappa_dict[(xg, t)] = kappa
            pi_dict[(xg, t)] = pi
            eta_full = np.zeros([nx*T_f,nx*T_f])
            eta_full[:, (xg * T_f):(xg * T_f)+T_f] = eta.reshape(nx * T_f, T_f)
            # eta_dict[(xg, t)] = sp.sparse.csr_matrix(eta.reshape(nx * T_f, nx * T_f))
            # test = sp.sparse.csr_matrix((nx*T_f, nx*T_f))
            # tt = sp.sparse.csr_matrix(eta.reshape(nx * T_f, T_f))
            # eta_dict[(xg, t)] = sp.sparse.csr_matrix(eta.reshape(nx * T_f, T_f))
            sparse_eta_dict[(xg, t)] = sp.sparse.csr_matrix(eta_full)
            # del eta

    return kappa_dict, pi_dict, sparse_eta_dict


def compute_feas_func_state_time_cover_no_goals(p_axt_x, f, p_ax_x):
    _, nx, T_f, _ = p_axt_x.shape
    kappa_dict = dict()
    pi_dict = dict()
    eta_dict = dict()
    for i, xg in enumerate(goalstates):
        valid_times = np.unique(np.where(f[1:, :, xg, :]>0)[2])
        for t in valid_times:
            fgt_axt = restrict_to_state_time(f, xg, i + 1, t)
            kappa, pi, eta = feasibility_iteration_flat_dense(p_axt_x, fgt_axt, p_ax_x = None, fill_in = False)
            kappa_dict[(xg, t)] = kappa
            pi_dict[(xg, t)] = pi
            # eta_dict[(xg, t)] = sp.sparse.csr_matrix(eta.reshape(nx * T_f, nx * T_f))
            eta_dict[(xg, t)] = sp.sparse.csr_matrix(eta.reshape(nx * T_f, T_f))
            del eta

    return kappa_dict, pi_dict, eta_dict

def sublimate(p_gyt_y, fg_y):
    # This is a simple wrapper for feasibility_iteration run on a high-level space (aka "sublimation"), which is where
    # the sublimated kappa function bounds the hierarchical problem.
    kappa, pi, eta = feasibility_iteration_flat_dense(p_gyt_y, fg_y)
    return kappa, pi, eta


# TODO: Not Finished!  Might want to delete and start over using new dense feas-iter code above as template.
def feasibility_iteration_flat_sparse(smat_p_axt_x, fg_axt, p_ax_x = None, fill_in = False):
    # P_x must be [nx',nx,na,T_f]:  P_x(x'|x,a,t)
    # f must be [nx,T_f]
    na, nx, T_f, _ = p_axt_x.shape
    x_g = np.where(fg_axt > 0)[1][0]
    a_g = np.where(fg_axt[:, x_g, :] > 0)[1][0]
    kappa = np.zeros([nx, T_f])
    eta = np.zeros([nx, T_f,
                    T_f])  # Typicially eta(x_f,t_f|x,t) should be [nx,T_f,nx,T_f] or [ng,T_f,nx,T_f], however assuming f encodes a single goal, we can drop the index
    pi = np.zeros([nx, T_f]).astype('int')

    # Caution: This assumes p_axt_x is stationary (even though it is indexed by time).  If it is non-stationary, one needs to compute time-dependent identity state-actions.
    id_x, id_ax = get_identity_state_actions(p_axt_x[:, :, 0, :])

    # Boundary Conditions
    kappa[:, -1] = np.amax(fg_axt[:, :, -1], axis=0)
    pi[:, -1] = np.argmax(fg_axt[:, :, -1], axis=0).astype('int')
    # eta[:,-1,-1] = f[range(nx),pi[:,-1],-1]
    for t_f in range(T_f):
        eta[:, t_f, t_f] = fg_axt[pi[:, t_f], range(nx), t_f]

    times_arr = np.arange(T_f)

    # Fill-in
    if fill_in:
        id_x, id_ax = get_identity_state_actions(p_ax_x)
        max_kappa_states = np.array([np.argmax(kappa[:, -1])])
        if np.any(kappa[max_kappa_states, -1] > 0):
            min_time_states = np.array([max_kappa_states[np.argmin(np.dot(eta[max_kappa_states, -1, :], times_arr))]])
            # Fill-in states are the states that are jointly cumulative maximizing, time-minimizing and stable
            fill_in_states = np.intersect1d(np.intersect1d(np.arange(nx)[id_x], max_kappa_states), min_time_states)
            fill_in_actions = np.array([np.argmax(id_ax[:, fill_in_states], axis=0)])
            for x, a in np.array([fill_in_states, fill_in_actions]).transpose():
                for t in np.flip(np.arange(T_f - 1)):
                    future_kappa = kappa[x, t + 1]
                    kappa[x, t] = fg_axt[a, x, t] + np.multiply((1 - fg_axt[a, x, t]), future_kappa)
                    pi[x, t] = a
                    eta[x, t, :] = (1 - fg_axt[a, x, t]) * np.tensordot(p_axt_x[a, x, t, :], eta[:, t + 1, :],
                                                                        axes=([0, 0]))

    for t in reversed(np.arange(T_f - 1)):
        eta_contraction_axt_t = np.tensordot(p_axt_x[:, :, t, :], eta[:, t + 1, :], axes=([2, 0]))
        future_kappa_w_actions_axt = np.sum(eta_contraction_axt_t, axis=2)
        kappa_w_actions_axt = fg_axt[:, :, t] + np.multiply((1 - fg_axt[:, :, t]), future_kappa_w_actions_axt)
        kappa_update = np.max(kappa_w_actions_axt, axis=0)
        kappa[:, t] = kappa_update
        best_inds = np.where(kappa_w_actions_axt == kappa_update)
        non_zero_inds = np.where(kappa_update > 0)[0]

        # We have to loop over individual states because python does not support jagged matrices to optimize over A^* actions. Perhaps there is another way.
        for x in non_zero_inds:
            a_star_set = best_inds[0][np.where(best_inds[1] == x)[0]].astype('int')
            eta_time_expec = np.dot(eta_contraction_axt_t[a_star_set, x, :], times_arr)
            pi[x, t] = a_star_set[np.argmin(np.multiply(1 - fg_axt[a_star_set, x, t], eta_time_expec)).astype('int')]
            new_eta = np.multiply((1 - fg_axt[pi[x, t], x, t]), eta_contraction_axt_t[pi[x, t], x, :])
            eta[x, t, :] = new_eta

            # Enforce Boundary Condition
            eta[x, t, t] = fg_axt[pi[x, t], x, t]
            # eta[x, t, t] = fg_axt[a_g, x, t]

    if np.all(eta.sum(axis=2) == kappa) == False:
        warnings.warn("Kappa not equal to summed Eta")


    return kappa, pi, eta


def feasibility_iteration_semi_markov_dense(jump_pxt_xt, fg_axt, pi_ens, p_ax_x = None, fill_in = False):
    # P_x must be [nx',nx,na,T_f]:  P_x(x'|x,a,t)
    # f must be [nx,T_f]
    # pi_ens = [npi,nx,T_f].  Matrix of actions which are selected from a given (policy,state,time))
    # This algorithm assumes that the special action, a_g, is chosen as the result of all pi
    # p_ax_x is an option parameter, which is the stationary transition operator used to form jump_pxt_xt, for fill-in
    # fg_axt = np.swapaxes(fg_xat, 0, 1)
    nx = jump_pxt_xt.shape[1]
    npi = jump_pxt_xt.shape[0]
    T_f = jump_pxt_xt.shape[2]
    x_g = np.where(fg_axt > 0)[1][0]
    a_g = np.where(fg_axt[:, x_g, :] > 0)[0][0]
    a_g_vec = np.array([a_g]*npi)

    kappa = np.zeros([nx,T_f])
    eta = np.zeros([nx,T_f,T_f]) # Typicially eta(x_f,t_f|x,t) should be [nx,T_f,nx,T_f] or [ng,T_f,nx,T_f], however assuming f encodes a single goal, we can drop the index
    mu = np.zeros([nx,T_f]).astype('int')

    # Boundary Conditions
    kappa[:,-1] = np.amax(fg_axt[np.array(pi_ens).flatten(), np.tile(np.arange(nx), npi),-1].reshape([npi,nx]), axis=0)
    mu[:,-1] = np.argmax(fg_axt[np.array(pi_ens).flatten(), np.tile(np.arange(nx), npi), -1].reshape([npi,nx]), axis=0).astype('int')
    eta[:, -1, -1] = fg_axt[a_g_vec, range(nx), -1]
    # for t_f in range(T_f):


    times_arr = np.arange(T_f)
    xt_arr_old = np.empty((0,2),dtype='int8')

    kappa_old = kappa.copy()
    for tau in reversed(np.arange(T_f)):
        eta_contraction_pxt_t = np.tensordot(jump_pxt_xt, eta, axes=([3, 4], [0, 1]))
        future_kappa_w_pols_pxt = np.sum(eta_contraction_pxt_t, axis=3)
        f_eval = fg_axt[np.array(pi_ens).flatten().repeat(T_f), np.arange(nx).repeat(T_f*npi),
                        np.tile(np.arange(T_f), npi * nx)].reshape([npi, nx, T_f])
        kappa_w_policies = f_eval + np.multiply((1 - f_eval), future_kappa_w_pols_pxt)
        kappa = np.max(kappa_w_policies, axis=0)

        # best_inds = np.where(kappa_w_actions_pxt == np.repeat(kappa_update[np.newaxis,:,:], npi,axis=0))
        best_inds = np.where(kappa_w_policies == kappa)
        kappa_diff = (kappa-kappa_old)[:,:-1]  # We remove the last time here because it is a boundary condition. Will throw an error if included b/c there is not best poilcy
        diff_inds = np.array(list(np.where(kappa_diff > 0))).transpose()
        # non_zero_inds_x, non_zero_inds_t = np.where(kappa > 0)
        # xt_arr_nonzero = np.array([non_zero_inds_x, non_zero_inds_t]).transpose()
        # xt_arr_nonzero_tuple = np.unique(xt_arr_nonzero.view(xt_arr_nonzero.dtype.descr * xt_arr_nonzero.shape[1]))
        # xt_tuples_nonzero = np.setdiff1d(xt_arr_nonzero_tuple, xt_tuples_nonzero_old)
        # # xt_arr = np.unique(np.concatenate([xt_tuples_nonzero, xt_tuples_old], axis=0), axis=0)

        # We have to loop over individual states because python does not support jagged matrices to optimize over A^* actions. Perhaps there is another way.
        for (x,t) in diff_inds:
            pi_star = best_inds[0][np.all(np.array([best_inds[1], best_inds[2]]).transpose() == np.array([x,t]), axis=1)]
            a_star = np.array([pi_ens[pi][x] for pi in pi_star])
            sub_eta_time_expec_ax_t = np.tensordot(eta_contraction_pxt_t[pi_star, x, t, :], times_arr, axes=[1,0])
            # sub_eta_time_expec_ax_t[sub_eta_time_expec_ax_t == 0] = np.inf
            min_arg = np.argmin(np.multiply(1 - fg_axt[a_star, x, t], sub_eta_time_expec_ax_t)).astype('int')
            mu[x, t] = pi_star[min_arg]
            new_eta = np.multiply((1 - fg_axt[a_star[min_arg], x, t]), eta_contraction_pxt_t[mu[x, t], x, t,:])
            eta[x,t,:] = new_eta

            # Enforce Boundary Condition
            eta[x, t, t] = fg_axt[mu[x, t][x, t], x, t]
            # eta[x,t,t] = fg_axt[a_g, x, t]

        if not np.all(eta.sum(axis=2) == kappa):
            warnings.warn("Kappa not equal to summed Eta")
        if np.all(kappa == kappa_old):
            print('Total number of iterations: ', T_f-1-tau)
            break
        else:
            kappa_old = kappa.copy()


    if np.all(eta.sum(axis=2)==kappa) == False:
        warnings.warn("Kappa not equal to summed Eta")

    return kappa, mu, eta


def feas_iter_sm_dense_period(jump_e_pxt_xt, fg_axt, pi_mat, kappa_init, mu_init, eta_init, D_k, k_start, T_f):
    # D_k is the period duration for period k of environment e_k
    # jump_e_pxt_xt is a jump policy formed from a set of stationary X policies conditioned on environment e,
    # for period k_e, and the algorithm modifies the jump-operator so the policies can only be used if
    # they do not violate the horizon parameter.
    # Also, this algorithm assumes that the time t_0 of the jump-operator is *relative* to the period start.
    # pi_ens = array([npi,nx,T_f]) -> action.  Tensor of actions which are selected from a given (policy, state, time))
    # This algorithm assumes that the special action, a_g, is chosen as the result of all pi
    # kappa, pi, and eta, are passed in from the subsequent period.

    npi = jump_e_pxt_xt.shape[0]
    nx = jump_e_pxt_xt.shape[1]
    # T_f = jump_e_pxt_xt.shape[2]
    x_g = np.where(fg_axt > 0)[1][0]
    D_k_ext = D_k + 1  # This extends the period duration by one for indexing into the next period
    a_g = np.where(fg_axt[:, x_g, :] > 0)[0][0]

    kappa = kappa_init
    mu = mu_init
    eta = eta_init

    times_arr = np.arange(T_f)
    xt_arr_old = np.empty((0,2), dtype='int8')

    kappa_old = kappa
    counter = 0

    for step in reversed(np.arange(D_k_ext)):  # step isn't used, it just bounds the allowed number of iterations by the number of timesteps
        eta_contraction_pxt_t = np.tensordot(jump_e_pxt_xt, eta, axes=([3, 4], [0, 1]))
        future_kappa_w_policies_pxt = np.sum(eta_contraction_pxt_t, axis=3)
        f_eval = fg_axt[pi_mat.flatten().repeat(D_k_ext), np.arange(nx).repeat(D_k_ext * npi),
                        np.tile(np.arange(k_start, k_start + D_k_ext), npi * nx)].reshape([npi, nx, D_k_ext])
        kappa_w_policies_pxt = f_eval + np.multiply((1 - f_eval), future_kappa_w_policies_pxt)
        kappa = np.max(kappa_w_policies_pxt, axis=0)

        # best_inds = np.where(kappa_w_actions_pxt == np.repeat(kappa_update[np.newaxis,:,:], npi,axis=0))
        best_inds = np.where(kappa_w_policies_pxt == kappa)
        kappa_diff = kappa - kappa_old
        diff_inds = np.array(list(np.where(kappa_diff > 0))).transpose()
        # non_zero_inds_x, non_zero_inds_t = np.where(kappa > 0)
        # xt_arr_nonzero = np.array([non_zero_inds_x, non_zero_inds_t]).transpose()
        # xt_arr_nonzero_tuple = np.unique(xt_arr_nonzero.view(xt_arr_nonzero.dtype.descr * xt_arr_nonzero.shape[1]))
        # xt_tuples_nonzero = np.setdiff1d(xt_arr_nonzero_tuple, xt_tuples_nonzero_old)
        # # xt_arr = np.unique(np.concatenate([xt_tuples_nonzero, xt_tuples_old], axis=0), axis=0)

        # We have to loop over individual states because python does not support jagged matrices to optimize over A^* actions. Perhaps there is another way.
        for (x, t) in diff_inds:
            tau = t + k_start  # t is relative to the period, tau is absolute time needed to correctly index.
            pi_star = best_inds[0][np.all(np.array([best_inds[1], best_inds[2]]).transpose() == np.array([x,t]), axis=1)]
            a_star = np.array([pi_mat[pi, x] for pi in pi_star])
            sub_eta_time_expec_ax_t = np.tensordot(eta_contraction_pxt_t[pi_star, x, t, :], times_arr, axes=[1, 0])
            # sub_eta_time_expec_ax_t[sub_eta_time_expec_ax_t == 0] = np.inf
            min_arg = np.argmin(np.multiply(1 - fg_axt[a_star, x, t], sub_eta_time_expec_ax_t)).astype('int')
            mu[x, t] = int(pi_star[min_arg])
            new_eta = np.multiply((1 - fg_axt[a_star[min_arg], x, t]), eta_contraction_pxt_t[int(mu[x, t]), x, t, :])
            eta[x, t, :] = new_eta

            # Enforce Boundary Condition.  This operation can technically be done at initiation, but it here means that we do not need to consider states which have been updated but have not converged in the main loop of the algorithm, which is what would occur if we initialized eta with all non-final-time boundary condition at start..
            eta[x, t, tau] = fg_axt[a_g, x, tau]  # The boundary condition uses both t and tau because we aren't representing the full time domain on the second dimension of the eta tensor.
        counter += 1
        if np.all(kappa == kappa_old):
            print('Total number of iterations: ', counter)
            break
        else:
            kappa_old = kappa


    if np.all(eta.sum(axis=2)[:,:-1] == kappa[:,:-1]) == False:
        warnings.warn("Kappa not equal to summed Eta")

    return kappa, mu, eta


def feas_iter_sm_piecewise_env(jump_pxt_xt_dict, fg_axt, pi_env_dict, env_durations, T_f):
    # Feasibility Iteration for Semi-Markov Jump operators conditioned on a specific period of an environment sequence.
    # The time domain of jump_pxt_xt only has to be as long as the longest period duration.
    # This is the same algorithm as the regular semi-markov feasibility iteration (SMFI), but it breaks down a
    # non-stationary environment into pieces and calls SMFI for each stationary portion of time.
    # pi_ens_list is a list of pi_ens (ensemble), where pi_ens_list[e] returns the one corresponding to environment e.
    # fg_axt = np.swapaxes(fg_xat, 0, 1)
    # T_max is the longest time allowed for the jump operator, T_f is the horizon for the optimiztion
    na, nx, T_max = fg_axt.shape
    T_f = env_durations.sum() + 1  # The plus one is by convention for this optimization so that we can index a matrix with the parameter T_f instead of T_f-1.  This is helpful for creating the kappa and eta functions for the final period when we breaking the problem down into individual periods.
    # T_f = env_durations.cumsum()
    ne = len(env_durations)
    npi = jump_pxt_xt_dict[0].shape[0]
    x_g = np.where(fg_axt > 0)[1][0]
    a_g = np.where(fg_axt[:, x_g, :] > 0)[0][0]
    a_g_vec = np.array([a_g] * npi)
    env_durations_rev = np.flip(env_durations)
    period_start_times = np.concatenate([np.array([0]), env_durations.cumsum()])
    remaining_time = env_durations_rev.cumsum() # Returns how much time is remaining until T_f for each period start-time.

    kappa = np.zeros([nx, T_f])
    eta = np.zeros([nx, T_f, T_f])  # Typicially eta(x_f,t_f|x,t) should be [nx,T_f,nx,T_f] or [ng,T_f,nx,T_f], however assuming f encodes a single goal, we can drop the index
    mu = np.zeros([nx, T_f]).astype('int')

    # Boundary Conditions
    kappa[:, -1] = np.amax(fg_axt[:, :, -1], axis=0)
    mu[:, -1] = np.argmax(fg_axt[:, :, -1], axis=0).astype('int')
    eta[:, -1, -1] = fg_axt[a_g_vec, range(nx), -1]


    # TODO: Change e to k and index environmets by the period.
    for e in np.flip(np.arange(ne)):
        jump_e_pxt_xt = jump_pxt_xt_dict[e]
        pi_ens = pi_env_dict[e]
        D_k = env_durations[e]

        # Create mask for jump_e_pxt_xt, setting xt -> x't' probabilities to zero if t' - t exceeds the duration D_k
        state_inds = np.arange(nx)
        arr = np.arange(D_k + 1)
        time_list_1_unedited = np.arange(D_k + 1).repeat(D_k + 1)
        time_list_2_unedited = np.array([arr + t for t in range(D_k + 1)]).flatten()
        bad_inds = np.where(time_list_2_unedited >= D_k + 1)[0]
        time_list_1 = np.delete(time_list_1_unedited, bad_inds)
        time_list_2 = np.delete(time_list_2_unedited, bad_inds)
        # time_list_2[time_list_2 >= D_k + 1] = D_k

        mask = np.zeros([npi, nx, D_k + 1, nx, D_k + 1])
        mask[:, :, time_list_1, :, time_list_2] = 1
        jump_e_pxt_xt_masked = jump_e_pxt_xt[:, :, :D_k+1, :, :D_k+1] * mask

        k_start = period_start_times[e]
        next_k_start = period_start_times[e+1]
        duration = int(T_f - k_start)
        time_span = np.arange(int(T_f - k_start))
        kappa_k_init = np.zeros([nx, D_k + 1])  # Initial kappa for period k
        kappa_k_init[:, -1] = kappa[:, next_k_start]
        mu_k_init = np.zeros([nx, D_k + 1])  # Initial kappa for period k
        mu_k_init[:, -1] = mu[:, next_k_start]
        eta_k_init = np.zeros([nx, D_k + 1, T_f])
        eta_k_init[:, -1, :] = eta[:, next_k_start, :]

        # pi_mat = np.array(pi_ens)[:, k_start:(next_k_start+1)]
        pi_mat = np.array(pi_ens)

        # NOTES: We must to include the range of time from k_start to the end in eta for t_f.

        kappa_period, mu_period, eta_period = feas_iter_sm_dense_period(jump_e_pxt_xt_masked, fg_axt, pi_mat, kappa_k_init, mu_k_init, eta_k_init, env_durations[e], k_start, T_f)
        kappa[:, k_start:(k_start + D_k)] = kappa_period[:, :-1]  # This -1 here is due to the fact that we represent the first time of the next period in the last entry of the array and we want to get rid of it.
        mu[:, k_start:(k_start + D_k)] = mu_period[:, :-1]
        eta[:, k_start:(k_start + D_k), :] = eta_period[:, :-1, :]

    return kappa, mu, eta

def value_iteration_min_cost_bse(p_as_s, cost_func_sa, terminal_inds, terminal_cost=0):
    # bse = Boundary State Extension, which runs one extra step of value iteration at the boundary after convergence.
    # bse is needed to know how to act from a boundary state under a policy which terminates at the same boundary state.
    # reward_func_sa is a state reward function (no action).  For tLMDP, s = (sigma, x)
    na = p_as_s.shape[0]
    ns = p_as_s.shape[1]
    terminal_inds_sa_lin_ind =  [np.array([terminal * na]).repeat(na)+np.arange(na) for terminal in terminal_inds]
    max_steps = p_as_s.shape[1] + 10
    arb_big_num = 1E7
    # v = np.ones(ns) * arb_big_num
    v_2 = np.ones(ns) * arb_big_num
    # v[terminal_inds] = terminal_cost
    v_2[terminal_inds] = terminal_cost
    state_inds = np.arange(ns)
    tall_as_s = p_as_s.reshape([ns * na, ns])
    tall_sa_s = np.swapaxes(p_as_s, 0, 1).reshape([ns * na, ns])
    tall_cost_as_vec = cost_func_sa.transpose().flatten()
    # state_inds_action_0 = [i * na for i in range(ns)]
    delta = 10000
    for t in range(max_steps):
        # expectation_v_prime = np.tensordot(p_as_s, v, axes=([2], [0]))
        # # expectation_v_prime[terminal_inds] = terminal_rewards[terminal_inds]
        # v_ax_new = cost_func_sa.transpose() + expectation_v_prime
        # v_ax_new[:,terminal_inds] = terminal_cost # Set the boundary states to have a constant value (First-exit formulation)
        # v_ax_argmax_inds = v_ax_new.argmin(axis = 0)
        # v_new = v_ax_new[v_ax_argmax_inds, state_inds]
        # # v_new[terminal_inds] = terminal_rewards[terminal_inds]
        # not_inf_inds = np.where(v_new < arb_big_num)
        # delta = np.sum(np.abs(v[not_inf_inds] - v_new[not_inf_inds]))
        # # print(delta)
        # v = v_new

        expectation_v_prime_2_as = np.dot(tall_as_s, v_2)
        v_ax_new_2 = tall_cost_as_vec + expectation_v_prime_2_as
        v_ax_new_2 = v_ax_new_2.reshape(na, ns)
        v_ax_new_2[:, terminal_inds] = terminal_cost
        v_ax_argmax_inds_2 = v_ax_new_2.reshape([na, ns]).argmin(axis=0)
        v_2_new = v_ax_new_2[v_ax_argmax_inds_2, state_inds]
        not_inf_inds = np.where(v_2_new < arb_big_num)
        delta_2 = np.sum(np.abs(v_2[not_inf_inds] - v_2_new[not_inf_inds]))
        # print('VI-step: ',t)
        # print(max_steps)
        # print(delta)
        v_2 = v_2_new
        if delta_2 < 0.001:
            # if t % (T_f-2) == 0: print("Total Iterations (Value Iter): ", t)
            break

    # Boundary State Extension:
    # First compute policy without BSE, otherwise the wrong actions will be selected near the boundary
    pi = np.argmin(np.dot(p_as_s, v_2), axis=0)

    tall_cost_sa = cost_func_sa.flatten()
    expectation_v_prime_2_sa = np.dot(tall_sa_s[terminal_inds_sa_lin_ind, :], v_2)
    v_sa_new_2 = tall_cost_sa[terminal_inds_sa_lin_ind] + expectation_v_prime_2_sa
    v_sa_max = v_sa_new_2.reshape([len(terminal_inds), na]).transpose().min(axis=0)
    v_2[terminal_inds] = v_sa_max

    # Update the policy on the terminal ind(s) once you've updated the value function with BSE.
    pi[terminal_inds] = np.argmin((np.dot(p_as_s, v_2)).reshape(na, ns)[:, terminal_inds], axis=0)


    return v_2, pi


def value_iteration_min_cost(p_as_s, cost_func_sa, terminal_inds, terminal_cost=0):
    # bse = Boundary State Extension, which runs one extra step of value iteration at the boundary after convergence.
    # bse is needed to know how to act from a boundary state under a policy which terminates at the same boundary state.
    # cost_func_sa is a state cost function (no action).  For tLMDP, s = (sigma, x)
    ns = p_as_s.shape[1]
    na = p_as_s.shape[0]
    terminal_inds_sa_lin_ind =  [np.array([terminal * na]).repeat(na)+np.arange(na) for terminal in terminal_inds]
    max_steps = p_as_s.shape[1] + 10
    arb_big_num = 1E7
    # v = np.ones(ns) * arb_big_num
    v_2 = np.ones(ns) * arb_big_num
    # v[terminal_inds] = terminal_cost
    v_2[terminal_inds] = terminal_cost
    state_inds = np.arange(ns)
    tall_as_s = p_as_s
    # tall_sa_s = np.swapaxes(p_as_s, 0, 1).reshape([ns * na, ns])

    tall_as_s = p_as_s.reshape([ns * na, ns])
    # tall_sa_s = np.swapaxes(p_as_s, 0, 1).reshape([ns * na, ns])

    tall_cost_as_vec = cost_func_sa.transpose().flatten()
    # state_inds_action_0 = [i * na for i in range(ns)]
    delta = 10000
    for t in range(max_steps):
        expectation_v_prime_2_as = tall_as_s.dot(v_2)
        v_ax_new_2 = tall_cost_as_vec + expectation_v_prime_2_as
        v_ax_new_2 = v_ax_new_2.reshape(na, ns)
        v_ax_new_2[:, terminal_inds] = terminal_cost
        v_ax_argmax_inds_2 = v_ax_new_2.reshape([na, ns]).argmin(axis=0)
        v_2_new = v_ax_new_2[v_ax_argmax_inds_2, state_inds]
        not_inf_inds = np.where(v_2_new < arb_big_num)
        delta_2 = np.sum(np.abs(v_2[not_inf_inds] - v_2_new[not_inf_inds]))

        v_2 = v_2_new
        if delta_2 < 0.001:
            # if t % (T_f-2) == 0: print("Total Iterations (Value Iter): ", t)
            break

    # Boundary State Extension:
    # First compute policy without BSE, otherwise the wrong actions will be selected near the boundary
    pi = np.argmin(np.dot(p_as_s, v_2), axis=0)

    # tall_cost_sa = cost_func_sa.flatten()
    # expectation_v_prime_2_sa = np.dot(tall_sa_s[terminal_inds_sa_lin_ind, :], v_2)
    # v_sa_new_2 = tall_cost_sa[terminal_inds_sa_lin_ind] + expectation_v_prime_2_sa
    # v_sa_max = v_sa_new_2.reshape([len(terminal_inds), na]).transpose().min(axis=0)
    # v_2[terminal_inds] = v_sa_max

    # Update the policy on the terminal ind(s) once you've updated the value function with BSE.
    pi[terminal_inds] = np.argmin((np.dot(p_as_s, v_2)).reshape(na, ns)[:, terminal_inds], axis=0)


    return v_2, pi



def create_short_path_stationary_pol_cover_bse(p_as_s):
    # Creates an ensmble of shortest path policies and value functions to each state.
    # BSE stands for boundary state extension, which means that we apply one extra step of value iteration at the end
    # to rerpresent policy and value of acting from the boundary state.

    ns = p_as_s.shape[1]
    na = p_as_s.shape[0]//ns
    terminal_cost = 0
    pi_ens = []
    v_ens = []

    # Value Iteration
    for i in range(ns):
        cost_func_sa = np.ones([ns, na])
        # cost_func_sa[i,:] = terminal_cost
        terminal_state = [i]
        v, pi = value_iteration_min_cost_bse(p_as_s, cost_func_sa, terminal_state, terminal_cost)
        v_ens.append(v)
        pi_ens.append(pi)


    # if t % (T_f-2) == 0: print("Policy Ensemble Completed")
    return v_ens, pi_ens


def create_dense_state_time_jump_op_from_vf(p_as_s, v_ens, T_f):
    # Creates a state-time jump transition operator from a value function basis,
    # where each value function has a single boundary state
    v_ens_arr = np.array(v_ens).astype('int')
    na = p_as_s.shape[0]
    ns = p_as_s.shape[1]
    npi = len(v_ens)  # the number of polices (npi) is equal to the number of value functions
    t_vec = np.arange(T_f)
    x_inds = np.arange(ns)
    jump_pxt_xt = np.zeros([npi, ns, T_f, ns, T_f])
    xt_mat = np.array(list(itertools.product(np.arange(ns), t_vec)))

    # Create next-time vectors for each policy, where each entry in the list is for a given policy:
    # next_time_vec_list = [xt_mat[:, 1] + v_ens[pi][xt_mat[:, 0]] for pi in range(npi)]  #
    # flattened_next_time_vec = np.array(next_time_vec_list).flatten().astype('int')
    # prob_vec = [(next_time_vec_list[pi_ind] < T_f).astype('int') for pi_ind in range(npi)]
    # prob_vec = np.array(prob_vec).flatten()
    # flattened_next_time_vec[flattened_next_time_vec >= T_f] = T_f - 1
    # pi_vec = np.repeat(np.arange(npi), ns * T_f)
    # jump_pxt_xt[pi_vec, np.tile(xt_mat[:, 0], npi), np.tile(xt_mat[:, 1], npi), pi_vec, flattened_next_time_vec] = prob_vec

    next_time_vec_list = np.tile(xt_mat[:, 1], npi) + v_ens_arr[np.arange(npi).repeat(ns*T_f), np.tile(xt_mat[:, 0],npi)]
    # Create next-time vectors for each policy, where each entry in the list is for a given policy:
    # next_time_vec_list2 = [xt_mat[:, 1] + v_ens[pi][xt_mat[:, 0]] for pi in range(npi)] #
    # flattened_next_time_vec2 = np.array(next_time_vec_list2).flatten().astype('int')
    prob_vec = (next_time_vec_list < T_f).astype('int')
    next_time_vec_list[next_time_vec_list >= T_f] = T_f-1
    pi_vec = np.repeat(np.arange(npi), ns * T_f)
    jump_pxt_xt[pi_vec, np.tile(xt_mat[:, 0], npi), np.tile(xt_mat[:, 1], npi), pi_vec, next_time_vec_list] = prob_vec

    # for pi_ind in range(npi):
    #     v = v_ens[pi_ind]
    #     ## t_tile = np.tile(t_vec, T_f)
    #     ## t_diff_tile = t_diff_mat2.flatten()
    #     # t_diff_mat = np.tile(t_vec, ns).reshape([ns, T_f]) + v[:,np.newaxis]
    #     # for xt in itertools.product(np.arange(ns),t_vec):
    #     #     next_time = int(xt[1] + v[xt[0]])
    #     #     prob = int(next_time < T_f)
    #     #     next_time_capped = min(next_time, T_f-1)
    #     #     jump_pxt_xt[pi_ind, xt[0], xt[1], pi_ind, next_time_capped] = prob
    #
    #     next_time_vec = (xt_mat[:,1] + v_ens[pi_ind][xt_mat[:,0]]).astype('int')
    #     prob_vec = (next_time_vec_list[pi_ind] < T_f).astype('int')
    #     next_time_vec[next_time_vec >= T_f] = T_f-1
    #     jump_pxt_xt[pi_ind, xt_mat[:, 0], xt_mat[:, 1], np.array([pi_ind]*len(prob_vec)), next_time_vec] = prob_vec

    # crow_indices = torch.tensor([0, 2, 4])
    # col_indices = torch.tensor([0, 1, 0, 1])
    # values = torch.tensor([1, 2, 3, 4])
    # csr2 = torch.sparse_csr_tensor(crow_indices, col_indices, values, dtype=torch.double)
    # csr = torch.sparse_csr_tensor(crow_indices, col_indices, values, dtype=torch.double)
    #
    # csr3 = torch.sparse_csr_tensor(crow_indices, col_indices, values, dtype=torch.double)
    #
    # coord_mat = np.array([pi_vec, np.tile(xt_mat[:, 0], npi), np.tile(xt_mat[:, 1], npi), pi_vec, next_time_vec_list]).transpose()
    # coord_mat2 = coord_mat.transpose()
    #
    # test = tf.sparse.SparseTensor(coord_mat2, prob_vec, [npi, ns, T_f, ns, T_f])

    # if t % (T_f-2) == 0: print("State-Time Jump Operator Completed")

    return jump_pxt_xt


def create_sparse_state_time_jump_op_from_vf(p_as_s, v_ens, T_f, clip_time = False):
    # Creates a state-time jump transition operator from a value function basis,
    # where each value function has a single boundary state
    v_ens_arr = np.array(v_ens).astype('int')
    na = p_as_s.shape[0]
    ns = p_as_s.shape[1]
    npi = len(v_ens)  # the number of polices (npi) is equal to the number of value functions
    t_vec = np.arange(T_f)
    x_inds = np.arange(ns)
    jump_pxt_xt = np.zeros([npi, ns, T_f, ns, T_f])
    jump_list = []
    xt_mat = np.array(list(itertools.product(np.arange(ns), t_vec)))

    # xt_vec = np.tile(np.arange(ns) * t_vec, T_f) + np.repeat(t_vec, ns)
    # next_xt_vec =

    # Create next-time vectors for each policy, where each entry in the list is for a given policy:
    # next_time_vec_list = [xt_mat[:, 1] + v_ens[pi][xt_mat[:, 0]] for pi in range(npi)]  #
    # flattened_next_time_vec = np.array(next_time_vec_list).flatten().astype('int')
    # prob_vec = [(next_time_vec_list[pi_ind] < T_f).astype('int') for pi_ind in range(npi)]
    # prob_vec = np.array(prob_vec).flatten()
    # flattened_next_time_vec[flattened_next_time_vec >= T_f] = T_f - 1
    # pi_vec = np.repeat(np.arange(npi), ns * T_f)
    # jump_pxt_xt[pi_vec, np.tile(xt_mat[:, 0], npi), np.tile(xt_mat[:, 1], npi), pi_vec, flattened_next_time_vec] = prob_vec

    next_time_vec_list = np.tile(xt_mat[:, 1], npi) + v_ens_arr[np.arange(npi).repeat(ns*T_f), np.tile(xt_mat[:, 0],npi)]
    # Create next-time vectors for each policy, where each entry in the list is for a given policy:
    # next_time_vec_list2 = [xt_mat[:, 1] + v_ens[pi][xt_mat[:, 0]] for pi in range(npi)] #
    # flattened_next_time_vec2 = np.array(next_time_vec_list2).flatten().astype('int')
    prob_vec = (next_time_vec_list < T_f).astype('int')
    next_time_vec_list[next_time_vec_list >= T_f] = T_f-1
    x_vec = np.repeat(np.arange(ns), ns * T_f)
    pi_vec = np.repeat(np.arange(npi), ns * T_f)

    jump_pxt_xt[pi_vec, np.tile(xt_mat[:, 0], npi), np.tile(xt_mat[:, 1], npi), x_vec, next_time_vec_list] = prob_vec

    # for pi in range(npi):
    #     # sp.csr(prob_vec,())


    # for pi_ind in range(npi):
    #     v = v_ens[pi_ind]
    #     ## t_tile = np.tile(t_vec, T_f)
    #     ## t_diff_tile = t_diff_mat2.flatten()
    #     # t_diff_mat = np.tile(t_vec, ns).reshape([ns, T_f]) + v[:,np.newaxis]
    #     # for xt in itertools.product(np.arange(ns),t_vec):
    #     #     next_time = int(xt[1] + v[xt[0]])
    #     #     prob = int(next_time < T_f)
    #     #     next_time_capped = min(next_time, T_f-1)
    #     #     jump_pxt_xt[pi_ind, xt[0], xt[1], pi_ind, next_time_capped] = prob
    #
    #     next_time_vec = (xt_mat[:,1] + v_ens[pi_ind][xt_mat[:,0]]).astype('int')
    #     prob_vec = (next_time_vec_list[pi_ind] < T_f).astype('int')
    #     next_time_vec[next_time_vec >= T_f] = T_f-1
    #     jump_pxt_xt[pi_ind, xt_mat[:, 0], xt_mat[:, 1], np.array([pi_ind]*len(prob_vec)), next_time_vec] = prob_vec

    # crow_indices = torch.tensor([0, 2, 4])
    # col_indices = torch.tensor([0, 1, 0, 1])
    # values = torch.tensor([1, 2, 3, 4])
    # csr2 = torch.sparse_csr_tensor(crow_indices, col_indices, values, dtype=torch.double)
    # csr = torch.sparse_csr_tensor(crow_indices, col_indices, values, dtype=torch.double)
    #
    # csr3 = torch.sparse_csr_tensor(crow_indices, col_indices, values, dtype=torch.double)
    #
    # coord_mat = np.array([pi_vec, np.tile(xt_mat[:, 0], npi), np.tile(xt_mat[:, 1], npi), pi_vec, next_time_vec_list]).transpose()
    # coord_mat2 = coord_mat.transpose()
    #
    # test = tf.sparse.SparseTensor(coord_mat2, prob_vec, [npi, ns, T_f, ns, T_f])

    # if t % (T_f-2) == 0: print("State-Time Jump Operator Completed")

    return jump_pxt_xt

def build_cartesian_operator_dense(ns, nx, na, nis, f, X, int_op_lst):
    as_tuples = list(itertools.product(np.arange(na), np.arange(nx), np.arange(nis), np.arange(nis), np.arange(nis)))
    s_tuples = list(itertools.product(np.arange(nx), np.arange(nis), np.arange(nis), np.arange(nis)))
    row_dict = dict(zip(as_tuples, np.arange(len(as_tuples))))
    row_dict_ind_2_tup = dict(zip(np.arange(len(as_tuples)), as_tuples))
    col_dict = dict(zip(s_tuples, np.arange(len(s_tuples))))
    col_dict_ind_2_tup = dict(zip(np.arange(len(s_tuples)), s_tuples))
    # sp.sparse.csr_matrix
    action_state_tuples = list(
        itertools.product(np.arange(na), np.arange(nx), np.arange(nis), np.arange(nis), np.arange(nis)))
    null_op = np.zeros(X.P_ax_x.shape)


    for a_ind in range(na):
        null_op[a_ind, np.arange(nx), np.arange(nx)] = 1

    row_inds = np.zeros(ns * na, dtype='int')
    col_inds = np.zeros(ns * na, dtype='int')
    data = np.zeros(ns * na)
    for ind, ast in enumerate(action_state_tuples):
        if any(np.array(ast[2:5]) == 0):
            p_ax_x = null_op
        else:
            p_ax_x = X.P_ax_x
        next_x_pos = np.where(p_ax_x[ast[0], ast[1], :])[0][0]
        g_var = np.where(f[:, ast[0], ast[1], 0])[0][0]
        next_pos_list = [next_x_pos]
        for i, int_ss in enumerate(int_op_lst):
            next_pos = np.where(int_ss.P_a_s_s[int_ss.task_ind_2_goal.get(g_var, 0), ast[i + 2], :])[0][0]
            next_pos_list.append(next_pos)
        next_pos_tuple = tuple(next_pos_list)
        col_inds[ind] = int(col_dict[next_pos_tuple])
        # next_w_pos = np.where(w_ss.P_a_s_s[w_ss.task_ind_2_goal.get(g_var, 0), ast[2], :])[0][0]
        # next_y_pos = np.where(y_ss.P_a_s_s[y_ss.task_ind_2_goal.get(g_var, 0), ast[3], :])[0][0]
        # next_z_pos = np.where(z_ss.P_a_s_s[z_ss.task_ind_2_goal.get(g_var, 0), ast[4], :])[0][0]
        # p_axr_xr[ast[0],ast[1],ast[2],ast[3],ast[4],next_w_pos,next_y_pos,next_z_pos]=1
        row_inds[ind] = int(row_dict[ast])
        # col_inds[ind] = int(col_dict[(next_x_pos, next_w_pos, next_y_pos, next_z_pos)])
        data[ind] = 1
        # p_axr_xr_sparse[row_dict[ast],col_dict[(next_x_pos, next_w_pos, next_y_pos, next_z_pos)]] = 1
        if ind % 2000 == 0:
            print(ind, '/', len(action_state_tuples))

    p_axr_xr_dense = sp.sparse.csr_matrix((data, (row_inds, col_inds)), shape=(ns * na, ns))
    return p_axr_xr_dense


def build_cartesian_operator_sparse(nx, na, nis, Big_f, X, int_op_lst):
    num_int_ss = len(int_op_lst)
    ns = nx * nis**num_int_ss
    nsa = nx * nis**num_int_ss * na

    aranges_tuple = tuple([np.arange(na), np.arange(nx)] + ([np.arange(nis)] * num_int_ss))
    aranges_tuple_no_action = tuple([np.arange(nx)] + ([np.arange(nis)] * num_int_ss))
    action_and_state_tuples = list(itertools.product(*aranges_tuple)) # Action and State tuples
    # action_and_state_tuples = list(itertools.product(np.arange(na), np.arange(nx), np.arange(nis), np.arange(nis), np.arange(nis)))
    s_tuples = list(itertools.product(*aranges_tuple_no_action)) # State-only tuples
    row_dict = dict(zip(action_and_state_tuples, np.arange(len(action_and_state_tuples))))
    row_dict_ind_2_tup = dict(zip(np.arange(len(action_and_state_tuples)), action_and_state_tuples))
    col_dict = dict(zip(s_tuples, np.arange(len(s_tuples))))
    col_dict_ind_2_tup = dict(zip(np.arange(len(s_tuples)), s_tuples))
    # sp.sparse.csr_matrix
    # action_and_state_tuples = list(
    #     itertools.product(np.arange(na), np.arange(nx), np.arange(nis), np.arange(nis), np.arange(nis)))
    null_op = np.zeros(X.dense_op_list[0].shape)

    for a_ind in range(na):
        null_op[a_ind, np.arange(nx), np.arange(nx)] = 1

    large_negative_reward = -1000
    large_postive_reward = 1000
    ihdc_reward_function = np.zeros(len(row_dict))

    row_inds = np.zeros(ns * na, dtype='int')
    col_inds = np.zeros(ns * na, dtype='int')
    data = np.zeros(ns * na)
    for ind, ast in enumerate(action_and_state_tuples):
        if any(np.array(ast[2:5]) == 0):
            p_ax_x = null_op
            ihdc_reward_function[ind] = large_negative_reward
        elif any(np.array(ast[2:5]) == nis-1):
            ihdc_reward_function[ind] = large_postive_reward
            p_ax_x = X.dense_op_list[0]
        else:
            p_ax_x = X.dense_op_list[0]
        next_x_pos = np.where(p_ax_x[ast[0], ast[1], :])[0][0]
        g_var = np.where(Big_f[:, ast[0], ast[1], 0])[0][0]
        # g_var = g_var_true.copy()
        next_pos_list = [next_x_pos]
        for i, int_ss in enumerate(int_op_lst):
            if g_var != int_ss.action_type_ind:
                next_pos = int_ss.P_s_s_given_g_dict[0][ast[i + 2], :].nonzero()[0][0]   # Sets to the defaul goal-action when there is no match.
                next_pos_list.append(next_pos)
            else:
                next_pos = int_ss.P_s_s_given_g_dict[1][ast[i + 2], :].nonzero()[0][0]  # The 0 in .get(_, 0) is the default goal variable index, 0.
                next_pos_list.append(next_pos)
        next_pos_tuple = tuple(next_pos_list)
        col_inds[ind] = int(col_dict[next_pos_tuple])
        # next_w_pos = np.where(w_ss.P_a_s_s[w_ss.task_ind_2_goal.get(g_var, 0), ast[2], :])[0][0]
        # next_y_pos = np.where(y_ss.P_a_s_s[y_ss.task_ind_2_goal.get(g_var, 0), ast[3], :])[0][0]
        # next_z_pos = np.where(z_ss.P_a_s_s[z_ss.task_ind_2_goal.get(g_var, 0), ast[4], :])[0][0]
        # p_axr_xr[ast[0],ast[1],ast[2],ast[3],ast[4],next_w_pos,next_y_pos,next_z_pos]=1
        row_inds[ind] = int(row_dict[ast])
        # col_inds[ind] = int(col_dict[(next_x_pos, next_w_pos, next_y_pos, next_z_pos)])
        data[ind] = 1
        # p_axr_xr_sparse[row_dict[ast],col_dict[(next_x_pos, next_w_pos, next_y_pos, next_z_pos)]] = 1
        if ind % 50000 == 0:
            print(ind, '/', len(action_and_state_tuples))

    p_axr_xr_sparse = sp.sparse.csr_matrix((data, (row_inds, col_inds)), shape=(ns * na, ns))
    return p_axr_xr_sparse, action_and_state_tuples, ihdc_reward_function


def build_cartesian_operator_w_logic_sparse(nx, na, nis, Big_f, X, int_op_lst, bit_tensor):
    # Not implemented yet.
    num_int_ss = len(int_op_lst)
    ns = nx * nis**num_int_ss
    nsa = nx * nis**num_int_ss * na

    aranges_tuple = tuple([np.arange(na), np.arange(nx)] + ([np.arange(nis)] * num_int_ss))
    aranges_tuple_no_action = tuple([np.arange(nx)] + ([np.arange(nis)] * num_int_ss))
    action_and_state_tuples = list(itertools.product(*aranges_tuple)) # Action and State tuples
    # action_and_state_tuples = list(itertools.product(np.arange(na), np.arange(nx), np.arange(nis), np.arange(nis), np.arange(nis)))
    s_tuples = list(itertools.product(*aranges_tuple_no_action)) # State-only tuples
    row_dict = dict(zip(action_and_state_tuples, np.arange(len(action_and_state_tuples))))
    row_dict_ind_2_tup = dict(zip(np.arange(len(action_and_state_tuples)), action_and_state_tuples))
    col_dict = dict(zip(s_tuples, np.arange(len(s_tuples))))
    col_dict_ind_2_tup = dict(zip(np.arange(len(s_tuples)), s_tuples))
    # sp.sparse.csr_matrix
    # action_and_state_tuples = list(
    #     itertools.product(np.arange(na), np.arange(nx), np.arange(nis), np.arange(nis), np.arange(nis)))
    null_op = np.zeros(X.dense_op_list[0].shape)

    for a_ind in range(na):
        null_op[a_ind, np.arange(nx), np.arange(nx)] = 1

    large_negative_reward = -1000
    large_postive_reward = 1000
    ihdc_reward_function = np.zeros(len(row_dict))

    row_inds = np.zeros(ns * na, dtype='int')
    col_inds = np.zeros(ns * na, dtype='int')
    data = np.zeros(ns * na)
    for ind, ast in enumerate(action_and_state_tuples):
        if any(np.array(ast[2:5]) == 0):
            p_ax_x = null_op
            ihdc_reward_function[ind] = large_negative_reward
        elif any(np.array(ast[2:5]) == nis-1):
            ihdc_reward_function[ind] = large_postive_reward
            p_ax_x = X.dense_op_list[0]
        else:
            p_ax_x = X.dense_op_list[0]
        next_x_pos = np.where(p_ax_x[ast[0], ast[1], :])[0][0]
        g_var = np.where(Big_f[:, ast[0], ast[1], 0])[0][0]
        # g_var = g_var_true.copy()
        next_pos_list = [next_x_pos]
        for i, int_ss in enumerate(int_op_lst):
            if g_var != int_ss.action_type_ind:
                next_pos = int_ss.P_s_s_given_g_dict[0][ast[i + 2], :].nonzero()[0][0]   # Sets to the defaul goal-action when there is no match.
                next_pos_list.append(next_pos)
            else:
                next_pos = int_ss.P_s_s_given_g_dict[1][ast[i + 2], :].nonzero()[0][0]  # The 0 in .get(_, 0) is the default goal variable index, 0.
                next_pos_list.append(next_pos)
        next_pos_tuple = tuple(next_pos_list)
        col_inds[ind] = int(col_dict[next_pos_tuple])
        # next_w_pos = np.where(w_ss.P_a_s_s[w_ss.task_ind_2_goal.get(g_var, 0), ast[2], :])[0][0]
        # next_y_pos = np.where(y_ss.P_a_s_s[y_ss.task_ind_2_goal.get(g_var, 0), ast[3], :])[0][0]
        # next_z_pos = np.where(z_ss.P_a_s_s[z_ss.task_ind_2_goal.get(g_var, 0), ast[4], :])[0][0]
        # p_axr_xr[ast[0],ast[1],ast[2],ast[3],ast[4],next_w_pos,next_y_pos,next_z_pos]=1
        row_inds[ind] = int(row_dict[ast])
        # col_inds[ind] = int(col_dict[(next_x_pos, next_w_pos, next_y_pos, next_z_pos)])
        data[ind] = 1
        # p_axr_xr_sparse[row_dict[ast],col_dict[(next_x_pos, next_w_pos, next_y_pos, next_z_pos)]] = 1
        if ind % 50000 == 0:
            print(ind, '/', len(action_and_state_tuples))

    p_axr_xr_sparse = sp.sparse.csr_matrix((data, (row_inds, col_inds)), shape=(ns * na, ns))
    return p_axr_xr_sparse, action_and_state_tuples, ihdc_reward_function

def build_cartesian_operator_wbinary_sparse(nx, na, nis, nb, Big_f, X, int_op_lst, bin_op, goal_state_to_ho_action):
    num_int_ss = len(int_op_lst)
    ns = nx * nis**num_int_ss * nb
    nsa = ns * na

    aranges_tuple = tuple([np.arange(na), np.arange(nx)] + ([np.arange(nis)] * num_int_ss) + (np.arange(nb)))
    aranges_tuple_no_action = tuple([np.arange(nx)] + ([np.arange(nis)] * num_int_ss))
    action_and_state_tuples = list(itertools.product(*aranges_tuple)) # Action and State tuples
    # action_and_state_tuples = list(itertools.product(np.arange(na), np.arange(nx), np.arange(nis), np.arange(nis), np.arange(nis)))
    s_tuples = list(itertools.product(*aranges_tuple_no_action)) # State-only tuples
    row_dict = dict(zip(action_and_state_tuples, np.arange(len(action_and_state_tuples))))
    row_dict_ind_2_tup = dict(zip(np.arange(len(action_and_state_tuples)), action_and_state_tuples))
    col_dict = dict(zip(s_tuples, np.arange(len(s_tuples))))
    col_dict_ind_2_tup = dict(zip(np.arange(len(s_tuples)), s_tuples))
    # sp.sparse.csr_matrix
    # action_and_state_tuples = list(
    #     itertools.product(np.arange(na), np.arange(nx), np.arange(nis), np.arange(nis), np.arange(nis)))
    null_op = np.zeros(X.dense_op_list[0].shape)

    for a_ind in range(na):
        null_op[a_ind, np.arange(nx), np.arange(nx)] = 1

    large_negative_reward = -1000
    large_postive_reward = 1000
    ihdc_reward_function = np.zeros(len(row_dict))

    row_inds = np.zeros(ns * na, dtype='int')
    col_inds = np.zeros(ns * na, dtype='int')
    data = np.zeros(ns * na)
    for ind, ast in enumerate(action_and_state_tuples):
        if any(np.array(ast[2:5]) == 0):
            p_ax_x = null_op
            ihdc_reward_function[ind] = large_negative_reward
        elif any(np.array(ast[2:5]) == nis-1):
            ihdc_reward_function[ind] = large_postive_reward
            p_ax_x = X.dense_op_list[0]
        else:
            p_ax_x = X.dense_op_list[0]
        next_x_pos = np.where(p_ax_x[ast[0], ast[1], :])[0][0]
        g_var = np.where(Big_f[:, ast[0], ast[1], 0])[0][0]
        # g_var = g_var_true.copy()
        next_pos_list = [next_x_pos]
        for i, int_ss in enumerate(int_op_lst):
            if g_var != int_ss.action_type_ind:
                next_pos = int_ss.P_s_s_given_g_dict[0][ast[i + 2], :].nonzero()[0][0]   # Sets to the defaul goal-action when there is no match.
                next_pos_list.append(next_pos)
            else:
                next_pos = int_ss.P_s_s_given_g_dict[1][ast[i + 2], :].nonzero()[0][0]  # The 0 in .get(_, 0) is the default goal variable index, 0.
                next_pos_list.append(next_pos)
        next_pos_tuple = tuple(next_pos_list)
        col_inds[ind] = int(col_dict[next_pos_tuple])
        # next_w_pos = np.where(w_ss.P_a_s_s[w_ss.task_ind_2_goal.get(g_var, 0), ast[2], :])[0][0]
        # next_y_pos = np.where(y_ss.P_a_s_s[y_ss.task_ind_2_goal.get(g_var, 0), ast[3], :])[0][0]
        # next_z_pos = np.where(z_ss.P_a_s_s[z_ss.task_ind_2_goal.get(g_var, 0), ast[4], :])[0][0]
        # p_axr_xr[ast[0],ast[1],ast[2],ast[3],ast[4],next_w_pos,next_y_pos,next_z_pos]=1
        row_inds[ind] = int(row_dict[ast])
        # col_inds[ind] = int(col_dict[(next_x_pos, next_w_pos, next_y_pos, next_z_pos)])
        data[ind] = 1
        # p_axr_xr_sparse[row_dict[ast],col_dict[(next_x_pos, next_w_pos, next_y_pos, next_z_pos)]] = 1
        if ind % 50000 == 0:
            print(ind, '/', len(action_and_state_tuples))

    p_axr_xr_sparse = sp.sparse.csr_matrix((data, (row_inds, col_inds)), shape=(ns * na, ns))
    return p_axr_xr_sparse, action_and_state_tuples, ihdc_reward_function

def build_cartesian_operator_sparse_v2_action_last(ns, nx, na, nis, f, X, int_op_lst, goal_state_to_ho_action):
    shape = (nx, nis, nis, nis, na)
    coords = []
    # coords = np.vstack([np.zeros(0, dim, size=100) for dim in shape])
    data = np.random.rand(100)
    # sparse_tensor = sparse.COO(coords, data, shape=shape)

    as_tuples = list(
        itertools.product(np.arange(nx), np.arange(nis), np.arange(nis), np.arange(nis), np.arange(na)))
    s_tuples = list(itertools.product(np.arange(nx), np.arange(nis), np.arange(nis), np.arange(nis)))
    row_dict = dict(zip(as_tuples, np.arange(len(as_tuples))))
    row_dict_ind_2_tup = dict(zip(np.arange(len(as_tuples)), as_tuples))
    col_dict = dict(zip(s_tuples, np.arange(len(s_tuples))))
    col_dict_ind_2_tup = dict(zip(np.arange(len(s_tuples)), s_tuples))
    # sp.sparse.csr_matrix
    action_state_tuples = list(
        itertools.product(np.arange(nx), np.arange(nis), np.arange(nis), np.arange(nis), np.arange(na)))
    null_op = np.zeros(X.dense_op_list[0].shape)

    for a_ind in range(na):
        null_op[a_ind, np.arange(nx), np.arange(nx)] = 1

    large_negative_reward = -1000
    large_postive_reward = 1000
    ihdc_reward_function = np.zeros(len(row_dict))

    row_inds = np.zeros(ns * na, dtype='int')
    col_inds = np.zeros(ns * na, dtype='int')
    data = np.zeros(ns * na)
    for ind, ast in enumerate(action_state_tuples):
        if any(np.array(ast[1:-1]) == 0): # check if dead in any physiological state
            p_ax_x = null_op
            ihdc_reward_function[ind] = large_negative_reward
        elif any(np.array(ast[1:-1]) == nis - 1):
            ihdc_reward_function[ind] = large_postive_reward
            p_ax_x = X.dense_op_list[0]
        else:
            p_ax_x = X.dense_op_list[0]
        next_x_pos = np.where(p_ax_x[ast[0], ast[1], :])[0][0]
        g_var = np.where(f[:, ast[-1], ast[0], 0])[0][0]
        # g_var = g_var_true.copy()
        next_pos_list = [next_x_pos]
        for i, int_ss in enumerate(int_op_lst):
            if g_var != int_ss.action_type_ind:
                next_pos = np.where(int_ss.P_a_s_s[0, ast[i + 2], :])[0][
                    0]  # Sets to the defaul goal-action when there is no match.
                next_pos_list.append(next_pos)
            else:
                next_pos = np.where(int_ss.P_a_s_s[1, ast[i + 2], :])[0][
                    0]  # The 0 in .get(_, 0) is the default goal variable index, 0.
                next_pos_list.append(next_pos)
        next_pos_tuple = tuple(next_pos_list)
        col_inds[ind] = int(col_dict[next_pos_tuple])
        # next_w_pos = np.where(w_ss.P_a_s_s[w_ss.task_ind_2_goal.get(g_var, 0), ast[2], :])[0][0]
        # next_y_pos = np.where(y_ss.P_a_s_s[y_ss.task_ind_2_goal.get(g_var, 0), ast[3], :])[0][0]
        # next_z_pos = np.where(z_ss.P_a_s_s[z_ss.task_ind_2_goal.get(g_var, 0), ast[4], :])[0][0]
        # p_axr_xr[ast[0],ast[1],ast[2],ast[3],ast[4],next_w_pos,next_y_pos,next_z_pos]=1
        row_inds[ind] = int(row_dict[ast])
        # col_inds[ind] = int(col_dict[(next_x_pos, next_w_pos, next_y_pos, next_z_pos)])
        data[ind] = 1
        # p_axr_xr_sparse[row_dict[ast],col_dict[(next_x_pos, next_w_pos, next_y_pos, next_z_pos)]] = 1
        if ind % 2000 == 0:
            print(ind, '/', len(action_state_tuples))

    p_axr_xr_sparse = sp.sparse.csr_matrix((data, (row_inds, col_inds)), shape=(ns * na, ns))
    return p_axr_xr_sparse, action_state_tuples, ihdc_reward_function

    # save_as_pickled_object(p_axr_xr_sparse, 'p_axr_xr_sparse.obj')

def get_next_state_tup(ind, p_axr_xr_sparse):
    p_axr_xr_sparse[ind, :]

def onehot(inds, size):
    oh = np.zeros(size)
    oh[inds] = 1
    return oh

class SPA_Object:
    def __init__(self, init_x, init_t, init_x_vec, initial_secondary_states, initial_secondary_state_vecs, eta_ens, pi_ens, kappa_ens, operators, omegas, init_plan):
        self.x = init_x
        self.t = init_t
        self.x_vec = init_x_vec
        self.initial_secondary_states = initial_secondary_states
        self.initial_secondary_state_vecs = initial_secondary_state_vecs
        self.eta_ens = eta_ens
        self.pi_ens = pi_ens
        self.kappa_ens = kappa_ens
        self.operators = operators
        self.omegas = omegas
        self.plan = init_plan

    # def advance_states(self, initial_state_vecs):
    #     x_new =
    #     for i, sv in enumerate(initial_secondary_state_vecs):
    #         sv_new = np.dot(sv, self.operators[i])


class Agent:
    def __init__(self, x_ss, solutions, state_list, ss_list, transition_kernels, affordance_dict, affordance_bool, meta_pol, goal_states, bool_goalstates, constraint_states, goalstate_to_picture_name, boolean_states_2_type_ind):
        self.goal_states = goal_states
        self.bool_goalstates = bool_goalstates
        self.constraint_states = constraint_states
        self.x_ss = x_ss  # Optional if non_stationary
        self.solutions = solutions  # Solutions contains kappa, pi, eta_pos, eta_neg, and beta
        self.state_list = state_list
        self.ss_list = ss_list
        self.ss_list_internal = ss_list[1:]
        self.transition_kernels = transition_kernels
        self.affordance_dict = affordance_dict
        self.affordance_bool = affordance_bool
        self.meta_pol = meta_pol
        self.current_pol_idx = 0
        self.cur_env = 0
        self.goal_progress = []
        self.boolean_states_2_type_ind = boolean_states_2_type_ind
        # self.goal_state_to_color_name = goal_state_to_color_name
        self.goalstate_to_picture_name = goalstate_to_picture_name
    # def advance_states(self, pol):

    def control_flow_single_step(self):
        if self.current_pol_idx >= len(self.goal_states[self.meta_pol]):
            self.current_pol_idx = len(self.meta_pol)-1
            # return
        x_state = self.ss_list[0].current_state
        print(x_state)
        print("meta_pol_index: ", self.current_pol_idx)
        cur_goal_state = self.goal_states[self.meta_pol][self.current_pol_idx]
        pi = self.solutions[(self.cur_env, cur_goal_state)]['pi']
        action = pi[x_state]
        Px = self.x_ss.dense_op_list[self.cur_env]
        self.ss_list[0].current_state = np.random.choice(len(Px[action, x_state, :]), p=Px[action, x_state, :])
        # new_state_list = [next_state]

        for i, ss_k in enumerate(self.ss_list[1:-1]):
            type_ind = ss_k.action_type_ind
            if type_ind not in self.affordance_dict:  # Skip updates if we don't have this type_ind
                continue
            P_k = ss_k.P_a_s_s
            z_state = self.ss_list[i + 1].current_state
            F = self.affordance_dict[type_ind]
            alpha = np.random.choice(len(F[:, action, x_state]), p=F[:, action, x_state])
            next_hl_state = np.random.choice(len(P_k[alpha, z_state, :]), p=P_k[alpha, z_state, :])
            self.ss_list[i + 1].current_state = next_hl_state

        # Update Boolean Task state (b_state)
        P_k = self.ss_list[-1].P_a_s_s.todense()
        b_state = self.ss_list[-1].current_state
        F = self.affordance_bool
        alpha = np.random.choice(len(F[:, action, x_state]), p=F[:, action, x_state])
        next_hl_state = np.random.choice(len(P_k[alpha, b_state, :]), p=P_k[alpha, b_state, :])
        self.ss_list[-1].current_state = next_hl_state

        termination_func = self.solutions[(self.cur_env, cur_goal_state)]['beta']
        termination_bool = np.random.binomial(1, termination_func[action, x_state])
        if termination_bool:
            self.current_pol_idx = self.current_pol_idx + 1
            print("updated_meta_pol_index: ", self.current_pol_idx)
            if x_state == 0:
                self.cur_env = 1

class TGMDP:
    def __init__(self, p_axt_x, p_ax_x, fg, init_state, T_f, is_non_stationary=True):
        self.p_axt_x = p_axt_x  # Optional if non_stationary
        self.p_ax_x = p_ax_x
        self.is_non_stationary = is_non_stationary
        self.fg = fg
        self.T_f = T_f
        self.init_state = init_state

    def compute_solution_dense(self):
        if self.is_non_stationary == True:
            self.kappa, self.pi, self.eta = feasibility_iteration_flat_dense(self.p_axt_x, self.fg)
        else:
            self.kappa, self.pi, self.eta = feasibility_iteration_flat_dense(self.p_axt_x, self.fg, p_ax_x=self.p_ax_x)

class HTGMDP:
    def __init__(self, state_space_dict, fg, omega_dict, init_state_vector_dict, goalstates, f, zeta, T_f, nx):
        self.nx = nx
        self.goalstates = goalstates
        self.T_f = T_f
        self.state_space_dict = state_space_dict  # Maps names to state-spaces {'X': x_space, 'W': w_space, 'Y': y_space, ...}
        self.state_space_names = self.state_space_dict.keys()
        self.X = self.state_space_dict['X']
        self.O_set  # Low-level Operator Dict {'X': P_x, 'W': P_w, 'Y': P_y, 'Z': P_z,...}
        self.fg = fg
        self.omega_dict = omega_dict  # {'W': P_w, 'Y': P_y, 'Z': P_z,...}
        self.time_vec = onehot(0, T_f + 1)
        self.state_vector_dict = init_state_vector_dict  # Has the form {'X': x, 'Y': y, 'Z': z,...} where the trailing states are non-X states, and all O_set and omegas must match
        self.zeta = zeta
        self.kappa_ens = np.zeros([len(self.goalstates), nx, T_f])
        self.pi_ens = np.zeros([len(self.goalstates), nx, T_f]).astype('int')
        self.eta_ens = np.zeros([len(self.goalstates), nx, T_f, nx, T_f])

        for g in range(len(self.goalstates)):
            fg_axt = restrict_to_state(f, self.goalstates[g], g + 1)
            # fg_axt = np.swapaxes(fg_xat, 0, 1)
            kappa, pi, eta = feasibility_iteration_flat_dense(p_axt_x, fg_axt, self.X.P_ax_x, False)
            # kappa_s, pi_s, eta_s = feasibility_iteration_flat_sparse(p_axt_x_sparse, fg_axt, X.P_ax_x, False)
            self.kappa_ens[g, :, :] = kappa
            self.pi_ens[g, :, :] = pi.astype('int')
            self.eta_ens[g, :, :, goalstates[g], :] = eta


    def get_time_restriction_deterministic(self, secondary_state_dict):
        #  Returns the minimum time before a mode change occurs over all other state-spaces. Assumes determinism
        min_time = np.inf
        for ss in self.state_space_dict:
            if ss.name != 'X':
                state = secondary_state_dict[ss.name]
                ttg = ss.time_to_mode_switch(state)
                min_time = np.min(min_time, ttg)

        return min_time


    def goal_restriction(self, p_ax_x, xa_list):
        # returns a restriction res_p_axt restricted to a particular mode by multiplying the original operator by a mask
        # ho_zeta maps ho_e -> xa_list
        # xa_list = ((x,a)_0,(x,a)_1,(x,a)_2,...)

        mask = self.ones(p_ax_x.shape())
        mask[xa_list,:] = 0
        res_p_ax_x = np.multiply(p_ax_x, mask)
        return res_p_ax_x


    def advance_state_vector(self, pi_ind):
        for ss in self.state_space_name_dict:
            self.state_vector_dict[ss] = self.state_vector_dict[ss].dot(self.state_vector_dict[ss])


def value_iteration_ihdc(P_as_s, reward_func_as, gamma, ns, na):
    # This algorithm assumes that actions "a" are the "inner" variable in the row space. e.g. row_i = (w,x,y,z,...,a)_i.
    # IDHC = Infinite Horizon Discounted Control
    # reward_func_sa is a state reward function (no action).  For tLMDP, s = (sigma, x)
    P_as_s = sp.sparse.csr_matrix(P_as_s)
    ns = P_as_s.shape[1]
    v = sp.sparse.csr_matrix(np.zeros(ns))
    v = v.reshape([ns, 1])
    max_steps = 10000
    for t in range(max_steps):
        # expectation_v_prime = T_as_s.dot(v)
        # v = v.reshape([1, ns])
        expectation_v_prime = sp.sparse.csr_matrix.dot(P_as_s, v)
        v_sxa_new = reward_func_as + gamma * expectation_v_prime.reshape([na, ns])
        v_new = v_sxa_new.max(axis=0).transpose()
        delta = np.sum(np.abs(v - v_new))
        # print(delta)
        v = v_new
        if delta < 0.0001:
            print("Total Iterations (Value Iter): ", t)
            break

    expectation_v_prime = sp.sparse.csr_matrix.dot(P_as_s, v)
    v_sxa_new = reward_func_as + gamma * expectation_v_prime.reshape([na, ns])
    pi = v_sxa_new.argmax(axis=0)

    # Plot Value Function
    # print("VI Policy Time Internal: ", VI_compute_time)

    # Uncomment for value function plotting:
    # Assumes v is already defined in your code and is a vector of length 49.
    # For a 7x7 grid:
    nrows, ncols = config.ROW_COUNT, config.COLUMN_COUNT
    nx = nrows * ncols

    # vrs = v.reshape([nx, ns//nx])
    # v_slice = vrs[:, -30]  # For visualizing projections onto X, as a 7x7 main_files.

    # if v_slice.size != nrows * ncols:
    #     raise ValueError("The size of v does not match the 7x7 grid (should be 49 elements).")

    # Reshape the value function into a 7x7 grid.
    # V_grid = v.reshape(nrows, ncols)
    #
    # # -----------------------------
    # # Option 1: 2D Mesh Plot using pcolormesh
    # # -----------------------------
    # # Define grid edges for pcolormesh.
    # x_edges = np.arange(ncols + 1)
    # y_edges = np.arange(nrows + 1)
    # X_edges, Y_edges = np.meshgrid(x_edges, y_edges)
    #
    # plt.figure(figsize=(8, 6))
    # mesh = plt.pcolormesh(X_edges, Y_edges, V_grid, shading='auto', cmap='viridis')
    # plt.colorbar(mesh, label='Value')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title("2D Mesh Plot of the Value Function")
    # plt.show()
    #
    # # -----------------------------
    # # Option 2: 3D Surface Plot using plot_surface
    # # -----------------------------
    # # Create grid centers for the 3D plot.
    # x = np.arange(ncols)
    # y = np.arange(nrows)
    # X, Y = np.meshgrid(x, y)
    #
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # surf = ax.plot_surface(X, Y, V_grid, cmap='viridis', edgecolor='none')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Value')
    # ax.set_title("3D Surface Plot of the Value Function")
    # ax.view_init(elev=45, azim=40)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.show()

    return v, pi


def value_iteration_ihdc_tensor(T_sa_s, reward_func_sa, gamma, ns, na):
    # This algorithm assumes that actions "a" are the "inner" variable in the row space. e.g. row_i = (w,x,y,z,...,a)_i.
    # IDHC = Infinite Horizon Discounted Control
    # reward_func_sa is a state reward function (no action).  For tLMDP, s = (sigma, x)
    max_steps = ns + 10
    reward_vec = reward_func_sa.flatten()
    v = np.zeros(ns)
    # while True:
    state_inds_action_0 = [i * na for i in range(ns)]
    delta = 10000
    # start_time = timer()
    for t in range(max_steps):
        # expectation_v_prime = T_as_s.dot(v)
        expectation_v_prime = sp.sparse.csr_matrix.dot(T_sa_s, v)
        v_sxa_new = reward_vec + gamma * expectation_v_prime
        v_sxa_new_rs = v_sxa_new.reshape(ns, na)
        v_sxa_argmax_inds = v_sxa_new_rs.argmax(axis = 1)
        v_new = v_sxa_new[state_inds_action_0+v_sxa_argmax_inds]
        delta = np.sum(np.abs(v - v_new))
        # print(delta)
        v = v_new
        print('VI-step: ',t,'/',max_steps)
        # print(max_steps)
        print('delta: ',delta)
        if delta < 0.01:
            print("Total Iterations (Value Iter): ", t)
            break

    # end_time = timer()
    # VI_compute_time = end_time-start_time
    expectation_v_prime = sp.sparse.csr_matrix.dot(T_sa_s, v)
    v_sxa_new = reward_vec + gamma * expectation_v_prime
    v_sxa_new_rs = v_sxa_new.reshape(ns, na)
    pi = v_sxa_new_rs.argmax(axis=1)
    # pi = np.argmax(np.tensordot(T_as_s,v),axis=0)

    # print("VI Policy Time Internal: ", VI_compute_time)
    return v, pi


def compute_TEF_given_joint_dist(state_time_kappa_comp_list, p_s, env, pi_ind, T_f):
    # TODO: Verify this function works and sums to one over future time given the joint distribution p_s.
    # p_s = nd_tensor, distribution over the joint product space.
    # This function outputs \xi_{p(s)}(t_f) = sum_{s}p(s)\xi(t_f|s) which is an expected TEF given a state-vector joint distribution p(s) = p(x,w,...,z).
    # state_time_kappa_list is the list of CEF functions for n state-spaces
    # state_time_kappa_comp_list is a list of kappa_compliments \bar{\kappa} for each state-space

    # marginal_list is a list of all marginals where we sum over all axes except dimension k for all dimensions k
    # Each marginal's index in the list must correspond to the same index in kappa_list
    p_s_dok = sparse.DOK(p_s.shape)
    new_shape = p_s.shape + (T_f,)
    xi_temporal = sparse.DOK(new_shape, dtype=p_s.dtype)

    # kappa_compliments_conditioned_on_state = [kappa_comp[] for kappa_comp in range(1, p_s.ndim)]
    coords = p_s.coords
    data = p_s.data

    # First get kappa_comp for the BL + env + pi_ind
    kappa_comp_BL = state_time_kappa_comp_list[0][env][pi_ind]
    for i in range(len(data)):
        coordinate = coords[:, i]  # Get the 5D coordinate
        # value = data[i]  # Get the corresponding value
        product_kappa_comp = kappa_comp_BL[coordinate[0]]
        for j in range(1, len(state_time_kappa_comp_list)):
            product_kappa_comp = product_kappa_comp * state_time_kappa_comp_list[j][coordinate[j]]
        product_kappa_comp = np.concatenate([[1], product_kappa_comp]) # Concatening a 1 at the end allows us to represent kappa(x,t=-1) defined to compute xi.
        xi_temporal[tuple(coordinate) + (slice(None),)] = np.diff(1 - product_kappa_comp)  # setting xi(t_f|s)

    # time_dim_size = state_time_kappa_comp_list[0][-1]  # or whatever length your time distribution is
    # output_shape = p_s_dok.shape + (time_dim_size,)
    # output_dok = sparse.DOK(output_shape, dtype=np.float32)
    # # for index, value in p_s_dok.items():
    #     # index is a tuple like (s1, s2, s3, s4)
    #     # value is p_s_dok[index]
    #
    # # total_sum = np.sum(xi)
    # if not np.isclose(xi_temporal.sum(), 1.0, atol=1e-6):
    #     raise warnings.warn(f"Error: xi_joint sums to {total_sum}, but it should sum to 1.")
    return sparse.COO(xi_temporal)



def rearrange_dimensions(P_composite, num_current_states, num_next_states, num_actions):
    """
    Rearrange the dimensions of the composite tensor.

    Parameters:
      P_composite : sparse.COO
          The composite tensor with shape (na, nx, ny, nw, nz, nsig,..., nx', ny', nw', nz', nsig',...).
      num_current_states : int
          Number of current state dimensions.
      num_next_states : int
          Number of next state dimensions.
      num_actions : int
          Number of action dimensions.

    Returns:
      sparse.COO with rearranged dimensions: (na, nx, ny, nw, nz, ... , nx', ny', nw', nz', ...).
    """
    # Determine the total number of dimensions.
    total_dims = P_composite.ndim

    # Create the new order of dimensions.
    # Actions come first, followed by current states, and then next states.
    new_order = [0] + list(1+2*i for i in range(num_current_states)) + list(2+2*i for i in range(num_current_states))
    # Rearrange the dimensions.
    P_rearranged = sparse.COO.transpose(P_composite, axes=new_order)

    return P_rearranged



def compose_transition_kernels(Px, affordance_list, HL_kernel_list):
    """
    Compose the base-level transition kernel Px with high-level transition kernels
    through the affordance list, ensuring dimensions match and are grouped appropriately.

    Parameters:
      Px : sparse.COO
          Base-level transition kernel with shape (nx, nx, na).
      affordance_list : list of sparse.COO
          List of affordance tensors, e.g., [F_xy, F_xw, F_xz, F_xsig],
          where each F has shape (nalpha, nx, na).
      HL_kernel_list : list of sparse.COO
          List of high-level transition kernels, e.g., [Py, Pw, Pz, Psig],
          where each P has shape (nalpha, n_state, n_state).

    Returns:
      sparse.COO
          The composite transition kernel with dimensions grouped as:
          (na, nx, ny, nw, nz, nsig, nx', ny', nw', nz', nsig').
    """
    # Step 0: Ensure the lists are of the same size
    assert len(affordance_list) == len(HL_kernel_list), \
        "affordance_dict and HL_kernel_list must be of the same length."

    # Step 1: Validate dimensions
    for F, P_hl in zip(affordance_list, HL_kernel_list):
        assert F.shape[0] == P_hl.shape[0], "Mismatch in nalpha dimension between F and P_hl."
        assert F.shape[1] == Px.shape[0], "Mismatch in na dimension between F and Px."
        assert F.shape[2] == Px.shape[1], "Mismatch in nx dimension between F and Px."

    # Step 2: Compose Px with affordance tensors
    PF = utilities.combine_F_and_Px(affordance_list, Px)

    # Step 3: Compose with HL transition kernels
    P_composite = PF
    for i, P_hl in enumerate(HL_kernel_list):
        # Use sparse.tensordot to contract over the alpha dimension
        P_composite = sparse.tensordot(P_composite, P_hl, axes=(0, 0))

    # Step 4: Rearrange dimensions
    # Determine the number of current and next states
    num_current_states = len(affordance_list) + 1  # +1 for x
    num_next_states = num_current_states
    num_actions = Px.shape[0]

    # Rearrange dimensions to group actions first, then current states, then next states
    P_rearranged = rearrange_dimensions(P_composite, num_current_states, num_next_states, num_actions)

    return P_rearranged


def compute_solution_sets(Px, affordance_list, HL_kernel_list, goalstates, constraint_states, wallmatlist, f_c, xspace, T_f):
    na, nx = Px.shape[0:2]
    kappa_dict = []
    pi_dict = []
    eigen_decomp_env_dict = dict()
    kappa_dict_env = dict()
    eta_pos_dict_env = dict()
    eta_neg_dict_env = dict()
    eta_dict_env = dict()
    pi_dict_env = dict()
    compute_eigs = True
    for env_ind in range(len(wallmatlist)):
        kappa_dict = dict()
        eta_pos_dict = dict()
        eta_neg_dict = dict()
        eta_dict = dict()
        pi_dict = dict()
        eig_decomp_dict = dict()
        for i, g in enumerate(np.sort(list(goalstates))):
            f_g = np.zeros([na, nx])
            f_g[0, g] = 1
            Px = xspace.dense_op_list[env_ind]
            kappa, pi, eta_pos, eta_neg = feas_iter_stationary_flat_dense(Px, f_g, f_c, max_time=T_f,
                                                                          wallmat=wallmatlist[env_ind],
                                                                          use_ET=True)
            kappa_dict[g] = kappa
            pi_dict[g] = pi
            eta_pos_dict[g] = eta_pos
            eta_neg_dict[g] = eta_neg
            eta_dict[g] = eta_pos + eta_neg
            if compute_eigs:
                p_x_x = xspace.dense_op_list[env_ind][pi, range(nx), :]
                p_x_x[g, :] = 0
                p_x_x[g, g] = 1
                if constraint_states:
                    for cs in constraint_states:
                        p_x_x[cs, :] = 0
                        p_x_x[cs, cs] = 1
                evals, Q = np.linalg.eig(p_x_x)
                # D = np.diag(evals)
                # eig_decomp_dict[g] = (evals, Q, np.linalg.inv(Q))
        eigen_decomp_env_dict[env_ind] = eig_decomp_dict
        eta_pos_dict_env[env_ind] = eta_pos_dict
        eta_neg_dict_env[env_ind] = eta_neg_dict
        eta_dict_env[env_ind] = eta_dict
        pi_dict_env[env_ind] = pi_dict
        kappa_dict_env[env_ind] = kappa_dict

    return kappa_dict_env, pi_dict_env, eta_pos_dict_env, eta_neg_dict_env, eta_dict_env


def create_affordance_function(ntypes, nx, na, goalstates_2_type_ind, goalstates):
    big_F = np.zeros([ntypes, na, nx])
    for g_ind, xg in enumerate(goalstates):
        big_F[goalstates_2_type_ind[xg] + 1, 0, xg] = 1
    big_F[0, :, :] = 1 - big_F[1:, :, :].sum(axis=(0))
    if np.all(big_F.sum(axis=0)) != True:
        warnings.warn('Operator Does not Sum to One')
    return big_F


def create_binned_eta(eta1, indices, n_bins):
    """
    Create 4D tensor by placing eta1 values at specific return bin indices

    Parameters:
    - eta1: shape (81, 81, 36) - your original tensor
    - indices: array of bin indices where to place values
    - n_bins: total number of return bins (len(bins))

    Returns:
    - eta1R: shape (81, 81, 36, n_bins)
    """
    # Initialize the 4D tensor with zeros
    eta1R = np.zeros((*eta1.shape, n_bins))

    # Place eta1 values at the specified bin indices
    # This assumes you want to broadcast eta1 to all specified indices
    for i, bin_idx in enumerate(indices):
        eta1R[:, :, :, bin_idx] += eta1  # Use += in case multiple indices point to same bin

    return eta1R
