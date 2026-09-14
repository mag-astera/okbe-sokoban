from config import *
# torch deferred into convert_dense_to_sparse_torch (saves ~1.2s at import)
import numpy as np
import sys
import matplotlib.pyplot as plt
from itertools import chain, combinations
from scipy.sparse import csr_matrix as csr
import signal, sys, os
import scipy as sp
from timeit import default_timer as timer
from salmdp import SALMDP
import itertools
import functools
import pickle
import matplotlib.colors as mcolors
import math
from scipy.sparse import csr_matrix, csc_matrix, diags, coo_matrix, hstack
import string
import sparse
import io
from PIL import Image

from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation
from matplotlib import animation



def value_iteration_ihdc_tensor(p_axr_xr, reward_func_sa, gamma, ns, na):
    # IDHC = Infinite Horizon Discounted Control
    # reward_func_sa is a state reward function (no action).  For tLMDP, s = (sigma, x)
    max_steps = ns * 2
    v = np.zeros(ns)
    # while True:
    state_inds_action_0 = [i * na for i in range(ns)]
    delta = 10000
    start_time = timer()
    for t in range(max_steps):
        expectation_v_prime = np.tensordot(p_axr_xr, v)
        v_sxa_new = reward_func_sa + gamma * expectation_v_prime
        v_sxa_new_rs = v_sxa_new.reshape(ns, na)
        v_sxa_argmax_inds = v_sxa_new_rs.argmax(axis = 1)
        v_new = v_sxa_new[state_inds_action_0+v_sxa_argmax_inds]
        delta = np.sum(np.abs(v - v_new))
        # print(delta)
        v = v_new
        # print('VI-step: ',t)
        # print(max_steps)
        # print(delta)
        if delta < 0.001:
            print("Total Iterations (Value Iter): ", t)
            break

    end_time = timer()
    VI_compute_time = end_time-start_time
    pi = np.argmax(np.tensordot(T_as_s,v),axis=0)

    print("VI Policy Time Internal: ", VI_compute_time)
    return v, pi

def value_iteration_ihdc(T_sa_s, reward_func_sa, gamma, ns, na):
    # This algorithm assumes that actions "a" are the "inner" variable in the row space. e.g. row_i = (w,x,y,z,...,a)_i.
    # IDHC = Infinite Horizon Discounted Control
    # reward_func_sa is a state reward function (no action).  For tLMDP, s = (sigma, x)
    max_steps = ns + 10
    reward_vec = reward_func_sa.flatten()
    v = np.zeros(ns)
    # while True:
    state_inds_action_0 = [i * na for i in range(ns)]
    delta = 10000
    start_time = timer()
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
        print('delta: ', delta)
        if delta < 0.01:
            print("Total Iterations (Value Iter): ", t)
            break

    end_time = timer()
    VI_compute_time = end_time-start_time
    expectation_v_prime = sp.sparse.csr_matrix.dot(T_sa_s, v)
    v_sxa_new = reward_vec + gamma * expectation_v_prime
    v_sxa_new_rs = v_sxa_new.reshape(ns, na)
    pi = v_sxa_new_rs.argmax(axis=1)
    # pi = np.argmax(np.tensordot(T_as_s,v),axis=0)

    print("VI Policy Time Internal: ", VI_compute_time)
    return v, pi

def value_iteration_ihdc_outer_action(T_as_s, reward_func_sa, gamma, ns, na):
    # TODO: This algorithm assumes that actions "a" are the "outer" variable in the row space. e.g. row_i = (a,...,w,x,y,z)_i.
    # IDHC = Infinite Horizon Discounted Control
    # reward_func_sa is a state reward function (no action).  For tLMDP, s = (sigma, x)
    max_steps = ns + 10
    reward_vec = reward_func_sa.flatten()
    v = np.zeros(ns)
    # while True:
    state_inds = np.arange(ns)
    delta = 10000
    start_time = timer()
    for t in range(max_steps):
        # expectation_v_prime = T_as_s.dot(v)
        expectation_v_prime = sp.sparse.csr_matrix.dot(T_as_s, v)
        v_sxa_new = reward_vec + gamma * expectation_v_prime
        v_sxa_new_rs = v_sxa_new.reshape(na, ns)
        v_sxa_argmax_inds = v_sxa_new_rs.argmax(axis = 0)
        v_new = v_sxa_new[state_inds + (v_sxa_argmax_inds * ns)]
        delta = np.sum(np.abs(v - v_new))
        # print(delta)
        v = v_new
        print('VI-step: ',t,'/',max_steps)
        # print(max_steps)
        print('delta: ',delta)
        if delta < 0.1:
            print("Total Iterations (Value Iter): ", t)
            break

    end_time = timer()
    VI_compute_time = end_time-start_time
    expectation_v_prime = sp.sparse.csr_matrix.dot(T_as_s, v)
    v_sxa_new = reward_vec + gamma * expectation_v_prime
    v_sxa_new_rs = v_sxa_new.reshape(ns, na)
    pi = v_sxa_new_rs.argmax(axis=1)
    # pi = np.argmax(np.tensordot(T_as_s,v),axis=0)

    print("VI Policy Time Internal: ", VI_compute_time)
    return v, pi


def value_iteration(T_s_sa, reward_func_sa, v_init, terminal_rewards, ns, na):
    # reward_func_sa is a state reward function (no action).  For tLMDP, s = (sigma, x)
    max_steps = T_s_sa.shape[1] + 10
    v = v_init
    terminal_inds = terminal_rewards.nonzero()[0]
    # while True:
    state_inds_action_0 = [i * na for i in range(ns)]
    delta = 10000
    start_time = timer()
    for t in range(max_steps):
        expectation_v_prime = T_s_sa.dot(v)
        # expectation_v_prime[terminal_inds] = terminal_rewards[terminal_inds]
        v_sxa_new = reward_func_sa + expectation_v_prime
        v_sxa_new[terminal_inds] = terminal_rewards[terminal_inds] # Set the boundary states to have a constant value (First-exit formulation)
        v_sxa_new_rs = v_sxa_new.reshape(ns, na)
        v_sxa_argmax_inds = v_sxa_new_rs.argmax(axis = 1)
        v_new = v_sxa_new[state_inds_action_0+v_sxa_argmax_inds]
        # v_new[terminal_inds] = terminal_rewards[terminal_inds]
        delta = np.sum(np.abs(v - v_new))
        # print(delta)
        v = v_new
        # print('VI-step: ',t)
        # print(max_steps)
        # print(delta)
        if delta < 0.001:
            print("Total Iterations (Value Iter): ", t)
            break

    end_time = timer()
    VI_compute_time = end_time-start_time
    print("VI Policy Time Internal: ", VI_compute_time)
    return v, VI_compute_time

def value_iteration_2(T_s_sa, cost_func_sa, v_init, terminal_ind, ns, na):
    # Cost minimization
    # reward_func_sa is a state reward function (no action).  For tLMDP, s = (sigma, x)
    max_steps = T_s_sa.shape[1] + 10
    v = v_init
    state_action_teminal_inds = np.arange(terminal_ind*na, terminal_ind*na+na)
    state_inds_action_0 = [i * na for i in range(ns)]
    delta = 10000
    start_time = timer()
    for t in range(max_steps):
        expectation_v_prime = T_s_sa.dot(v)
        v_sxa_new = cost_func_sa + expectation_v_prime
        v_sxa_new[state_action_teminal_inds] = 0 # Set the boundary states to have a constant value (First-exit formulation)
        v_sxa_new_rs = v_sxa_new.reshape(ns, na)
        v_sxa_argmax_inds = v_sxa_new_rs.argmin(axis = 1)
        v_new = v_sxa_new[state_inds_action_0+v_sxa_argmax_inds]
        # v_new[terminal_inds] = terminal_rewards[terminal_inds]
        delta = np.sum(np.abs(v - v_new))
        # print(delta)
        v = v_new
        # print('VI-step: ',t)
        # print(max_steps)
        # print(delta)
        if delta < 0.001:
            print("Total Iterations (Value Iter): ", t)
            break

    # One-step update for the final value of the terminal state for periodic polices:
    expectation_v_prime_terminals = T_s_sa[state_action_teminal_inds,:].dot(v)
    v_sxa_new_terminals = cost_func_sa[state_action_teminal_inds] + expectation_v_prime_terminals
    v_sxa_new_terminals_argmax_ind = v_sxa_new_terminals.argmin()
    v[terminal_ind] = v_sxa_new_terminals[v_sxa_new_terminals_argmax_ind]

    end_time = timer()
    VI_compute_time = end_time-start_time
    print("VI Policy Time Internal: ", VI_compute_time)
    return v


def z_learning(P, q, terminals, max_iter=1000):
    # P: passive dynamics
    # q: cost function vector
    # terminals: indices for the boundary states, which will be clamped to a desirability of 1
    # Samples from each state of the passive dynamics are drawn using gumble-softmax trick.

    # M_as_numbers = np.argmax(sp.log(P) + np.random.gumbel(size=P.shape), axis=1)
    # M_one_hot = np.eye(P.shape[1])[M_as_numbers].reshape(P.shape)

    P_d = P.todense()
    # q[q==np.inf]=1E10000 # Can't have Inf
    z_hat = np.ones(P.shape[0])*0.1
    eta = 1 # Learning rate
    c = 100 # Learning rate constant.  Same as used in Todorov (2009) PNAS
    delta = 1E5
    for t in range(1, max_iter):
        eta = c/(c+t)
        P_samp_inds = np.argmax(np.log(P_d) + np.random.gumbel(size=P_d.shape), axis=1).A1  # Gumble-softmax
        z_new = (1-eta)*z_hat + eta*np.exp(-q)*z_hat[P_samp_inds]
        z_new[terminals] = 1
        delta = sum(abs(z_hat-z_new))
        # print(z_new)
        print("z-leraning iteration: ", t)
        print("z-learning delta: ", delta)
        z_hat = z_new
        if delta == 0:
            break

    policy = SALMDP.rescalePassiveWithZ(z_hat, P)

    return z_hat, policy

def z_learning_greedy(P, q, terminals, max_iter=5000):
    # TODO: finish implementing this.  Needs the importance sampler.
    # z_learning_greedy is the same as z_learning but with importance sampling in the update
    # P: passive dynamics
    # q: cost function vector
    # terminals: indices for the boundary states, which will be clamped to a desirability of 1

    Q = sp.sparse.diags(np.exp(-q))
    QP = csr.dot(Q, P)
    P_d = QP.todense()
    summed = 1/P_d.sum(axis=1).A1
    summed[np.isinf(summed)]=0
    P_d = P_d*np.diag(summed)
    # P_d = P.todense()
    # q[q==np.inf]=1E10000 # Can't have Inf
    z_hat = np.ones(P.shape[0]) * 0.1
    ns = len(z_hat)
    eta = 1  # Learning rate
    c = 400  # Learning rate constant.  Same as used in Todorov (2009) PNAS
    delta = 1E5
    np.seterr(divide='ignore')
    for t in range(max_iter):
        eta = c / (c + t)
        P_samp_inds = np.argmax(np.log(P_d) + np.random.gumbel(size=P_d.shape), axis=1).A1  # Gumble-softmax.  This requires a dense matrix.  TODO: Might be able to work with sparse.
        z_old = z_hat.copy()

        u = SALMDP.rescalePassiveWithZ(z_hat, P)
        w = P[np.arange(ns), P_samp_inds] / u[np.arange(ns), P_samp_inds] # Importance sampling weight vector
        w[np.isinf(w)]=0
        z_hat = (1 - eta) * z_hat + eta * np.exp(-q) * z_hat[P_samp_inds] * w.A1


        # for i in reversed(range(ns)):
        #     # This loop is so we update z one at a time instead of in full batch, should be more sample efficient.
        #     un_normed = P[i, :].multiply(z_hat)
        #     u = un_normed.multiply(1/un_normed.sum())
        #     w = P[i,P_samp_inds[i]]/u[0,P_samp_inds[i]]
        #     z_hat[i] = (1 - eta) * z_hat[i] + eta * np.exp(-q[i]) * z_hat[P_samp_inds[i]] * w

        z_hat[terminals] = 1
        delta = sum(abs(z_old - z_hat))
        # print(z_new)
        print("z-leraning iteration: ", t)
        print("z-learning delta: ", delta)

        # if delta == 0:
        #     break

    policy = SALMDP.rescalePassiveWithZ(z_hat, P)

    return z_hat, policy


def sigquit_handler(signum, frame):
    print('SIGQUIT received; exiting')
    sys.exit(os.EX_SOFTWARE)
signal.signal(signal.SIGQUIT, sigquit_handler)


def compute_new_local_env(wallmat, P):
    qs = createAllQs(wallmat, sparse=True)
    u_local, _ = computeLMDP(qs, P)
    return u_local


def create_cost_function(termLISet, wall_list, ns, baseCost=3):
    cfunction = np.ones(ns) * np.exp(-baseCost)
    cfunction[termLISet] = 1
    cfunction[wall_list] = 0  # Order matters here, we want to overwrite goal states that are in walls.
    return cfunction

def createAllQs(wallmat, sparse=True):
    cFuncAll = []
    ns = wallmat.size
    baseCost = 3
    wall_list = [i for (i, x) in enumerate(wallmat.flatten()) if x == 1]
    for i in range(ns):
        # self.cFuncAll.append(self.createCostFunction(i, self.ss.wall_list, baseCost))
        if i not in wall_list:
            cFuncAll.append(create_cost_function(i, wall_list, ns, baseCost))
        else:
            cFuncAll.append(np.ones(ns))

    if sparse == True:
        bigQ = sp.sparse.diags(np.array(cFuncAll).flatten())
    elif sparse == False:
        bigQ = np.diag(np.array(cFuncAll).flatten())

    return bigQ


def computeLMDP(Q, P):
    Z = computeZFunction(Q, P)
    U = computePolicy(Z, P)
    return U, Z

def computeZFunction(Q, P):
    # Power iteration.
    eps = 1E-70
    if sp.sparse.issparse(Q):
        QP = csr.dot(Q, P)
    else:
        QP = np.dot(Q, P)

    zold = np.ones(QP.shape[1], dtype=np.double)
    # diff = 10000
    iter = 0
    while iter < 60:
        znew = QP.dot(zold)
        znew = znew/znew.max()
        # diff = np.sum(np.abs(zold-znew))
        if np.sum(np.abs(znew-zold))<eps:
            print("Breaking at iter = " + iter)
            break
        zold = znew

        iter += 1

    # znew[znew == 0] = znew[np.where(znew > 0)].min()
    return znew

def computePolicy(z, p):
    if sp.sparse.issparse(p):
        z = csr(z)
        normalizer = sp.sparse.diags(np.divide(1, p.dot(z.transpose()).toarray()).squeeze(), dtype=np.double)
        scaled = p.multiply(z)
        pol = sp.dot(normalizer, scaled)
    else:
        normalizer = np.diag(np.divide(1, p.dot(z.transpose()).toarray()).squeeze(), dtype=np.double)
        scaled = p.multiply(z)
        pol = sp.dot(normalizer, scaled)
    return pol

def makePolDeterministic(pol):
    rows = np.arange(pol.shape[0])
    cols = np.asarray(pol.argmax(axis=1))[:,0]
    data = np.ones(cols.shape[0])
    # # Fixing the problem where argmax selects ind=0 for empty rows. Set those state dynamics to identity.
    # zero_row_inds = np.argwhere(pol.astype(bool).sum(axis=1) == 0)[:, 0]
    # cols[zero_row_inds] = rows[zero_row_inds]
    detpol = csr((data, (rows, cols)), shape=(pol.shape[0], pol.shape[1]))
    return detpol


def get_all_grounded_states(goal_list):
    return [g.xg for g in goal_list]

def inverse_grounding(goalvarlist, state):
    lst = [g for g in goalvarlist if g.xg == state]
    return lst

def get_satisfying_goals_for_certificate(state, time, goalvarlist):
    lst = [g for g in goalvarlist if (g.xg == state and time<=g.trueDL)]
    return lst

def to_sparse(x):
    """ From https://discuss.pytorch.org/t/how-to-convert-a-dense-matrix-to-a-sparse-one/7809 """
    """ converts dense tensor x to sparse format """
    import torch
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def delete_from_csr(mat, row_indices=[], col_indices=[]):
    """
    Author: Philip: https://stackoverflow.com/users/2142071/philip
    From: https://stackoverflow.com/questions/13077527/is-there-a-numpy-delete-equivalent-for-sparse-matrices
    Remove the rows (denoted by ``row_indices``) and columns (denoted by ``col_indices``) from the CSR sparse matrix ``mat``.
    WARNING: Indices of altered axes are reset in the returned matrix
    """
    if not isinstance(mat, csr):
        raise ValueError("works only for CSR format -- use .tocsr() first")

    rows = []
    cols = []
    if row_indices:
        rows = list(row_indices)
    if col_indices:
        cols = list(col_indices)

    if len(rows) > 0 and len(cols) > 0:
        row_mask = np.ones(mat.shape[0], dtype=bool)
        row_mask[rows] = False
        col_mask = np.ones(mat.shape[1], dtype=bool)
        col_mask[cols] = False
        return mat[row_mask][:,col_mask]
    elif len(rows) > 0:
        mask = np.ones(mat.shape[0], dtype=bool)
        mask[rows] = False
        return mat[mask]
    elif len(cols) > 0:
        mask = np.ones(mat.shape[1], dtype=bool)
        mask[cols] = False
        return mat[:, mask]
    else:
        return mat

def save_data(data, name):
    with open(name, "wb") as f:
        pickle.dump(data, f)

def save_as_pickled_object(obj, filepath):
    """
    This is a defensive way to write pickle.write, allowing for very large files on all platforms
    """
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(obj)
    n_bytes = sys.getsizeof(bytes_out)
    with open(filepath, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])


def try_to_load_as_pickled_object_or_None(filepath):
    """
    This is a defensive way to write pickle.load, allowing for very large files on all platforms
    """
    max_bytes = 2**31 - 1
    try:
        input_size = os.path.getsize(filepath)
        bytes_in = bytearray(0)
        with open(filepath, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        obj = pickle.loads(bytes_in)
    except:
        return None
    return obj


def plot(data, *walls):
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    if walls:
        w = walls[0]
        w = np.abs(w.flatten() - 1)
        data = data * w

    temp = data.copy()
    temp[temp > 0.02] = np.max(temp.flatten())

    if data.ndim == 1 and data.size == ROW_COUNT*COLUMN_COUNT:
        data = data.reshape(ROW_COUNT, COLUMN_COUNT)

    plt.imshow(data, vmin=0, vmax=1, origin='lower')
    # plt.imshow(data, norm=norm, origin='lower')

    # plt.imshow(data, aspect='auto', norm=norm, origin='lower')
    # ax.view_init(30, angle)

    plt.show()

def plotgrid(data, line_width, *walls):
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    if walls:
        w = walls[0]
        w = np.abs(w.flatten() - 1)
        w[w == 0] = np.nan
        data = data * w

    # temp = data.copy()
    # temp[temp > 0.02] = np.max(temp.flatten())

    if data.ndim == 1 and data.size == ROW_COUNT*COLUMN_COUNT:
        data = data.reshape(ROW_COUNT, COLUMN_COUNT)

    cmap = plt.cm.viridis
    cmap.set_bad(color='maroon', alpha=None)
    plt.figure()
    # if vmin and vmax:
    #     im = plt.imshow(data, interpolation='none',  vmin=vmin, vmax=vmax, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax), origin='lower')
    # im = plt.imshow(data, interpolation='none', vmin=vmin, vmax=vmax, aspect='equal', origin='lower', cmap=cmap)
    # else:
    im = plt.imshow(data, interpolation='none', aspect='equal', origin='lower', cmap=cmap)

    ax = plt.gca()
    ax = plt.gca()

    # # Major ticks
    # # if ticks == True:
    # ax.set_yticks(np.arange(0, ROW_COUNT, 1))
    # ax.set_xticks(np.arange(0, COLUMN_COUNT, 1))
    #
    # # Labels for major ticks
    # ax.set_yticklabels(np.arange(1, ROW_COUNT+1, 1))
    # ax.set_xticklabels(np.arange(1, COLUMN_COUNT+1, 1))

    # Minor ticks
    ax.set_yticks(np.arange(-.5, ROW_COUNT, 1), minor=True)
    ax.set_xticks(np.arange(-.5, COLUMN_COUNT, 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=line_width)
    plt.show()

def plotgrid2(data, line_width, vmin=None, vmax=None, *walls):
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    if walls:
        w = walls[0]
        w = np.abs(w.flatten() - 1)
        w[w == 0] = np.nan
        data = data * w

    # temp = data.copy()
    # temp[temp > 0.02] = np.max(temp.flatten())

    if data.ndim == 1 and data.size == ROW_COUNT*COLUMN_COUNT:
        data = data.reshape(ROW_COUNT, COLUMN_COUNT)

    # if not cmap:
    #     cmap = plt.cm.plasma
    #     # cmap = plt.cm.viridis

    cmap = plt.cm.viridis
    cmap = plt.cm.plasma
    cmap.set_bad(color='silver', alpha=None)
    plt.figure()
    # if vmin and vmax:
    #     im = plt.imshow(data, interpolation='none',  vmin=vmin, vmax=vmax, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax), origin='lower')
    im = plt.imshow(data, interpolation='none', vmin=vmin, vmax=vmax, aspect='equal', origin='lower', cmap=cmap)
    # else:
    # im = plt.imshow(data, interpolation='none', aspect='equal', origin='lower', cmap=cmap)

    ax = plt.gca()
    ax = plt.gca()

    # if ticks == True:
    #     # Major ticks
    #     ax.set_yticks(np.arange(0, ROW_COUNT, 1))
    #     ax.set_xticks(np.arange(0, COLUMN_COUNT, 1))
    #
    #     # Labels for major ticks
    #     ax.set_yticklabels(np.arange(1, ROW_COUNT+1, 1))
    #     ax.set_xticklabels(np.arange(1, COLUMN_COUNT+1, 1))
    #
    # Minor ticks
    ax.set_yticks(np.arange(-.5, ROW_COUNT, 1), minor=True)
    ax.set_xticks(np.arange(-.5, COLUMN_COUNT, 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=line_width)
    plt.show()


def plotgrid2_save(data, line_width, header_text, vmin=None, vmax=None, *walls, save_to_list=None, show_plot=True):
    """Modified plotgrid2 that can save images to a list"""
    if walls:
        w = walls[0]
        w = np.abs(w.flatten() - 1)
        w[w == 0] = np.nan
        data = data * w

    if data.ndim == 1 and data.size == ROW_COUNT * COLUMN_COUNT:
        data = data.reshape(ROW_COUNT, COLUMN_COUNT)

    cmap = plt.cm.plasma
    cmap.set_bad(color='silver', alpha=None)

    # Create figure
    fig = plt.figure(figsize=(8, 8))  # Set consistent size

    plt.title(header_text, fontsize=20, fontweight='bold', pad=20)

    im = plt.imshow(data, interpolation='none', vmin=vmin, vmax=vmax,
                    aspect='equal', origin='lower', cmap=cmap)

    ax = plt.gca()
    ax.set_yticks(np.arange(-.5, ROW_COUNT, 1), minor=True)
    ax.set_xticks(np.arange(-.5, COLUMN_COUNT, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=line_width)

    # Save to list if requested
    if save_to_list is not None:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img = Image.open(buf)
        save_to_list.append(img.copy())
        buf.close()

    # Only show if requested
    if show_plot:
        plt.show()

    plt.close(fig)  # Clean up to prevent memory issues


def map_to_range(x, new_min=0, new_max=0.5, old_min=-1, old_max=1):
    # Normalize x to range [0, 1]
    x_normalized = (x - old_min) / (old_max - old_min)
    # Scale to new range [new_min, new_max]
    x_mapped = x_normalized * (new_max - new_min) + new_min
    return x_mapped

def plotgrid2_w_bar(data, line_width, name, vmin=None, vmax=None, *walls):
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    if walls:
        w = walls[0]
        w = np.abs(w.flatten() - 1)
        w[w == 0] = np.nan
        data = data * w

    # temp = data.copy()
    # temp[temp > 0.02] = np.max(temp.flatten())

    if data.ndim == 1 and data.size == ROW_COUNT*COLUMN_COUNT:
        data = data.reshape(ROW_COUNT, COLUMN_COUNT)

    # if not cmap:
    #     cmap = plt.cm.plasma
    #     # cmap = plt.cm.viridis

    cmap = plt.cm.viridis
    cmap = plt.cm.plasma
    cmap.set_bad(color='silver', alpha=None)
    plt.figure()
    # if vmin and vmax:
    #     im = plt.imshow(data, interpolation='none',  vmin=vmin, vmax=vmax, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax), origin='lower')
    im = plt.imshow(data, interpolation='none', vmin=vmin, vmax=vmax, aspect='equal', origin='lower', cmap=cmap)
    # else:
    # im = plt.imshow(data, interpolation='none', aspect='equal', origin='lower', cmap=cmap)

    ax = plt.gca()
    ax = plt.gca()

    # if ticks == True:
    #     # Major ticks
    #     ax.set_yticks(np.arange(0, ROW_COUNT, 1))
    #     ax.set_xticks(np.arange(0, COLUMN_COUNT, 1))
    #
    #     # Labels for major ticks
    #     ax.set_yticklabels(np.arange(1, ROW_COUNT+1, 1))
    #     ax.set_xticklabels(np.arange(1, COLUMN_COUNT+1, 1))
    #
    # Minor ticks
    ax.set_yticks(np.arange(-.5, ROW_COUNT, 1), minor=True)
    ax.set_xticks(np.arange(-.5, COLUMN_COUNT, 1), minor=True)

    # Create a color bar

    cbar = plt.colorbar(im)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cbar.set_ticklabels([0, 0.25, 0.5, 0.75, 1])
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('right')
    cbar.ax.set_position([0.85, 0.1, 0.03, 0.8])

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=line_width)
    plt.savefig(name, transparent=True)
    plt.show()


def plotgrid_frame(data, line_width, vmin=None, vmax=None, gl='w', null_color='silver', cmap1=plt.cm.PiYG, cmap2=plt.cm.PiYG, alpha=1, fig=None, *walls):
    if fig in locals():
        print('fig loaded')
    else:
        fig, ax = plt.subplots()
    frames = []
    # data_1 = data_1_lst[0]
    # data_2 = data_2_lst[0]
    if walls:
        w = walls[0]
        w = np.abs(w.flatten() - 1)
        w[w == 0] = np.nan
        data = data.flatten() * w

    if data.ndim == 1 and data.size == ROW_COUNT*COLUMN_COUNT:
        data = data.reshape(ROW_COUNT, COLUMN_COUNT)


    cmap1.set_bad(color=null_color, alpha=alpha)
    cmap2.set_bad(color=null_color, alpha=alpha)

    # combined_data = np.zeros(data_1.shape)
    # combined_data = combined_data + data_1 + data_2

    colors1 = cmap1(np.linspace(0, 0.001, 128))
    colors2 = cmap2(np.linspace(0, 0.001, 128))
    all_colors = np.vstack((colors1, colors2))
    custom_cmap = mcolors.ListedColormap(all_colors)
    custom_cmap.set_bad(color=null_color, alpha=alpha)

    img11 = ax.imshow(data, interpolation='none', vmin=0, vmax=1, aspect='equal', origin='lower', cmap=cmap1)

    # img_1 = ax.imshow(data_1, interpolation='none', vmin=vmin, vmax=vmax, aspect='equal', origin='lower', cmap=cmap1)
    # img_2 = ax.imshow(data_2, interpolation='none', vmin=0, vmax=1, aspect='equal', origin='lower', cmap=cmap2)
    # frame.append(img_1)
    # frame.append(img_2)

    ax = plt.gca()

    # Minor ticks
    ax.set_yticks(np.arange(-.5, ROW_COUNT, 1), minor=True)
    ax.set_xticks(np.arange(-.5, COLUMN_COUNT, 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which='minor', color=gl, linestyle='-', linewidth=1)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_xlim(-0.5, data.shape[1] - 0.5)
    plt.gca().set_ylim(-0.5, data.shape[0] - 0.5)
    plt.gca().spines[:].set_visible(False)

    cbar = plt.colorbar(img11)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cbar.set_ticklabels([0, 0.25, 0.5, 0.75, 1])

    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.set_position([0.85, 0.1, 0.03, 0.8])
    plt.show()
    print('plot done')
    return img11, fig

    # def update(frame):
    #     data1 = data_1_lst[frame]
    #     data2 = data_2_lst[frame]
    #     if walls:
    #         w = walls[0]
    #         w = np.abs(w.flatten() - 1)
    #         w[w == 0] = np.nan
    #         data = data1 * w
    #         data.reshape(ROW_COUNT, COLUMN_COUNT)
    #     if walls:
    #         w = walls[0]
    #         w = np.abs(w.flatten() - 1)
    #         w[w == 0] = np.nan
    #         data = data2 * w
    #         data.reshape(ROW_COUNT, COLUMN_COUNT)
    #     img_1.set_data(data1)
    #     img_2.set_data(data2)
    # return frame

    # ani = animation.FuncAnimation(fig, update, frames=len(data_1_lst)-1, repeat=False)
    # import matplotlib as mpl
    # mpl.rcParams['animation.ffmpeg_path'] = '/opt/homebrew/bin/ffmpeg'
    # ani.save('animation_v1.mp4', writer='ffmpeg', fps=30)


def plotgrid3(data, line_width, vmin=None, vmax=None, gl='w', null_color='silver', cmap=plt.cm.PiYG, alpha=0, *walls):
    # fig, ax = plt.subplots()
    if walls:
        w = walls[0]
        w = np.abs(w.flatten() - 1)
        w[w == 0] = np.nan
        data = data * w


    if data.ndim == 1 and data.size == ROW_COUNT*COLUMN_COUNT:
        data = data.reshape(ROW_COUNT, COLUMN_COUNT)

    # null_color = 'silver'
    # cmap = plt.cm.viridis
    cmap = plt.cm.plasma
    # cmap = plt.cm.PiYG
    # cmap = cmap
    cmap.set_bad(color=null_color, alpha=alpha)
    plt.figure()
    im = plt.imshow(data, interpolation='none', vmin=vmin, vmax=vmax, aspect='equal', origin='lower', cmap=cmap)

    # ax = plt.gca()
    ax = plt.gca()

    # Minor ticks
    ax.set_yticks(np.arange(-.5, ROW_COUNT, 1), minor=True)
    ax.set_xticks(np.arange(-.5, COLUMN_COUNT, 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which='minor', color=gl, linestyle='-', linewidth=line_width)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_xlim(-0.5, data.shape[1] - 0.5)
    plt.gca().set_ylim(-0.5, data.shape[0] - 0.5)
    plt.gca().spines[:].set_visible(False)

    # cax = divider.append_axes("left", size="5%", pad=0.1)

    # Create a color bar

    cbar = plt.colorbar(im)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cbar.set_ticklabels([0, 0.25, 0.5, 0.75, 1])
    # cbar.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    # cbar.set_ticklabels([1, 0.75, 0.5, 0.25, 0, 0.25, 0.5, 0.75, 1])
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.set_position([0.85, 0.1, 0.03, 0.8])
    # Move color bar ticks to the left
    # cax.yaxis.set_ticks_position('left')
    # cax.yaxis.set_label_position('left')
    # plt.colorbar()
    plt.show()

    # return frame


def nonzero_min(arr):
    """Return the minimum nonzero value in a NumPy array."""
    nonzeros = arr[arr != 0]
    return nonzeros.min() if nonzeros.size > 0 else 0  # or np.inf if you prefer

def plotgrid3_1D(data, line_width, row_cnt, col_cnt, vmin=None, vmax=None, gl='w', null_color='silver', cmap=plt.cm.PiYG, alpha=0):

    if data.ndim == 1 and data.size == row_cnt * col_cnt:
        data = data.reshape(row_cnt, col_cnt)

    cmap = cmap
    cmap.set_bad(color=null_color, alpha=alpha)
    plt.figure()
    im = plt.imshow(data, interpolation='none', vmin=vmin, vmax=vmax, aspect='equal', origin='lower', cmap=cmap)

    # ax = plt.gca()
    ax = plt.gca()

    # Minor ticks
    ax.set_yticks(np.arange(-.5, row_cnt, 1), minor=True)
    ax.set_xticks(np.arange(-.5, col_cnt, 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which='minor', color=gl, linestyle='-', linewidth=line_width)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_xlim(-0.5, data.shape[1] - 0.5)
    plt.gca().set_ylim(-0.5, data.shape[0] - 0.5)
    plt.gca().spines[:].set_visible(False)

    cbar = plt.colorbar(im)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cbar.set_ticklabels([0, 0.25, 0.5, 0.75, 1])
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.set_position([0.85, 0.1, 0.03, 0.8])

    plt.show()

    # return frame


def plotgrid4(data1, data2, line_width, vmin=None, vmax=None, gl='w', null_color='silver', cmap=plt.cm.PiYG, alpha=0, *walls):

    if walls:
        w = walls[0]
        w = np.abs(w.flatten() - 1)
        w[w == 0] = np.nan
        data1 = data1 * w
        data2 = data2 * w


    if data1.ndim == 1 and data1.size == ROW_COUNT*COLUMN_COUNT:
        data = data1.reshape(ROW_COUNT, COLUMN_COUNT)

    if data2.ndim == 1 and data2.size == ROW_COUNT*COLUMN_COUNT:
        data = data2.reshape(ROW_COUNT, COLUMN_COUNT)

    # cmap = plt.cm.viridis
    # cmap = plt.cm.plasma
    # cmap = plt.cm.PiYG
    cmap = cmap
    cmap.set_bad(color=null_color, alpha=alpha)
    plt.figure()
    im = plt.imshow(data1, interpolation='none', vmin=vmin, vmax=vmax, aspect='equal', origin='lower', cmap=cmap, alpha=0.5)
    im = plt.imshow(data2, interpolation='none', vmin=vmin, vmax=vmax, aspect='equal', origin='lower', cmap=cmap, alpha=0.5)

    ax = plt.gca()
    ax = plt.gca()

    # Minor ticks
    ax.set_yticks(np.arange(-.5, ROW_COUNT, 1), minor=True)
    ax.set_xticks(np.arange(-.5, COLUMN_COUNT, 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which='minor', color=gl, linestyle='-', linewidth=line_width)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_xlim(-0.5, data.shape[1] - 0.5)
    plt.gca().set_ylim(-0.5, data.shape[0] - 0.5)
    plt.gca().spines[:].set_visible(False)

    # cax = divider.append_axes("left", size="5%", pad=0.1)

    # Create a color bar

    cbar = plt.colorbar(im)
    cbar.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    cbar.set_ticklabels([1, 0.75, 0.5, 0.25, 0, 0.25, 0.5, 0.75, 1])
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.set_position([0.85, 0.1, 0.03, 0.8])
    # Move color bar ticks to the left
    # cax.yaxis.set_ticks_position('left')
    # cax.yaxis.set_label_position('left')

    # plt.colorbar()
    plt.show()




def plot2(data, onezero, *walls):
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    if walls:
        w = walls[0]
        w = np.abs(w.flatten() - 1)
        data = data * w

    temp = data.copy()
    temp[temp > 0.02] = np.max(temp.flatten())

    if data.ndim == 1 and data.size == ROW_COUNT*COLUMN_COUNT:
        data = data.reshape(ROW_COUNT, COLUMN_COUNT)

    norm = plt.Normalize(vmin=np.min(temp), vmax=np.max(temp))
    if onezero:
        plt.imshow(data, vmin=0, vmax=1, origin='lower')
    else:
        plt.imshow(data, norm=norm, origin='lower')
        # plt.imshow(data, aspect='auto', norm=norm, origin='lower')
    # ax.view_init(30, angle)

    plt.show()


def plot_at_y(arr, val, **kwargs):
    plt.plot(arr, np.zeros_like(arr) + val, 'x', **kwargs)
    plt.show()


def polquiver(u, liToCoord, wall_list, wallMat, valmap=[]):
    u = u.todense()
    u[u < 0.03] = 0
    X, Y, U, V = [], [], [], []
    for i in range(u.shape[0]):
        if i not in wall_list:
            for j in u[i, :].nonzero()[1]:
                xx = liToCoord[i][1]
                yy = liToCoord[i][0]
                uu = (liToCoord[j][1] - xx) * u[i, j]
                vv = (liToCoord[j][0] - yy) * u[i, j]
                X.append(xx)
                Y.append(yy)
                U.append(uu)
                V.append(vv)
    X = np.array(X)
    Y = np.array(Y)
    U = np.array(U)
    V = np.array(V)
    Z = np.array([0]*X.size)
    # plt.hold
    fig = plt.figure()
    # ax = fig.gca(projection='3d')

    xx, yy = np.meshgrid(np.linspace(0, 1, COLUMN_COUNT), np.linspace(0, 1, ROW_COUNT))

    # create vertices for a rotated mesh (3D rotation matrix)
    Xp = xx
    Yp = yy
    Zp = 0 * np.ones(Xp.shape)


    # ax.plot_surface(valmap)

    # ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
    # cset = ax.contourf(X, Y, Z, zdir='z', offset=-100,
    #                    levels=np.linspace(-100, 100, 1200), cmap=plt.cm.jet)
    # cset = ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap=plt.cm.jet)
    # cset = ax.contourf(X, Y, Z, zdir='y', offset=40, cmap=plt.cm.jet)
    # ax.set_xlabel('X')
    # ax.set_xlim(-40, 40)
    # ax.set_ylabel('Y')
    # ax.set_ylim(-40, 40)
    # ax.set_zlabel('Z')
    # ax.set_zlim(-100, 100)
    # plt.show()

    # ax = Axes3D(fig)

    # ax2 = fig.add_subplot(111, projection='3d')
    # ax2.plot_surface(Xp, Yp, Zp, rstride=1, cstride=1, facecolors=plt.cm.BrBG(valmap), shade=False)

    # ax1 = fig.add_subplot(121)
    # ax1.imshow(valmap, cmap=plt.cm.BrBG, interpolation='nearest', origin='lower', extent=[0, 1, 0, 1])

    # ax.quiver(X, Y, np.array([1]*X.size), U, V, np.array([0]*X.size), color='r')
    # ax = fig.add_subplot(111, projection='3d')
    # if valmap.ndim == 1 and valmap.size == ROW_COUNT*COLUMN_COUNT:
    #     data = valmap.reshape(ROW_COUNT, COLUMN_COUNT)
    # plt.imshow(valmap, aspect='auto', origin='lower')
    # if valmap.any():
    #     plot(valmap)

    w = np.abs(wallMat.flatten() - 1)

    plt.quiver(X, Y, U, V, width=0.006, scale=18, headwidth=3, headlength=3, headaxislength=3, color='r')
    plt.imshow(np.abs(wallMat-1)*.85, vmin=0, vmax=1, origin='lower', cmap='gray')
    plt.show()


def polquiver_pi(pi, Px, liToCoord, wall_list, wallMat, valmap=[]):
    u[u < 0.01] = 0
    X, Y, U, V = [], [], [], []
    for i in range(u.shape[0]):
        if i not in wall_list:
            for j in u[i, :].nonzero()[1]:
                xx = liToCoord[i][1]
                yy = liToCoord[i][0]
                uu = (liToCoord[j][1] - xx) * u[i, j]
                vv = (liToCoord[j][0] - yy) * u[i, j]
                X.append(xx)
                Y.append(yy)
                U.append(uu)
                V.append(vv)
    X = np.array(X)
    Y = np.array(Y)
    U = np.array(U)
    V = np.array(V)
    Z = np.array([0]*X.size)
    # plt.hold
    fig = plt.figure()
    # ax = fig.gca(projection='3d')

    xx, yy = np.meshgrid(np.linspace(0, 1, COLUMN_COUNT), np.linspace(0, 1, ROW_COUNT))

    # create vertices for a rotated mesh (3D rotation matrix)
    Xp = xx
    Yp = yy
    Zp = 0 * np.ones(Xp.shape)

    w = np.abs(wallMat.flatten() - 1)

    plt.quiver(X, Y, U, V, width=0.006, scale=18, headwidth=3, headlength=3, headaxislength=3, color='r')
    plt.imshow(np.abs(wallMat-1)*.85, vmin=0, vmax=1, origin='lower', cmap='gray')
    plt.show()

def polquiver(u, liToCoord, wall_list, wallMat, valmap=[]):
    # u = u.todense()
    u[u < 0.01] = 0
    X, Y, U, V = [], [], [], []
    for i in range(u.shape[0]):
        if i not in wall_list:
            for j in u[i, :].nonzero()[0]:
                xx = liToCoord[i][1]
                yy = liToCoord[i][0]
                uu = (liToCoord[j][1] - xx) * u[i, j]
                vv = (liToCoord[j][0] - yy) * u[i, j]
                X.append(xx)
                Y.append(yy)
                U.append(uu)
                V.append(vv)
    X = np.array(X)
    Y = np.array(Y)
    U = np.array(U)
    V = np.array(V)
    Z = np.array([0]*X.size)
    # plt.hold
    fig = plt.figure()
    # ax = fig.gca(projection='3d')

    xx, yy = np.meshgrid(np.linspace(0, 1, COLUMN_COUNT), np.linspace(0, 1, ROW_COUNT))

    # create vertices for a rotated mesh (3D rotation matrix)
    Xp = xx
    Yp = yy
    Zp = 0 * np.ones(Xp.shape)

    w = np.abs(wallMat.flatten() - 1)

    plt.quiver(X, Y, U, V, width=0.006, scale=18, headwidth=3, headlength=3, headaxislength=3, color='r')
    plt.imshow(np.abs(wallMat-1)*.85, vmin=0, vmax=1, origin='lower', cmap='gray')
    plt.show()


def resetbuttons(list):
    for e in list:
        if e.kind == 'button':
            e.active = True

def onehot(inds, size):
    oh = np.zeros(size)
    oh[inds] = 1
    return oh


def powerset(iterable):
    "list(powerset([1,2,3])) --> [(), (1,), (2,), (3,), (1,2), (1,3), (2,3), (1,2,3)]"
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))

def get_obj_size(obj):
    marked = {id(obj)}
    obj_q = [obj]
    sz = 0

    while obj_q:
        sz += sum(map(sys.getsizeof, obj_q))

        # Lookup all the object referred to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}

        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())

    return sz

def pset(myset):
  if not myset: # Empty list -> empty set
    return [set()]

  r = []
  for y in myset:
    sy = set((y,))
    for x in pset(myset - sy):
      if x not in r:
        r.extend([x, x|sy])
  return r

def matstring(mat):
    return "".join(str(x) for x in mat.flatten())

def angle_between_vectors(u, v):
    return np.arccos(u.dot(v)/(np.sum(u**2)*np.sum(v**2)))

def get_bounding_box(x, y, radius):
     xlow = x - radius
     xhi = x + radius
     ylow = y - radius
     yhi = y + radius

     # If the agent is near the edge, the bounding box will not center around the agent, mod_params account for this
     x_mod = xlow - abs(xlow)
     y_mod = ylow - abs(ylow)

     if xhi >= COLUMN_COUNT:
         x_mod = COLUMN_COUNT - (xhi+1)
     elif xlow < 0:
         x_mod = abs(xlow)

     if yhi >= ROW_COUNT:
         y_mod = ROW_COUNT - (yhi+1)
     elif ylow < 0:
         y_mod = abs(ylow)

     xlow += x_mod
     xhi += x_mod
     ylow += y_mod
     yhi += y_mod

     return xlow, xhi, ylow, yhi

def wallTransform(buttonwallinds, curwallmat):
    buttonwallmat = np.abs(onehot(buttonwallinds, COLUMN_COUNT*ROW_COUNT)-1)
    newwallmat = np.reshape((np.logical_and(curwallmat.flatten(), buttonwallmat) + 0), [ROW_COUNT, COLUMN_COUNT])
    return newwallmat

def wallTransform2(buttonwallinds, curwallmat, logic_function=np.logical_and):
    buttonwallmat = np.abs(onehot(buttonwallinds, COLUMN_COUNT*ROW_COUNT)-1)
    newwallmat = np.reshape((logic_function(curwallmat.flatten(), buttonwallmat) + 0), [ROW_COUNT, COLUMN_COUNT])
    return newwallmat

def compute_empowerment(init_state, p, horizon):
    # For Empowerment Computation
    number_of_actions = p.shape[0]
    lst = list(np.arange(number_of_actions))
    n = horizon  # Horizon
    # Generate all sequences of length n with replacement
    all_sequences = list(itertools.product(lst, repeat=n))

    nx = p.shape[-1]
    vec = onehot(init_state, nx)
    channel = []
    for i, seq in enumerate(all_sequences):
        result = functools.reduce(lambda v, a: np.dot(v, p[a]), seq, vec)
        channel.append(result)
    channel = np.array(channel)
    channel_sparse = sp.sparse.csr_matrix(channel)
    C, r, q = BlahutArimotoSparse(channel_sparse)
    return C




# Random Plotting tools for Blahut-Arimoto
    # X, Y = np.meshgrid(np.arange(COLUMN_COUNT), np.arange(ROW_COUNT))
    # Xf = X.flatten() + 0.5
    # Yf = Y.flatten() + 0.5
    # endpnts = np.array([self.cEnv.ss.liToCoord[i] for i in self.piSet[3]])

    # for k, _ in enumerate(self.piSet):
    #     print(k)
    #     endpnts = np.array([self.cEnv.ss.liToCoord[j] for j in self.piSet[k]])
    #     xarrowlength = endpnts[:, 1] - X.flatten()
    #     yarrowlength = endpnts[:, 0] - Y.flatten()
    #     plt.figure()
    #     plt.title('Simple_Agent Dynamics, Period 0')
    #     plt.ylim(0, ROW_COUNT + 1)
    #     plt.xlim(0, COLUMN_COUNT + 1)
    #     for i, a in enumerate(Xf):
    #         plt.arrow(Xf[i], Yf[i], xarrowlength[i], yarrowlength[i], head_width=0.15, length_includes_head=False, head_length=0.1, fc='k', ec='k')
    #     plt.show()
    # plt.quiver(X, Y, endpnts[:, 1], endpnts[:, 0])
    # plt.show()

    # # Blauth-Arimotho Algorithm
    # Assuming X and Y as input and output variables of the channel respectively and r(x) is the input distributions. <br>
    # The capacity of a channel is defined by <br>
    # $C = \max_{r(x)} I(X;Y) = \max_{r(x)} \sum_{x} \sum_{y} r(x) p(y|x) \log \frac{r(x) p(y|x)}{r(x) \sum_{\tilde{x}} r(\tilde{x})p(y|\tilde{x})}$

    # In[79]:

    # https://github.com/kobybibas/blahut_arimoto_algorithm/blob/master/blahut_arimoto_algorithm.py
def blahut_arimoto_2(p_y_x: np.ndarray, log_base: float = 2, thresh: float = 1e-12, max_iter: int = 1e3) -> tuple:
    '''
    Maximize the capacity between I(X;Y)
    p_y_x: each row represnets probability assinmnet
    log_base: the base of the log when calaculating the capacity
    thresh: the threshold of the update, finish the calculation when gettting to it.
    max_iter: the maximum iterations of the calculation
    '''

    # Input test
    assert np.abs(p_y_x.sum(axis=1).mean() - 1) < 1e-6
    assert p_y_x.shape[0] > 1

    # The number of inputs: size of |X|
    m = p_y_x.shape[0]

    # The number of outputs: size of |Y|
    n = p_y_x.shape[1]

    # Initialize the prior uniformly
    r = np.ones((1, m)) / m

    # Compute the r(x) that maximizes the capacity
    for iteration in range(int(max_iter)):

        q = r.T * p_y_x
        q = q / np.sum(q, axis=0)

        r1 = np.prod(np.power(q, p_y_x), axis=1)
        r1 = r1 / np.sum(r1)

        tolerance = np.linalg.norm(r1 - r)
        r = r1
        if tolerance < thresh:
            break

    # Calculate the capacity
    r = r.flatten()
    c = 0
    for i in range(m):
        if r[i] > 0:
            c += np.sum(r[i] * p_y_x[i, :] *
                        np.log(q[i, :] / r[i] + 1e-16))
    c = c / np.log(log_base)
    return c, r

# From ChatGPT-4: DO NOT USE YET, NOT TESTED!
def generalized_blahut_arimoto(p_var_given_a_factors, tol=1e-6, max_iter=1000):
    num_factors = len(p_var_given_a_factors)
    num_inputs = p_var_given_a_factors[0].shape[0]
    num_outputs = [factor.shape[1] for factor in p_var_given_a_factors]

    # Initialize input distribution p(x) and auxiliary distributions q(y|x)
    p_a = np.full(num_inputs, 1 / num_inputs)
    q_var_given_a_list = [np.full((num_inputs, output_size), 1 / output_size) for output_size in num_outputs]

    for _ in range(max_iter):
        p_x_prev = p_a.copy()

        # Update input distribution p(x)
        p_a = np.prod([q_var_given_a_list[i].dot(p_var_given_a_factors[i].T) for i in range(num_factors)], axis=0)
        p_a /= np.sum(p_a)

        # Update auxiliary distributions q(y|x)
        for i in range(num_factors):
            numerator = p_var_given_a_factors[i] * p_a[:, np.newaxis]
            denominator = np.sum(numerator, axis=0)
            q_var_given_a_list[i] = numerator / denominator

        # Check for convergence
        if np.sum(np.abs(p_a - p_x_prev)) < tol:
            break

    # Compute channel capacity
    mutual_info = 0
    for i in range(num_factors):
        joint_distribution = p_a[:, np.newaxis] * p_var_given_a_factors[i] * q_var_given_a_list[i]
        marginal_distribution_y = np.sum(joint_distribution, axis=0)
        mutual_info += np.sum(joint_distribution * np.log2(joint_distribution / (p_a[:, np.newaxis] * marginal_distribution_y)))

    return mutual_info, p_a


def BlahutArimotoSparse(p_sparse, max_iter=1000, tol=1e-5):
    """
    Port of MATLAB code.
    Vectorized Blahut-Arimoto algorithm using SciPy sparse matrices,
    with minimal explicit Python loops.

    Parameters
    ----------
    p_sparse : (m x n) scipy.sparse matrix
        Transition probabilities p(i, j).
        Row i: distribution of outputs given input i.
    max_iter : int
        Maximum number of Blahut-Arimoto iterations.
    tol : float
        Convergence tolerance (on the input distribution r).

    Returns
    -------
    C : float
        Channel capacity in bits.
    r : (m,) np.ndarray
        Capacity-achieving input distribution.
    q : scipy.sparse.csr_matrix, shape (m, n)
        Conditional distribution over inputs given each output (q(i,j)).
    """

    # 1) If sum of the entire matrix is 0, no transitions => capacity = 0
    total_sum = p_sparse.sum()
    if total_sum == 0:
        return 0.0, np.zeros(0), csr_matrix((0,0))

    # 2) Check for negative entries
    if p_sparse.data.min() < 0:
        print("Error: some entry in the input matrix is negative")
        return 0.0, None, None

    # Convert to CSR for consistent row-based operations
    p_sparse = p_sparse.tocsr()
    m, n = p_sparse.shape

    # 3) Remove zero columns
    col_sum = np.asarray(p_sparse.sum(axis=0)).ravel()  # sum along columns
    zero_cols = np.where(col_sum == 0)[0]
    if zero_cols.size > 0:
        keep_cols = [j for j in range(n) if col_sum[j] != 0]
        p_sparse = p_sparse[:, keep_cols]  # slice out zero columns
        n = p_sparse.shape[1]             # update n

    # 4) Remove zero rows
    row_sum = np.asarray(p_sparse.sum(axis=1)).ravel()  # sum along rows
    zero_rows = np.where(row_sum == 0)[0]
    if zero_rows.size > 0:
        print("Error: there is a zero row in the input matrix")
        # In the original code, it then tries p(:, zero_rows)=[] and returns
        # We'll replicate that logic: remove those columns and return capacity=0
        # (Although it's likely a bug/quirk in the original.)
        keep_cols = [j for j in range(p_sparse.shape[1]) if j not in zero_rows]
        if len(keep_cols) < p_sparse.shape[1]:
            p_sparse = p_sparse[:, keep_cols]
        return 0.0, None, None
    else:
        # Normalize each row to sum to 1
        row_sum = np.asarray(p_sparse.sum(axis=1)).ravel()
        inv_row_sum = 1.0 / row_sum
        # Multiply on the left by diag(1 / row_sum)
        D = diags(inv_row_sum, 0, shape=(m, m))
        p_sparse = D.dot(p_sparse)

    # Possibly re-check shape after trimming
    m, n = p_sparse.shape

    # 5) Initialize r uniformly
    r = np.full(m, 1.0 / m, dtype=float)
    error_tolerance = tol / m

    # We'll run up to max_iter times
    for _ in range(max_iter):
        # ----  A) Compute q = each column normalized.  ----
        #
        #  q(i,j) = [ r(i)*p(i,j) ] / sum_i [ r(i)*p(i,j) ]
        #
        # We can do that in bulk by:
        #    temp = p(i,j) * r(i)        (elementwise multiplication by row)
        # then for each column j, divide by the sum.  We'll do:
        #    colSum = temp.sum(axis=0)
        #    q = temp.multiply(1/colSum)

        # Multiply each row of p by r(i):
        # Easiest is to do a diagonal multiply: p_r = diag(r) * p
        # but that is a normal sparse matmul. We can do a simpler elementwise multiply:
        # We'll turn r into a column for broadcast:
        #   Actually a direct SciPy trick: p_r = p_sparse.multiply(r[:, None])
        #   means each row i is scaled by r[i].
        p_r = p_sparse.multiply(r[:, None])  # shape (m,n)

        # Sum over rows => shape (1,n)
        col_sum = np.asarray(p_r.sum(axis=0)).ravel()
        # Avoid zero-division if a column is still zero
        nonzero_mask = (col_sum != 0)
        col_inv = np.zeros_like(col_sum)
        col_inv[nonzero_mask] = 1.0 / col_sum[nonzero_mask]

        # q = p_r.multiply(col_inv)
        # But col_inv is length n, so we can broadcast along columns by:
        q = p_r.multiply(col_inv)

        # ----  B) Update r by log-domain formula:  r(i) = exp( sum_j [ p(i,j)*log(q(i,j)) ] )  ----
        # We'll compute row_sums of p(i,j)*log(q(i,j))
        #
        #  1) compute log(q) in place
        #  2) do elementwise multiply p * log(q)
        #  3) sum over columns => log r(i)
        #  4) exponentiate and normalize

        # Convert q to COOrdinate so we can do q.data = log(q.data) easily
        q_coo = q.tocoo(copy=True)
        # q_coo.data might have zeros => log(0) => -inf. This is expected if q(i,j)=0.
        # We'll ignore those since p(i,j)*0 won't contribute anyway.
        # But to keep from getting NaNs, we can do a safe log.
        # We'll do:
        mask_nz = (q_coo.data > 0)
        q_coo.data[mask_nz] = np.log(q_coo.data[mask_nz])
        # The rest remain -inf (or some negative large) but won't matter once multiplied by p(i,j) = 0 or sum.

        # p * log(q) => elementwise
        # We can do p_coo = p_sparse in coo as well, but let's do it as a direct multiply if shapes match.
        # We'll turn p_sparse also into coo so we can do parallel index-wise operations.
        p_coo = p_sparse.tocoo(copy=False)

        # We want sum_{j} p(i,j)*log(q(i,j)) for each row i
        # The easiest way is:
        #   E = (p * log(q)) as a new sparse. Then sum rows.
        # But we have to ensure we only multiply matching (i,j).
        # Let's do a dictionary approach or direct approach:
        # Instead, let's do an efficient built-in: E = p.multiply(logq) => then E.sum(axis=1)

        # But p and q are both in CSR format. Actually q is in coo. Let's re-convert q_coo -> q_csr
        q_csr = q_coo.tocsr()
        # Now E = p_sparse.multiply(q_csr) is the elementwise multiply
        E = p_sparse.multiply(q_csr)
        # sum rows => shape (m,1)
        log_r1 = np.asarray(E.sum(axis=1)).ravel()
        # r1(i) = exp(log_r1(i))
        r1 = np.exp(log_r1)
        # Normalize r1
        sum_r1 = r1.sum()
        if sum_r1 > 0:
            r1 /= sum_r1

        # ----  C) Check convergence  ----
        diff = np.linalg.norm(r1 - r, 1)  # L1 or L2 norm, either is typical
        r = r1
        if diff < error_tolerance:
            break

    # ----  D) Compute channel capacity in bits  ----
    #
    #    C = sum_{i,j} r(i)*p(i,j)* log( q(i,j)/r(i) )
    #      = sum_{i,j} r(i)*p(i,j)* [log q(i,j) - log r(i)]
    #
    # We'll do this in a vectorized way:
    # 1) ratio = log(q) - log(r(i))   (row-wise broadcast)
    # 2) multiply by r(i)*p(i,j), sum
    #
    # We already have q from above. Let's re-log it in a stable manner, then subtract log(r) row by row.

    # A) log(q) again (since we overwrote q_coo with log values):
    # Actually we *already* have log q in q_csr as q_csr.data, which is log(q).
    # We'll retrieve that:
    log_q_csr = q_csr  # This contains log(q) in .data

    # B) We'll build ratio = log(q) - log(r(i))
    #    We can do that in coordinate form so we can subtract log(r[row]) easily.
    ratio_coo = log_q_csr.tocoo(copy=True)
    log_r = np.log(r)
    ratio_coo.data -= log_r[ratio_coo.row]

    # C) ratio in CSR
    ratio_csr = ratio_coo.tocsr()

    # D) Multiply by r(i)*p(i,j) => define rp = p.multiply(r row-wise)
    #    Then sum up rp*ratio
    rp = p_sparse.multiply(r[:, None])  # each row i scaled by r[i]
    # elementwise multiply => rp.multiply(ratio_csr), then sum everything
    final_mat = rp.multiply(ratio_csr)
    C_nats = final_mat.sum()
    C_bits = C_nats / math.log(2)
    C = float(C_bits)
    q = q.tocsr()
    return C, r, q


def BlahutArimotoSparse(p_sparse, max_iter=1000, tol=1e-5):
    """
    Vectorized Blahut-Arimoto algorithm using SciPy sparse matrices,
    with minimal explicit Python loops.

    Parameters
    ----------
    p_sparse : (m x n) scipy.sparse matrix
        Transition probabilities p(i, j).
        Row i: distribution of outputs given input i.
    max_iter : int
        Maximum number of Blahut-Arimoto iterations.
    tol : float
        Convergence tolerance (on the input distribution r).

    Returns
    -------
    C : float
        Channel capacity in bits.
    r : (m,) np.ndarray
        Capacity-achieving input distribution.
    q : scipy.sparse.csr_matrix, shape (m, n)
        Conditional distribution over inputs given each output (q(i,j)).
    """

    # 1) If sum of the entire matrix is 0, no transitions => capacity = 0
    total_sum = p_sparse.sum()
    if total_sum == 0:
        return 0.0, np.zeros(0), csr_matrix((0,0))

    # 2) Check for negative entries
    if p_sparse.data.min() < 0:
        print("Error: some entry in the input matrix is negative")
        return 0.0, None, None

    # Convert to CSR for consistent row-based operations
    p_sparse = p_sparse.tocsr()
    m, n = p_sparse.shape

    # 3) Remove zero columns
    col_sum = np.asarray(p_sparse.sum(axis=0)).ravel()  # sum along columns
    zero_cols = np.where(col_sum == 0)[0]
    if zero_cols.size > 0:
        keep_cols = [j for j in range(n) if col_sum[j] != 0]
        p_sparse = p_sparse[:, keep_cols]  # slice out zero columns
        n = p_sparse.shape[1]             # update n

    # 4) Remove zero rows
    row_sum = np.asarray(p_sparse.sum(axis=1)).ravel()  # sum along rows
    zero_rows = np.where(row_sum == 0)[0]
    if zero_rows.size > 0:
        print("Error: there is a zero row in the input matrix")
        # In the original code, it then tries p(:, zero_rows)=[] and returns
        # We'll replicate that logic: remove those columns and return capacity=0
        # (Although it's likely a bug/quirk in the original.)
        keep_cols = [j for j in range(p_sparse.shape[1]) if j not in zero_rows]
        if len(keep_cols) < p_sparse.shape[1]:
            p_sparse = p_sparse[:, keep_cols]
        return 0.0, None, None
    else:
        # Normalize each row to sum to 1
        row_sum = np.asarray(p_sparse.sum(axis=1)).ravel()
        inv_row_sum = 1.0 / row_sum
        # Multiply on the left by diag(1 / row_sum)
        D = diags(inv_row_sum, 0, shape=(m, m))
        p_sparse = D.dot(p_sparse)

    # Possibly re-check shape after trimming
    m, n = p_sparse.shape

    # 5) Initialize r uniformly
    r = np.full(m, 1.0 / m, dtype=float)
    error_tolerance = tol / m

    # We'll run up to max_iter times
    for _ in range(max_iter):
        # ----  A) Compute q = each column normalized.  ----
        #
        #  q(i,j) = [ r(i)*p(i,j) ] / sum_i [ r(i)*p(i,j) ]
        #
        # We can do that in bulk by:
        #    temp = p(i,j) * r(i)        (elementwise multiplication by row)
        # then for each column j, divide by the sum.  We'll do:
        #    colSum = temp.sum(axis=0)
        #    q = temp.multiply(1/colSum)

        # Multiply each row of p by r(i):
        # Easiest is to do a diagonal multiply: p_r = diag(r) * p
        # but that is a normal sparse matmul. We can do a simpler elementwise multiply:
        # We'll turn r into a column for broadcast:
        #   Actually a direct SciPy trick: p_r = p_sparse.multiply(r[:, None])
        #   means each row i is scaled by r[i].
        p_r = p_sparse.multiply(r[:, None])  # shape (m,n)

        # Sum over rows => shape (1,n)
        col_sum = np.asarray(p_r.sum(axis=0)).ravel()
        # Avoid zero-division if a column is still zero
        nonzero_mask = (col_sum != 0)
        col_inv = np.zeros_like(col_sum)
        col_inv[nonzero_mask] = 1.0 / col_sum[nonzero_mask]

        # q = p_r.multiply(col_inv)
        # But col_inv is length n, so we can broadcast along columns by:
        q = p_r.multiply(col_inv)

        # ----  B) Update r by log-domain formula:  r(i) = exp( sum_j [ p(i,j)*log(q(i,j)) ] )  ----
        # We'll compute row_sums of p(i,j)*log(q(i,j))
        #
        #  1) compute log(q) in place
        #  2) do elementwise multiply p * log(q)
        #  3) sum over columns => log r(i)
        #  4) exponentiate and normalize

        # Convert q to COOrdinate so we can do q.data = log(q.data) easily
        q_coo = q.tocoo(copy=True)
        # q_coo.data might have zeros => log(0) => -inf. This is expected if q(i,j)=0.
        # We'll ignore those since p(i,j)*0 won't contribute anyway.
        # But to keep from getting NaNs, we can do a safe log.
        # We'll do:
        mask_nz = (q_coo.data > 0)
        q_coo.data[mask_nz] = np.log(q_coo.data[mask_nz])
        # The rest remain -inf (or some negative large) but won't matter once multiplied by p(i,j) = 0 or sum.

        # p * log(q) => elementwise
        # We can do p_coo = p_sparse in coo as well, but let's do it as a direct multiply if shapes match.
        # We'll turn p_sparse also into coo so we can do parallel index-wise operations.
        p_coo = p_sparse.tocoo(copy=False)

        # We want sum_{j} p(i,j)*log(q(i,j)) for each row i
        # The easiest way is:
        #   E = (p * log(q)) as a new sparse. Then sum rows.
        # But we have to ensure we only multiply matching (i,j).
        # Let's do a dictionary approach or direct approach:
        # Instead, let's do an efficient built-in: E = p.multiply(logq) => then E.sum(axis=1)

        # But p and q are both in CSR format. Actually q is in coo. Let's re-convert q_coo -> q_csr
        q_csr = q_coo.tocsr()
        # Now E = p_sparse.multiply(q_csr) is the elementwise multiply
        E = p_sparse.multiply(q_csr)
        # sum rows => shape (m,1)
        log_r1 = np.asarray(E.sum(axis=1)).ravel()
        # r1(i) = exp(log_r1(i))
        r1 = np.exp(log_r1)
        # Normalize r1
        sum_r1 = r1.sum()
        if sum_r1 > 0:
            r1 /= sum_r1

        # ----  C) Check convergence  ----
        diff = np.linalg.norm(r1 - r, 1)  # L1 or L2 norm, either is typical
        r = r1
        if diff < error_tolerance:
            break

    # ----  D) Compute channel capacity in bits  ----
    #
    #    C = sum_{i,j} r(i)*p(i,j)* log( q(i,j)/r(i) )
    #      = sum_{i,j} r(i)*p(i,j)* [log q(i,j) - log r(i)]
    #
    # We'll do this in a vectorized way:
    # 1) ratio = log(q) - log(r(i))   (row-wise broadcast)
    # 2) multiply by r(i)*p(i,j), sum
    #
    # We already have q from above. Let's re-log it in a stable manner, then subtract log(r) row by row.

    # A) log(q) again (since we overwrote q_coo with log values):
    # Actually we *already* have log q in q_csr as q_csr.data, which is log(q).
    # We'll retrieve that:
    log_q_csr = q_csr  # This contains log(q) in .data

    # B) We'll build ratio = log(q) - log(r(i))
    #    We can do that in coordinate form so we can subtract log(r[row]) easily.
    ratio_coo = log_q_csr.tocoo(copy=True)
    log_r = np.log(r)
    ratio_coo.data -= log_r[ratio_coo.row]

    # C) ratio in CSR
    ratio_csr = ratio_coo.tocsr()

    # D) Multiply by r(i)*p(i,j) => define rp = p.multiply(r row-wise)
    #    Then sum up rp*ratio
    rp = p_sparse.multiply(r[:, None])  # each row i scaled by r[i]
    # elementwise multiply => rp.multiply(ratio_csr), then sum everything
    final_mat = rp.multiply(ratio_csr)
    C_nats = final_mat.sum()
    C_bits = C_nats / math.log(2)
    C = float(C_bits)
    q = q.tocsr()
    return C, r, q


def combine_F_and_Px(F_list, Px):
    """
    Multiply an arbitrary number of F tensors with Px using sparse.einsum.

    Parameters:
      F_list : list of sparse.COO
          Each F tensor is assumed to have shape (free, common1, common2, ..., commonK)
      Px : sparse.COO
          Px is assumed to have shape (common1, common2, ..., commonK, Px_free1, ..., Px_freem)

    Returns:
      sparse.COO result of the elementwise product with broadcast along the common dimensions.

    The resulting tensor will have shape:
      (F0_free, F1_free, ..., common1, common2, ..., commonK, Px_free1, ..., Px_freem)
    """
    # Number of F tensors.
    N = len(F_list)

    # Determine number of common dimensions.
    # (Assumes all F tensors have the same number of common dims.)
    K = F_list[0].ndim - 1

    # Determine the number of free dimensions in Px (after the common ones).
    p_free = Px.ndim - K

    # We'll use letters from the alphabet.
    letters = string.ascii_lowercase

    # Use the first N letters for the free index of each F tensor.
    free_letters = letters[:N]

    # Use the next K letters for the common indices.
    common_letters = letters[N:N + K]

    # Use the next p_free letters for the free indices of Px.
    Px_free_letters = letters[N + K:N + K + p_free]

    # Build the input subscripts for each F tensor:
    # Each F tensor gets its free letter followed by the common letters.
    input_subscripts = []
    for fl in free_letters:
        input_subscripts.append(fl + common_letters)

    # For Px, its subscript is the common letters followed by its free letters.
    input_subscripts.append(common_letters + Px_free_letters)

    # Now build the output subscript.
    # In our convention, we want to keep:
    #  - The free indices from all F tensors (in order),
    #  - The common indices,
    #  - And the free indices from Px.
    output_subscript = ''.join(free_letters) + common_letters + Px_free_letters

    # Construct the full einsum string.
    einsum_str = ','.join(input_subscripts) + '->' + output_subscript
    print("Einsum string:", einsum_str)  # for debugging

    # Use sparse.einsum to compute the result.
    # The order of tensors must match the order of subscripts we built.
    return sparse.einsum(einsum_str, *(F_list + [Px]))


def compute_fundamental_matrix_with_absorption(pol_dynamics, goalstates, constraint_states, wall_list, plot_row=0,
                                               line_width=0.1):
    """Compute fundamental matrix and absorption probabilities."""
    constraint_states = list(constraint_states)
    goalstates = list(goalstates)

    # Remove wall states, constraint states, and all goal states
    states_to_remove = constraint_states + goalstates + list(wall_list)

    # Create mask for transient states (excludes walls, constraints, and goals)
    mask = np.ones(pol_dynamics.shape[0], dtype=bool)
    mask[states_to_remove] = False
    transient_indices = np.where(mask)[0]

    # Only constraint states and goal states are absorbing (not walls)
    absorbing_indices = list(constraint_states) + goalstates

    # Extract Q (transient to transient) and R (transient to absorbing)
    Q = pol_dynamics[np.ix_(transient_indices, transient_indices)]
    R = pol_dynamics[np.ix_(transient_indices, absorbing_indices)]

    # Compute fundamental matrix N = (I - Q)^(-1)
    N = np.linalg.inv(np.eye(Q.shape[0]) - Q)

    # Compute absorption probabilities B = NR
    B = np.dot(N, R)

    # Create full-size matrices
    full_occupancy_matrix = np.zeros((pol_dynamics.shape[0], pol_dynamics.shape[0]))
    B_full = np.zeros((pol_dynamics.shape[0], len(absorbing_indices)))

    # Fill in the fundamental matrix (transient to transient)
    full_occupancy_matrix[np.ix_(transient_indices, transient_indices)] = N

    # Fill in absorption probabilities (transient to absorbing)
    for i, abs_idx in enumerate(absorbing_indices):
        full_occupancy_matrix[transient_indices, abs_idx] = B[:, i]

    # Also return separate absorption matrix if needed
    B_full[transient_indices, :] = B

    # plotgrid2(full_Occupancy[plot_row, :], line_width)

    return full_occupancy_matrix


def compute_partial_sums_efficient(Q, R, t_max):
    """More efficient computation using geometric series formula"""
    if t_max == 0:
        N_partial = np.eye(Q.shape[0])
        B_partial = R
    else:
        # Compute (I - Q^(t+1)) * (I - Q)^(-1)
        Q_power_t_plus_1 = np.linalg.matrix_power(Q, t_max + 1)
        I_minus_Q_inv = np.linalg.inv(np.eye(Q.shape[0]) - Q)

        N_partial = np.dot(np.eye(Q.shape[0]) - Q_power_t_plus_1, I_minus_Q_inv)
        B_partial = np.dot(N_partial, R)

    return N_partial, B_partial

def compute_SR(pol_dynamics, gamma):
   """Compute the resolvent matrix SR = (I - γP)^(-1)"""
   return np.linalg.inv(np.eye(pol_dynamics.shape[0]) - gamma * pol_dynamics)