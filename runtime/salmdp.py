import numpy as np
import scipy as sp
from lmdp import LMDP
from scipy.sparse import csr_matrix as csr
from scipy.sparse import csc_matrix as csc
from scipy.sparse import lil_matrix as lil
from timeit import default_timer as timer


class SALMDP:
    def __init__(self, pxa, walls, actions, target, base_cost):
        self.pxa = pxa

    # These methods convert the given policy from stochastic to deterministic
    @staticmethod
    def computeStateActionTrajectory(pol, target_state, ns, na, init_sa_vec):
        pol = SALMDP.makePolDeterministic(pol)
        curState = None
        curAction = None
        sa_vec = init_sa_vec
        s_lst = []
        a_lst = []
        target_success = False
        # At most ns iterations since no stationary policy should ever produce a trajectory longer than ns
        for i in range(ns):
            cur_s, cur_a, sa_vec = SALMDP.get_next_sa_salmdp_jointpol(pol, ns, na, sa_vec)
            s_lst.append(cur_s)
            a_lst.append(cur_a)
            if cur_s == target_state:
                target_success = True
                break

        return s_lst, a_lst, target_success

    @staticmethod
    def get_next_sa_salmdp_jointpol(policy, ns, na, current_sa_vector):
        nextvec = csr(current_sa_vector).dot(policy)
        next_sa = np.argmax(nextvec)
        next_s = next_sa // na
        next_a = next_sa % na
        return next_s, next_a, nextvec

    @staticmethod
    def get_state_and_action_from_sa_vector(sa_vec, na):
        x = sa_vec // na
        a = sa_vec % na
        return x, a

    @staticmethod
    def compute_desirability_linsys_bicg(cost_vec, P, terminal_inds, initializer):
        # Q is the standard exp(-cost) matrix we would normally give to z_iteration
        # exp(-q_terminal(x)) will be defined to be 1
        s_time = timer()
        non_terminal_inds = np.setdiff1d(np.arange(len(cost_vec)), terminal_inds)
        P_nt = P[non_terminal_inds,:][:,terminal_inds]
        P_nn = P[non_terminal_inds,:][:,non_terminal_inds]
        q_terminal = cost_vec[terminal_inds]
        q_nonterminal = cost_vec[non_terminal_inds]
        Q = sp.sparse.diags(np.exp(q_nonterminal))
        z_known = P_nt * np.exp(-q_terminal)
        init_nonterm = initializer[non_terminal_inds]
        z_nonterminal = sp.sparse.linalg.bicgstab((Q - P_nn), z_known, x0 = init_nonterm)
        z_final = np.zeros(cost_vec.size)
        z_final[non_terminal_inds] = z_nonterminal
        z_final[terminal_inds] = [np.exp(-q_terminal)]
        f_time = timer()
        return z_final, f_time - s_time

    @staticmethod
    def compute_desirability_linsys(cost_vec, P, terminal_inds):
        # Q is the standard exp(-cost) matrix we would normally give to z_iteration
        # exp(-q_terminal(x)) will be defined to be 1
        s_time = timer()
        non_terminal_inds = np.setdiff1d(np.arange(len(cost_vec)), terminal_inds)
        P_nt = P[non_terminal_inds,:][:,terminal_inds]
        # P_nn = P[np.ix_(non_terminal_inds, non_terminal_inds)]
        P_nn = P[non_terminal_inds,:][:,non_terminal_inds]
        q_terminal = cost_vec[terminal_inds]
        q_nonterminal = cost_vec[non_terminal_inds]
        Q = sp.sparse.diags(np.exp(q_nonterminal))
        z_known = P_nt*np.exp(-q_terminal)
        q_minus_pnn = (Q - P_nn)
        z_nonterminal = sp.sparse.linalg.spsolve(q_minus_pnn, z_known)
        # q_minus_pnn_mod = q_minus_pnn + sp.sparse.diags(np.random.normal(0.5,0.001,q_minus_pnn.shape[0]))
        # z_nonterminal_mod = sp.sparse.linalg.spsolve(q_minus_pnn_mod, z_known)

        z_final = np.zeros(cost_vec.size)
        z_final[non_terminal_inds] = z_nonterminal
        # z_final[terminal_inds] = [np.exp(-q_terminal)]
        z_final[terminal_inds] = 1
        f_time = timer()
        return z_final, f_time - s_time

    @staticmethod
    def computePolicy(Q, P):
        z = LMDP.computeZFunction(Q, P)
        return SALMDP.rescalePassiveWithZ(z, P), z

    @staticmethod
    def computeBlockPolicy(P, ns, na, wall_list, base_cost, target_state_action_list):
        #  If target_list == 'all' then this method will construct an all-to-all policy set.
        ntargets = len(target_state_action_list)
        big_Q = SALMDP.computeBlockQ(target_state_action_list, wall_list, ns, na, base_cost)
        nullpassive = csc(sp.sparse.identity(ns*na))
        passive_list = [P] * ntargets
        # If the goal state is in a wall, we set the passive dynamics to the nullPassive (Do nothing)
        for i, sa_ind in enumerate(target_state_action_list):
            if sa_ind//na in wall_list:
                passive_list[i] = nullpassive
        block_P = csc(sp.sparse.block_diag(passive_list))
        target_xa_block_expanded = target_state_action_list + (ns*na * np.arange(ntargets))
        big_z, compute_time = SALMDP.compute_desirability_poweriter_v2(big_Q, block_P, target_xa_block_expanded)
        return SALMDP.rescalePassiveWithZ(big_z, block_P), big_z, compute_time

    @staticmethod
    def compute_policy_set(P, ns, na, wall_list, base_cost, target_state_action_list):
        #  If target_list == 'all' then this method will construct an all-to-all policy set.
        ntargets = len(target_state_action_list)
        big_Q = SALMDP.computeBlockQ(target_state_action_list, wall_list, ns, na, base_cost)
        nullpassive = csr(sp.sparse.identity(ns*na))
        passive_list = [P] * ntargets
        big_z = np.zeros((ntargets, ns*na))
        compute_time = 0
        # If the goal state is in a wall, we set the passive dynamics to the nullPassive (Do nothing)
        for i, xa_ind in enumerate(target_state_action_list):
            if xa_ind // na in wall_list:
                passive_list[i] = nullpassive
        block_P = csr(sp.sparse.block_diag(passive_list))

        for i, xa in enumerate(target_state_action_list):
            Q = SALMDP.create_sa_negexp_costfunc(xa, wall_list, na, ns, base_cost, return_as_vector=False)
            big_z[i, :], t = SALMDP.compute_desirability_poweriter_v2(Q, P, [xa])
            compute_time += t

        big_z = big_z.flatten()
        return SALMDP.rescalePassiveWithZ(big_z, block_P), big_z

    @staticmethod
    def computeBlockPolicy_split(P, ns, na, wall_list, base_cost, target_state_action_list, num_pols_per_block):
        #  If target_list == 'all' then this method will construct an all-to-all policy set.
        ntargets = len(target_state_action_list)
        big_Q = SALMDP.computeBlockQ(target_state_action_list, wall_list, ns, na, base_cost)
        nullpassive = csr(sp.sparse.identity(ns * na))
        passive_list = [P] * ntargets
        # If the goal state is in a wall, we set the passive dynamics to the nullPassive (Do nothing)
        for i, sa_ind in enumerate(target_state_action_list):
            if sa_ind // na in wall_list:
                passive_list[i] = nullpassive
        block_P = csr(sp.sparse.block_diag(passive_list))

        big_z = SALMDP.computeBlockZFunction(big_Q, block_P)
        return SALMDP.rescalePassiveWithZ(big_z, block_P), big_z

    @staticmethod
    def getPolicyFromBlockDiag(blockpol, polnumber, ns, na):
        idxs = np.linspace(0, blockpol.shape[0], blockpol.shape[0] // (ns*na) + 1).astype(int)
        pol_inds = np.arange(idxs[polnumber], idxs[polnumber + 1])
        return blockpol[pol_inds, :][:, pol_inds]

    @staticmethod
    def computeBlockQ(target_list, wall_list, ns, na, base_cost):
        cFuncAll = []
        for i in target_list:
            # self.cFuncAll.append(self.createCostFunction(i, self.ss.wall_list, baseCost))
            if i//na not in wall_list:
                cFuncAll.append(SALMDP.create_sa_negexp_costfunc(i, wall_list, na, ns, base_cost, return_as_vector=True))
            else:
                cFuncAll.append(np.ones(ns*na))

        bigQ = sp.sparse.diags(np.array(cFuncAll).flatten())
        return bigQ


    @staticmethod
    def computePolicyWithWallsAndTarget(P, terminalstate, wall_list, basecost):
        Q = SALMDP.createCostFunction(terminalstate, wall_list, basecost)
        return SALMDP.rescalePassiveWithZ(SALMDP.compute_desirability_poweriter(Q, P), P, terminalstate)

    @staticmethod
    def compute_desirability_poweriter(Q, P, terminals, max_iter = 100):
        # Power iteration.
        s_time = timer()
        QP = csr.dot(Q, P)
        # z_old = np.ones(QP.shape[1], dtype=np.double)
        z_old = onehot(terminals, QP.shape[1])
        # diff = 10000
        for i in range(max_iter):
            z_old[terminals] = 1
            # z_old = z_old/z_old.max() # Doesn't work for coupled optimization
            z_new = QP.dot(z_old)
            # z[terminals] = 1
            # diff = np.sum(np.abs(zold-znew))
            z_old = z_new

        # znew[znew == 0] = znew[np.where(znew > 0)].min()
        f_time = timer()
        return z_new, f_time - s_time

    @staticmethod
    def compute_desirability_poweriter_v2(Q, P, terminal_inds, max_iter=None):
        # Power iteration.
        # ordering_cost = np.exp(-const_check)
        # ordering_cost[ordering_cost>0] = 1

        nsa = P.shape[0]
        # QP = csr.dot(Q, P)
        QP = Q.dot(P)
        # QP = sp.sparse.csc_matrix(csr.dot(Q, P))
        non_terminal_inds = np.setdiff1d(np.arange(nsa), terminal_inds)
        QP_nn = QP[non_terminal_inds,:][:,non_terminal_inds]
        QP_nt = QP[non_terminal_inds, :][:, terminal_inds]
        # z_old = np.ones(QP.shape[1], dtype=np.double)
        z_boundary = np.ones(len(terminal_inds))
        z_t2n = QP_nt * z_boundary  # terminal to non-terminal desirability
        z_old = z_t2n
        delta = 10000
        start_time = timer()
        if max_iter == None:
            max_iter = QP.shape[0] + 10
        for i in range(max_iter):

            # s_time = timer()
            # f_time = timer()
            # print(f_time - s_time)
            z_new = QP_nn.dot(z_old) + z_t2n   # Z-iteration:  z_n = M_nn z_n + M_nt z_t  where  M = QP
            delta = np.sum(np.abs(z_old-z_new))
            # print(delta)
            if delta < 1E-300:
                z_old = z_new
                break
            z_old = z_new

        print("Total Iterations (PowerIter): ", i+1)
        print("Final Delta: ", delta)

        z_final = np.zeros(nsa)
        z_final[non_terminal_inds] = z_old
        z_final[terminal_inds] = z_boundary
        end_time = timer()
        compute_time = end_time - start_time
        print("Power_Iter Time: ", compute_time)
        # znew[znew == 0] = znew[np.where(znew > 0)].min()
        return z_final, compute_time

    @staticmethod
    def computeBlockZFunction(Q, P):
        # Power iteration.
        QP = csr.dot(Q, P)
        zold = np.ones(QP.shape[1], dtype=np.double)
        # diff = 10000
        delta = 10000
        iter = 0
        max_iter = QP.shape[0] + 10
        while iter < max_iter:
            znew = QP.dot(zold)
            znew = znew / znew.max()
            delta = np.sum(np.abs(zold - znew))
            if delta < 1E-50:
                zold = znew
                break
            zold = znew
            iter += 1

        print("Ensemble Number of steps: ", iter)
        # znew[znew == 0] = znew[np.where(znew > 0)].min()
        return znew

    @staticmethod
    def rescalePassiveWithZ(z, p):
        z = csr(z)
        normalizer = sp.sparse.diags(np.divide(1, p.dot(z.transpose()).toarray()).squeeze(), dtype=np.double)
        scaled = p.multiply(z)
        pol = sp.dot(normalizer, scaled)
        return pol

    @staticmethod
    def makePolDeterministic(pol):
        rows = np.arange(pol.shape[0])
        cols = np.asarray(pol.argmax(axis=1))[:, 0]
        data = np.ones(cols.shape[0])

        # Fixing the problem where argmax selects ind=0 for empty rows. Set those state dynamics to identity.
        zero_row_inds = np.argwhere(pol.astype(bool).sum(axis=1) == 0)[:, 0]
        cols[zero_row_inds] = rows[zero_row_inds]

        detpol = csr((data, (rows, cols)), shape=(pol.shape[0], pol.shape[1]))
        return detpol

    @staticmethod
    def create_sa_costfunc(target_state_action, wall_list, numberOfActions, numberOfStates, base_cost, wall_cost=1E6):
        na = numberOfActions
        ns = numberOfStates
        cfunction = np.ones(ns * na) * base_cost
        wallLIs = np.array([list(range(i * na, (i * na) + na)) for i in wall_list]).flatten()  # Walls cover all state-action pairs that for all wall-states
        cfunction[target_state_action] = 0
        if wallLIs.size:
            cfunction[wallLIs] = wall_cost  # Order matters here, we want to overwrite goal states that are in walls.

        return cfunction

    @staticmethod
    def create_sa_negexp_costfunc(target_state_action, wall_list, numberOfActions, numberOfStates, baseCost, wall_cost=0, return_as_vector=False):
        na = numberOfActions
        ns = numberOfStates
        cfunction = np.ones(ns * na) * np.exp(-baseCost)
        wallLIs = np.array([list(range(i * na, (i * na) + na)) for i in wall_list]).flatten()  # Walls cover all state-action pairs that for all wall-states
        cfunction[target_state_action] = 1
        if wallLIs.size:
            cfunction[wallLIs] = wall_cost  # Order matters here, we want to overwrite goal states that are in walls.

        if return_as_vector:
            return cfunction
        return sp.sparse.diags(cfunction)


def onehot(inds, size):
    oh = np.zeros(size)
    oh[inds] = 1
    return oh