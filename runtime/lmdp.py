import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix as csr

class LMDP:
    def __init__(self, passive, costfunction, make_detministic=True):
        self.Q = costfunction
        self.P = passive
        self.z = self.computeZFunction(self.Q, self.P)
        self.u = self.computePolicy(self.z, self.P)
        if make_detministic:
            self.u = self.makePolDeterministic(self.u)

    @staticmethod
    def computeBlockPolicy(P, ns, wall_list, base_cost, target_list='all'):
        #  If target_list == 'all' then this method will construct an all-to-all policy set.
        if target_list == 'all':
            target_list = list(range(ns))
        ntargets = len(target_list)
        big_Q = LMDP.computeBlockQ(target_list, wall_list, ns, base_cost)
        nullpassive = csr(sp.sparse.identity(ns))
        passive_list = [P] * ntargets
        # If the goal state is in a wall, we set the passive dynamics to the nullPassive (Do nothing)
        for i in target_list:
            if i in wall_list:
                passive_list[i] = nullpassive
        block_P = csr(sp.sparse.block_diag(passive_list))

        big_z = LMDP.computeBlockZFunction(big_Q, block_P)
        return LMDP.rescalePassiveWithZ(big_z, block_P), big_z

    @staticmethod
    def getPolicyFromBlockDiag(blockpol, polnumber, ns):
        idxs = np.linspace(0, blockpol.shape[0], blockpol.shape[0] / ns + 1).astype(int)
        i = polnumber
        return blockpol[np.ix_(range(idxs[i], idxs[i + 1]), range(idxs[i], idxs[i + 1]))]

    @staticmethod
    def computeBlockQ(target_list, wall_list, ns, base_cost):
        cFuncAll = []
        for i in target_list:
            # self.cFuncAll.append(self.createCostFunction(i, self.ss.wall_list, baseCost))
            if i not in wall_list:
                cFuncAll.append(LMDP.createCostFunction(i, wall_list, ns, base_cost))
            else:
                cFuncAll.append(np.ones(ns))

        bigQ = sp.sparse.diags(np.array(cFuncAll).flatten())
        return bigQ

    @staticmethod
    def computeBlockZFunction(Q, P):
        # Power iteration.
        QP = csr.dot(Q, P)
        zold = np.ones(QP.shape[1], dtype=np.double)
        # diff = 10000
        iter = 0
        while iter < 100:
            znew = QP.dot(zold)
            znew = znew / znew.max()
            # diff = np.sum(np.abs(zold - znew))
            zold = znew
            iter += 1

        return znew

    @staticmethod
    def computePolicy(Q, P):
        return LMDP.rescalePassiveWithZ(LMDP.computeZFunction(Q, P), P)

    @staticmethod
    def computePolicyWithWallsAndTarget(P, terminalstate, wall_list, basecost):
        Q = LMDP.createCostFunction(terminalstate, wall_list, basecost)
        return LMDP.rescalePassiveWithZ(LMDP.computeZFunction(Q, P), P)

    @staticmethod
    def computeZFunction(Q, P):
        # Power iteration.
        QP = csr.dot(Q, P)
        zold = np.ones(QP.shape[1], dtype=np.double)
        iter = 0
        while iter < 100:
            znew = QP.dot(zold)
            znew = znew/znew.max()
            zold = znew
            iter += 1

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
        cols = np.asarray(pol.argmax(axis=1))[:,0]
        data = np.ones(cols.shape[0])

        # Fixing the problem where argmax selects ind=0 for empty rows. Set those state dynamics to identity.
        zero_row_inds = np.argwhere(pol.astype(bool).sum(axis=1)==0)[:, 0]
        cols[zero_row_inds] = rows[zero_row_inds]

        detpol = csr((data, (rows, cols)), shape=(pol.shape[0], pol.shape[1]))
        return detpol

    @staticmethod
    def createCostFunction(termLISet, wall_list, numberOfStates, baseCost):
        cfunction = np.ones(numberOfStates) * np.exp(-baseCost)
        cfunction[termLISet] = 1
        cfunction[wall_list] = 0  # Order matters here, we want to overwrite goal states that are in walls.
        return cfunction


def onehot(inds, size):
    oh = np.zeros(size)
    oh[inds] = 1
    return oh