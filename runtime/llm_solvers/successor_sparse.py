"""Discounted successor matrix using sparse LU (Codex).

The transition operator and factorization are sparse. The result is generally
dense, and the tutorial needs all rows for interactive start-state selection.
Solve in small column blocks to bound temporary storage.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def successor_matrix(policy_kernel, gamma, block_size=32):
    if not 0 < gamma < 1:
        raise ValueError("gamma must be between zero and one")
    P = sparse.csc_matrix(policy_kernel)
    n = P.shape[0]
    if P.shape != (n, n):
        raise ValueError("policy kernel must be square")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    lu = splu(sparse.eye(n, format="csc") - gamma * P)
    result = np.empty((n, n))
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        rhs = np.zeros((n, end - start), order="F")
        rhs[np.arange(start, end), np.arange(end - start)] = 1
        result[:, start:end] = lu.solve(rhs)
    return result
