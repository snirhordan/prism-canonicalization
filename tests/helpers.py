"""Small utilities shared by the tests (no fixtures)."""

import numpy as np

from prism import gf2
from prism.core import detect_multiplicities


def er_adjacency(n, p, seed):
    rng = np.random.default_rng(seed)
    U = rng.random((n, n)) < p
    A = np.triu(U, 1)
    return (A | A.T).astype(np.float64)


def is_connected(A):
    n = A.shape[0]
    seen = {0}
    stack = [0]
    while stack:
        v = stack.pop()
        for u in np.nonzero(A[v])[0]:
            if int(u) not in seen:
                seen.add(int(u))
                stack.append(int(u))
    return len(seen) == n


def is_generic(canon, A, decimals=6):
    """Simple spectrum and injective absolute-value signature for ``canon``'s operator."""
    lam, V = canon.eigendecompose(A)
    if lam.size == 0 or max(detect_multiplicities(lam)) > 1:
        return False
    gaps = np.diff(lam)
    if gaps.min() < 1e-6:
        return False
    return gf2.is_injective_signature(V, decimals)


def generic_graph(configs, n=13, p=0.4, start=0):
    """First connected ER graph that is generic for every canonicalizer in ``configs``."""
    for seed in range(start, start + 500):
        A = er_adjacency(n, p, seed)
        if is_connected(A) and all(is_generic(c, A) for c in configs):
            return A
    raise RuntimeError("no generic graph found")


def permute(A, perm):
    return A[np.ix_(perm, perm)]


def scramble_eigenspaces(V, lam, rng, rtol=1e-10):
    """Apply a Haar-random orthogonal matrix inside every eigenspace."""
    V = V.copy()
    col = 0
    for m in detect_multiplicities(lam, rtol):
        if m > 1:
            Q, R = np.linalg.qr(rng.standard_normal((m, m)))
            Q = Q * np.sign(np.diag(R))
            V[:, col:col + m] = V[:, col:col + m] @ Q
        col += m
    return V
