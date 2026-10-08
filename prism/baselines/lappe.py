"""Laplacian positional encoding without canonicalization (LapPE / SignNet input).

LapPE feeds the raw eigenvectors of the normalized Laplacian to the model.  The
sign ambiguity is handled on the model side: by random sign flips during
training (LapPE) or by a sign-invariant network (SignNet).  Only the pure
preprocessing is provided here; the networks live with the experiments.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .. import operators


def lap_pe(A, k: int) -> np.ndarray:
    """Eigenvectors 1..k of ``I - D^-1/2 A D^-1/2`` as an (n, k) float32 array.

    Eigenvectors come from ``numpy.linalg.eig``, sorted by ascending
    eigenvalue, with signs left as the solver returns them; the trivial
    eigenvector (column 0) is skipped and the result is zero padded when the
    graph has fewer than ``k`` non-trivial eigenvectors.
    """
    A = operators.as_dense_adjacency(A)
    n = A.shape[0]
    deg = A.sum(axis=0)
    d = np.maximum(deg, 1.0) ** -0.5
    L = np.eye(n) - (d[:, None] * A) * d[None, :]
    eig_val, eig_vec = np.linalg.eig(L)
    idx = eig_val.argsort()
    eig_vec = np.real(eig_vec[:, idx])
    out = eig_vec[:, 1:k + 1].astype(np.float32)
    if n <= k:
        out = np.concatenate([out, np.zeros((n, k - n + 1), dtype=np.float32)], axis=-1)
    return out


def random_sign_flip(pe: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Multiply each column of ``pe`` by an independent random sign (LapPE training)."""
    rng = np.random.default_rng() if rng is None else rng
    signs = np.where(rng.random(pe.shape[1]) >= 0.5, 1.0, -1.0).astype(pe.dtype)
    return pe * signs[np.newaxis, :]
