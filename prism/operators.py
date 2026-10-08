"""Graph matrices and their eigendecompositions (numpy, dense).

Three operators are used by the experiments of the paper:

``norm_adj``  ``D^-1/2 A D^-1/2`` with degrees clipped below at 1; the trivial
              eigenpair is the last one.  Used by the molecular GNN harness.
``norm_lap``  ``I - D^-1/2 A D^-1/2``; eigenvalues are clipped at 0.  Used by
              the transformer and BREC harness.
``comb_lap``  ``D - A``.  Used by the rebuttal experiments and Algorithm 3.

All decompositions are ``numpy.linalg.eigh`` in float64; eigenvalues ascend.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

OPERATORS = ("norm_adj", "norm_lap", "comb_lap")


def as_dense_adjacency(A) -> np.ndarray:
    """Return ``A`` (dense array or scipy sparse matrix) as a float64 ndarray."""
    if hasattr(A, "toarray"):
        A = A.toarray()
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"adjacency must be square, got shape {A.shape}")
    return A


def normalized_adjacency(A: np.ndarray) -> np.ndarray:
    """``D^-1/2 A D^-1/2`` with ``deg`` clipped below at 1 (isolated nodes)."""
    deg = A.sum(axis=0)
    d = np.maximum(deg, 1.0) ** -0.5
    return (d[:, None] * A) * d[None, :]


def normalized_laplacian(A: np.ndarray) -> np.ndarray:
    """``I - D^-1/2 A D^-1/2`` with ``deg`` clipped below at 1."""
    deg = A.sum(axis=1)
    d12 = 1.0 / np.sqrt(np.maximum(deg, 1.0))
    return np.eye(A.shape[0]) - (d12[:, None] * A) * d12[None, :]


def combinatorial_laplacian(A: np.ndarray) -> np.ndarray:
    """``D - A``."""
    return np.diag(A.sum(axis=1)) - A


def eigendecompose(A, operator: str = "norm_adj") -> Tuple[np.ndarray, np.ndarray]:
    """Full eigendecomposition ``(eigenvalues, eigenvectors)`` of an operator."""
    A = as_dense_adjacency(A)
    if operator == "norm_adj":
        lam, V = np.linalg.eigh(normalized_adjacency(A))
    elif operator == "norm_lap":
        lam, V = np.linalg.eigh(normalized_laplacian(A))
        lam = np.clip(lam, 0.0, None)
    elif operator == "comb_lap":
        lam, V = np.linalg.eigh(combinatorial_laplacian(A))
    else:
        raise ValueError(f"unknown operator {operator!r}; use one of {OPERATORS}")
    return lam, V


def column_weights(weight, lam: np.ndarray, param=None) -> np.ndarray:
    """Per-column scalar weights as functions of the adjacency eigenvalues.

    ``eigval``        ``sqrt(max(lam, 0))``
    ``inv_sqrt_lap``  ``1/sqrt(max(1 - lam, eps))``, ``eps = param or 1e-6``
    ``heat``          ``exp(-t (1 - lam))``, ``t = param or 1.0``
    """
    if weight == "eigval":
        return np.sqrt(np.maximum(lam, 0))
    if weight == "inv_sqrt_lap":
        eps = 1e-6 if param is None else param
        return 1.0 / np.sqrt(np.maximum(1.0 - lam, eps))
    if weight == "heat":
        t = 1.0 if param is None else param
        return np.exp(-t * (1.0 - lam))
    raise ValueError(f"unknown weight {weight!r}")
