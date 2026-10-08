"""Maximal Axis Projection (MAP), a sign and basis canonicalization baseline.

Adapted from the reference implementation of Ma et al., "A Canonicalization
Perspective on Invariant and Equivariant Learning" (NeurIPS 2024),
https://github.com/PKU-ML/canonicalization (MIT License, Copyright (c) 2024
PKU-ML).  MAP itself is introduced in Ma, Wang and Wang, "Laplacian
Canonization: A Minimalist Approach to Sign and Basis Invariant Spectral
Embedding" (NeurIPS 2023).

``unique_sign`` and ``unique_basis`` are the upstream routines, unchanged
except for dtype/device-aware constants and parameterized rounding.  The
wrapper :meth:`MAP.pos_enc` follows the positional-encoding recipe of the
molecular GNN harness.  Requires PyTorch.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .. import operators


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("prism.baselines.MAP requires PyTorch") from exc
    return torch


def unique_sign(U, decimals: int = 14, zero_decimals: int = 12):
    """Eliminate the sign ambiguity of the columns of ``U`` (modified in place).

    :param U: Tensor of shape [n, d]; each column is an eigenvector.
    :return: Tensor of shape [n, d].
    """
    torch = _torch()
    n, d = U.shape
    for i in range(d):
        u = U[:, i].view(n, 1)
        P = u @ u.T.view(1, n)
        E = torch.eye(n, dtype=U.dtype, device=U.device)
        J = torch.ones(n, dtype=U.dtype, device=U.device)
        Pe = [torch.linalg.vector_norm(P[:, j]).round(decimals=decimals).item()
              for j in range(n)]
        Pe = [j for j in enumerate(Pe)]
        Pe.sort(key=lambda x: x[1])
        indices = [j[0] for j in Pe]
        lengths = [j[1] for j in Pe]
        _, counts = np.unique(lengths, return_counts=True)
        step = 0
        X = torch.zeros([len(counts), n], dtype=U.dtype, device=U.device)
        for j in range(len(counts)):
            for _ in range(counts[j]):
                X[j] += E[indices[step]]
                step += 1
            X[j] += 10 * J
        u_0, x = torch.zeros(n, dtype=U.dtype, device=U.device), torch.zeros(
            n, dtype=U.dtype, device=U.device)
        flag = True
        for j in range(len(counts)):
            u_0 = P @ X[j]
            if torch.linalg.vector_norm(u_0).round(decimals=zero_decimals) != 0:
                x = X[j]
                flag = False
                break
        if flag:  # violates sign assumption, skip
            continue
        u = u.view(n)
        u_0 /= torch.abs(u @ x)
        U[:, i] = u_0
    return U


def _orthogonalize(U):
    torch = _torch()
    Q, R = torch.linalg.qr(U)
    return Q


def find_complementary_space(U, u_span):
    """Orthogonal complement of ``u_span`` inside the column space of ``U``."""
    torch = _torch()
    n, d = U.shape
    s = u_span.shape[1]
    u_base = u_span.clone()
    for j in range(d):
        i = u_base.shape[1]
        u_j = U[:, j].unsqueeze(dim=1)
        u_temp = torch.cat([u_base, u_j], dim=1)
        if torch.linalg.matrix_rank(u_temp) == i + 1:
            u_base = u_temp
        if u_base.shape[1] == d:
            break
    u_base = _orthogonalize(u_base)
    return u_base[:, s:d]


def unique_basis(U_i, decimals: int = 14):
    """Eliminate the basis ambiguity of an eigenspace ``U_i`` of shape [n, d].

    Raises ``AssertionError`` when the MAP assumptions are violated.
    """
    torch = _torch()
    n, d = U_i.shape
    E = torch.eye(n, dtype=U_i.dtype, device=U_i.device)
    J = torch.ones(n, dtype=U_i.dtype, device=U_i.device)
    P = U_i @ U_i.T
    Pe = [torch.linalg.vector_norm(P[:, j]).round(decimals=decimals).item()
          for j in range(n)]
    Pe = [j for j in enumerate(Pe)]
    Pe.sort(key=lambda x: x[1])
    indices = [j[0] for j in Pe]
    lengths = [j[1] for j in Pe]
    _, counts = np.unique(lengths, return_counts=True)
    assert len(counts) >= d  # basis assumption 1
    X = torch.zeros([d, n], dtype=U_i.dtype, device=U_i.device)
    step = -1
    for i in range(1, d + 1):
        x = torch.zeros(n, dtype=U_i.dtype, device=U_i.device)
        for _ in range(counts[-i]):
            x += E[indices[step]]
            step -= 1
        X[i - 1] = x + 10 * J
    U_0 = torch.zeros([n, d], dtype=U_i.dtype, device=U_i.device)
    u_span = torch.empty([n, 0], dtype=U_i.dtype, device=U_i.device)
    u_perp = U_i.clone()
    for i in range(d):
        P_perp = u_perp @ u_perp.T
        u_i = P_perp @ X[i]
        assert torch.linalg.vector_norm(u_i) != 0  # basis assumption 2
        u_i = torch.nn.functional.normalize(u_i, dim=0)
        U_0[:, i] = u_i
        u_span = torch.cat([u_span, u_i.unsqueeze(dim=1)], dim=1)
        u_perp = find_complementary_space(U_i, u_span)
    return U_0


def apply_blocks(U, mult, offset, sign_fn, basis_fn):
    """Canonicalize ``U`` block-wise: simple eigenspaces by sign, others by basis.

    ``mult`` holds the multiplicities of consecutive eigenspaces starting at
    column ``offset``.  A block whose basis algorithm fails is left unchanged.
    """
    torch = _torch()
    ind = torch.cat([torch.LongTensor([0]), torch.cumsum(mult, dim=0)]) + offset
    for i in range(mult.shape[0]):
        if mult[i] == 1:
            U[:, ind[i]:ind[i + 1]] = sign_fn(U[:, ind[i]:ind[i + 1]])
    for i in range(mult.shape[0]):
        if mult[i] == 1:
            continue
        try:
            U[:, ind[i]:ind[i + 1]] = basis_fn(U[:, ind[i]:ind[i + 1]])
        except Exception:  # assumption violated, skip the block
            continue
    return U


class MAP:
    """MAP canonicalizer.

    Parameters
    ----------
    eig_decimals : decimals used to round eigenvalues before grouping them.
    sign_decimals : decimals used for the projection norms in the sign step.
    use_eig_val : scale columns by ``sqrt(max(lambda, 0))`` in :meth:`pos_enc`.
    k : default number of columns of :meth:`pos_enc`.
    """

    def __init__(self, eig_decimals: int = 14, sign_decimals: int = 14,
                 use_eig_val: bool = True, k: Optional[int] = None):
        self.eig_decimals = eig_decimals
        self.sign_decimals = sign_decimals
        self.use_eig_val = use_eig_val
        self.k = k

    @classmethod
    def from_profile(cls, profile: str = "gnn", **kwargs) -> "MAP":
        """``gnn``: molecular GNN harness.  ``brec``: transformer / BREC harness."""
        if profile == "gnn":
            return cls(eig_decimals=14, sign_decimals=14, use_eig_val=True, **kwargs)
        if profile == "brec":
            return cls(eig_decimals=14, sign_decimals=14, use_eig_val=False, **kwargs)
        raise ValueError("profile must be 'gnn' or 'brec'")

    def _fns(self):
        sd = self.sign_decimals
        return (lambda U: unique_sign(U, sd), lambda U: unique_basis(U, sd))

    def canonicalize(self, eigenvectors: np.ndarray, eigenvalues: np.ndarray) -> np.ndarray:
        """Canonicalize all given columns; eigenvalues group the eigenspaces."""
        torch = _torch()
        U = torch.from_numpy(np.array(eigenvectors, copy=True)).double()
        E = torch.from_numpy(np.array(eigenvalues, copy=True)).double().round(
            decimals=self.eig_decimals)
        _, mult = torch.unique(E, return_counts=True)
        sign_fn, basis_fn = self._fns()
        apply_blocks(U, mult, 0, sign_fn, basis_fn)
        return U.numpy().astype(np.asarray(eigenvectors).dtype)

    def pos_enc(self, A, k: Optional[int] = None) -> np.ndarray:
        """Positional encoding as in the molecular GNN harness.

        Eigendecomposes ``D^-1/2 A D^-1/2`` with ``torch.linalg.eigh``, drops
        the trivial eigenpair, canonicalizes the last ``k`` columns and keeps
        them (scaled by ``sqrt(max(lambda, 0))`` if ``use_eig_val``).  Returns
        an (n, k) float32 array padded with zero columns.
        """
        torch = _torch()
        k = self.k if k is None else k
        A = torch.from_numpy(operators.normalized_adjacency(operators.as_dense_adjacency(A)))
        n = A.shape[0]
        E, U = torch.linalg.eigh(A)
        E, U = E.round(decimals=self.eig_decimals)[:-1], U[:, :-1]
        dim = min(n - 1, k)
        _, mult = torch.unique(E[-dim:], return_counts=True)
        sign_fn, basis_fn = self._fns()
        apply_blocks(U, mult, max(n - 1 - k, 0), sign_fn, basis_fn)
        if self.use_eig_val:
            Lambda = torch.nn.ReLU()(torch.diag(E))
            U = U @ torch.sqrt(Lambda)
        if n - 1 < k:
            U = torch.cat([U, torch.zeros([n, k - n + 1], dtype=U.dtype)], dim=-1)
        return U[:, -k:].numpy().astype(np.float32)
