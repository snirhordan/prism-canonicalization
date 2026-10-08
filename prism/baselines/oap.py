"""Orthogonalized Axis Projection (OAP), a sign and basis canonicalization baseline.

Adapted from the reference implementation of Ma, Wang, Lim, Jegelka and Wang,
"A Canonicalization Perspective on Invariant and Equivariant Learning"
(NeurIPS 2024), https://github.com/PKU-ML/canonicalization (MIT License,
Copyright (c) 2024 PKU-ML).

``oap_sign`` and ``oap_basis`` are the upstream routines, unchanged except for
dtype/device-aware constants and parameterized rounding.  The wrapper
:meth:`OAP.pos_enc` follows the positional-encoding recipe of the molecular GNN
harness.  Requires PyTorch.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .. import operators
from .map import _torch, apply_blocks


def hash_tensor(u, i: int) -> float:
    """Hash of ``(u_i, {u_j : j != i})``: ``sum(u^3) + u_i`` to 5 significant digits."""
    torch = _torch()
    hash_value = torch.sum(u ** 3) + u[i]
    return float('%.5g' % hash_value)


def oap_sign(U, decimals: int = 6):
    """Eliminate the sign ambiguity of the columns of ``U`` (modified in place).

    ``decimals`` rounds the projection norms and the zero test.
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
            if torch.linalg.vector_norm(u_0).round(decimals=decimals) != 0:
                x = X[j]
                flag = False
                break
        if flag:  # violates sign assumption, skip
            continue
        u = u.view(n)
        u_0 /= torch.abs(u @ x)
        U[:, i] = u_0
    return U


def orthogonalize(U):
    """QR orthogonalization with positive diagonal of R."""
    torch = _torch()
    Q, R = torch.linalg.qr(U)
    S = torch.sign(torch.diag(R))
    return Q * S


def oap_basis(U_i):
    """Eliminate the basis ambiguity of an eigenspace ``U_i`` of shape [n, d].

    Raises ``AssertionError`` when the OAP assumptions are violated.
    """
    torch = _torch()
    n, d = U_i.shape
    E = torch.eye(n, dtype=U_i.dtype, device=U_i.device)
    J = torch.ones(n, dtype=U_i.dtype, device=U_i.device)
    P = U_i @ U_i.T
    Pe = [hash_tensor(P[:, j], j) for j in range(n)]
    Pe = [j for j in enumerate(Pe)]
    Pe.sort(key=lambda x: x[1])
    indices = [j[0] for j in Pe]
    lengths = [j[1] for j in Pe]
    _, counts = np.unique(lengths, return_counts=True)
    k = len(counts)
    assert k >= d
    X = torch.zeros([k, n], dtype=U_i.dtype, device=U_i.device)
    step = -1
    for i in range(1, k + 1):
        x = torch.zeros(n, dtype=U_i.dtype, device=U_i.device)
        for _ in range(counts[-i]):
            x += E[indices[step]]
            step -= 1
        X[i - 1] = x + 10 * J
    u_span = torch.empty([n, 0], dtype=U_i.dtype, device=U_i.device)
    current_rank = 0
    for i in range(k):
        u_i = P @ X[i]
        u_i = torch.nn.functional.normalize(u_i, dim=0)
        if torch.isclose(torch.linalg.vector_norm(u_i),
                         torch.tensor(0., dtype=U_i.dtype, device=U_i.device)):
            continue
        u_span_tmp = torch.cat([u_span, u_i.unsqueeze(dim=1)], dim=1)
        if torch.linalg.matrix_rank(u_span_tmp) == current_rank + 1:
            u_span = u_span_tmp
            current_rank += 1
            if current_rank == d:
                break
    assert current_rank == d
    return orthogonalize(u_span)


class OAP:
    """OAP canonicalizer.

    Parameters
    ----------
    eig_decimals : decimals used to round eigenvalues before grouping them.
    sign_decimals : decimals used in the sign step.
    use_eig_val : scale columns by ``sqrt(max(lambda, 0))`` in :meth:`pos_enc`.
    k : default number of columns of :meth:`pos_enc`.
    """

    def __init__(self, eig_decimals: int = 14, sign_decimals: int = 6,
                 use_eig_val: bool = False, k: Optional[int] = None):
        self.eig_decimals = eig_decimals
        self.sign_decimals = sign_decimals
        self.use_eig_val = use_eig_val
        self.k = k

    @classmethod
    def from_profile(cls, profile: str = "gnn", **kwargs) -> "OAP":
        """``gnn``: molecular GNN harness.  ``brec``: transformer / BREC harness."""
        if profile == "gnn":
            return cls(eig_decimals=14, sign_decimals=6, use_eig_val=False, **kwargs)
        if profile == "brec":
            return cls(eig_decimals=8, sign_decimals=8, use_eig_val=False, **kwargs)
        raise ValueError("profile must be 'gnn' or 'brec'")

    def _fns(self):
        sd = self.sign_decimals
        return (lambda U: oap_sign(U, sd), oap_basis)

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
        """Positional encoding as in the molecular GNN harness (see :meth:`MAP.pos_enc`)."""
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
