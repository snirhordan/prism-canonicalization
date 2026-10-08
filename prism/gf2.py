"""PRiSM sign canonicalization: Partition -> Refine -> Solve -> Match.

Implements the base procedure of the paper (Algorithm 1) for eigenvector
matrices of simple-spectrum graphs, together with the ``FastSign`` shortcut
(Algorithm 2) that applies when the absolute-value signature is injective.

Conventions
-----------
* ``DECIMALS = 8`` is the rounding precision used for the absolute-value
  signature in Partition, for the support test ``|U_{v,j}| > 0`` in Solve,
  and for the elementwise-product comparisons in Refine.
* Classes and the final Match sort use DESCENDING lexicographic order.  This
  reproduces the block order of the worked example of the paper.  The sign
  vector produced by Solve does not depend on the class numbering, because
  Solve visits classes in the order given by a sign- and
  permutation-invariant key (:func:`_class_sort_key`).
* Sign vectors are GF(2) bit vectors: ``s[j] = 1`` means column ``j`` is
  negated.

Refine deviates from the literal pseudocode in one respect: the multiset of
pairwise products is taken over the members of the current class only, which
is the object certified by the proof (multiset stability inside each class).
The literal all-of-[n] variant would split classes that the worked example of
the paper keeps together.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

DECIMALS = 8


# =============================================================================
# GF(2) linear algebra primitives
# =============================================================================

def gf2_rref(rows: List[np.ndarray], ncols: int) -> Tuple[List[np.ndarray], List[int]]:
    """Reduced row-echelon form of GF(2) row vectors (Gauss-Jordan).

    Returns the nonzero rows of the RREF and their pivot columns.
    """
    M = [np.asarray(r, dtype=np.uint8).copy() for r in rows]
    n = len(M)
    pivots: List[int] = []
    pivot_row = 0
    for col in range(ncols):
        sel = None
        for i in range(pivot_row, n):
            if M[i][col] == 1:
                sel = i
                break
        if sel is None:
            continue
        M[pivot_row], M[sel] = M[sel], M[pivot_row]
        for i in range(n):
            if i != pivot_row and M[i][col] == 1:
                M[i] = M[i] ^ M[pivot_row]
        pivots.append(col)
        pivot_row += 1
        if pivot_row == n:
            break
    basis = [M[i] for i in range(pivot_row)]
    return basis, pivots


def gf2_null_space(rows: List[np.ndarray], ncols: int) -> List[np.ndarray]:
    """Basis of ``{x : rows . x = 0}`` over GF(2).

    This is the parity-check matrix ``T`` with ``ker(T)`` equal to the span of
    ``rows``' orthogonal complement.
    """
    basis, pivots = gf2_rref(rows, ncols) if rows else ([], [])
    pivot_set = set(pivots)
    free_cols = [c for c in range(ncols) if c not in pivot_set]
    null_basis = []
    for f in free_cols:
        v = np.zeros(ncols, dtype=np.uint8)
        v[f] = 1
        for row, piv in zip(basis, pivots):
            if row[f] == 1:
                v[piv] = 1
        null_basis.append(v)
    return null_basis


def gf2_solve_affine(E_rows: List[np.ndarray], f_bits: List[int],
                     ncols: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Solve ``E s = f`` over GF(2): a particular solution and a basis of ker(E)."""
    if E_rows:
        aug = [
            np.concatenate([np.asarray(r, dtype=np.uint8), [np.uint8(int(b) & 1)]])
            for r, b in zip(E_rows, f_bits)
        ]
    else:
        aug = []
    basis, pivots = gf2_rref(aug, ncols + 1)
    if ncols in pivots:
        raise ValueError(
            "inconsistent GF(2) system E s = f; constraints derived from a real "
            "eigenvector matrix are always consistent, so this signals a bug"
        )
    s0 = np.zeros(ncols, dtype=np.uint8)
    e_part_rows = []
    for row, piv in zip(basis, pivots):
        s0[piv] = row[ncols]
        e_part_rows.append(row[:ncols])
    kernel_basis = gf2_null_space(e_part_rows, ncols)
    return s0, kernel_basis


def gf2_lex_min_in_coset(s0: np.ndarray, kernel_basis: List[np.ndarray],
                         ncols: int) -> np.ndarray:
    """Lex-smallest element of the coset ``s0 + span(kernel_basis)`` in O(k^2)."""
    basis, pivots = gf2_rref(kernel_basis, ncols) if kernel_basis else ([], [])
    s = s0.copy()
    for row, piv in zip(basis, pivots):
        if s[piv] == 1:
            s = s ^ row
    return s


class GF2System:
    """Incrementally assembled system ``E s = f`` keeping independent rows only.

    A row that lies in the span of the rows kept so far is dropped, whatever
    its right-hand side (the "if r not in rowspace(E): append" step of the
    Solve pseudocode).  The first class in the canonical class order thus wins
    a contested direction.
    """

    def __init__(self, ncols: int):
        self.ncols = ncols
        self._pivot_rows: List[Tuple[np.ndarray, int, int]] = []
        self.kept_rows: List[Tuple[np.ndarray, int]] = []
        self.rhs_conflicts_dropped = 0

    def add_row(self, row: np.ndarray, rhs: int) -> bool:
        row = np.asarray(row, dtype=np.uint8).copy()
        rhs = int(rhs) & 1
        for prow, prhs, pcol in self._pivot_rows:
            if row[pcol] == 1:
                row = row ^ prow
                rhs ^= prhs
        nz = np.nonzero(row)[0]
        if nz.size == 0:
            if rhs == 1:
                self.rhs_conflicts_dropped += 1
            return False
        pcol = int(nz[0])
        self._pivot_rows.append((row, rhs, pcol))
        self.kept_rows.append((row.copy(), rhs))
        return True

    def finalize(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        if not self.kept_rows:
            return (
                np.zeros(self.ncols, dtype=np.uint8),
                [np.eye(self.ncols, dtype=np.uint8)[i] for i in range(self.ncols)],
            )
        rows = [r for r, _ in self.kept_rows]
        rhs = [b for _, b in self.kept_rows]
        return gf2_solve_affine(rows, rhs, self.ncols)


# =============================================================================
# Stage 1: Partition
# =============================================================================

def partition_stage(U: np.ndarray, decimals: int = DECIMALS) -> np.ndarray:
    """Group vertices by ``sigma(v) = |U[v]|`` rounded to ``decimals``.

    Returns ``class_id`` (length n); classes are numbered in descending
    lexicographic order of sigma.
    """
    n, k = U.shape
    sigma = np.round(np.abs(U), decimals)
    seen = {}
    raw_id = np.empty(n, dtype=np.int64)
    for v in range(n):
        key = tuple(sigma[v].tolist())
        if key not in seen:
            seen[key] = len(seen)
        raw_id[v] = seen[key]
    unique_keys_desc = sorted(seen.keys(), reverse=True)
    remap = {seen[key]: i for i, key in enumerate(unique_keys_desc)}
    return np.array([remap[c] for c in raw_id], dtype=np.int64)


# =============================================================================
# Stage 2: Refine
# =============================================================================

def refine_stage(U: np.ndarray, initial_partition: np.ndarray,
                 decimals: int = DECIMALS) -> np.ndarray:
    """Spectral-WL refinement scoped to the members of each class.

    Within each class, vertices whose sorted multisets of products
    ``{U_v * U_u : u in class}`` differ are split; repeated until stable.
    """
    n, k = U.shape
    color = initial_partition.copy()
    changed = True
    while changed:
        changed = False
        class_to_verts: dict = defaultdict(list)
        for v in range(n):
            class_to_verts[int(color[v])].append(v)
        new_color = color.copy()
        next_free_id = int(color.max()) + 1 if n > 0 else 0
        for c in sorted(class_to_verts.keys()):
            verts = class_to_verts[c]
            if len(verts) <= 1:
                continue
            sig_of = {}
            for v in verts:
                entries = sorted(
                    tuple(np.round(U[v] * U[u], decimals).tolist()) for u in verts
                )
                sig_of[v] = tuple(entries)
            distinct_sigs = sorted(set(sig_of.values()))
            if len(distinct_sigs) <= 1:
                continue
            changed = True
            sig_to_new_id = {s: (c if i == 0 else next_free_id + i - 1)
                             for i, s in enumerate(distinct_sigs)}
            next_free_id += len(distinct_sigs) - 1
            for v in verts:
                new_color[v] = sig_to_new_id[sig_of[v]]
        color = new_color
    unique_ids = sorted(set(color.tolist()))
    remap = {old: i for i, old in enumerate(unique_ids)}
    return np.array([remap[c] for c in color.tolist()], dtype=np.int64)


# =============================================================================
# Stage 3: Solve
# =============================================================================

def _class_sort_key(U: np.ndarray, verts: List[int], decimals: int) -> tuple:
    """Sign- and permutation-invariant key fixing the class visiting order.

    Made of the class' absolute-value signature and the sorted multiset of
    within-class products, both unchanged by column sign flips and by vertex
    relabeling.
    """
    row0 = U[verts[0]]
    sigma = tuple(np.round(np.abs(row0), decimals).tolist())
    prod_multiset = tuple(sorted(
        tuple(np.round(U[v] * U[u], decimals).tolist())
        for v in verts for u in verts
    ))
    return (sigma, prod_multiset)


def solve_stage(U: np.ndarray, classes: np.ndarray,
                decimals: int = DECIMALS) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Assemble ``E s = f`` from the partition and solve it.

    For each class ``C``: ``K`` = support columns, ``A`` = sign-bit vectors of
    the rows of ``U[C, K]``, ``T`` = parity-check matrix of the span of
    ``{a + a0}``, ``b = T a0``; every row of ``T`` (embedded in k coordinates)
    that is not in the row space of ``E`` is appended to ``(E, f)``.  The
    result is the lex-minimal element of the solution coset.

    Returns ``(s, kernel_basis)``.
    """
    n, k = U.shape
    class_to_verts = defaultdict(list)
    for v in range(n):
        class_to_verts[int(classes[v])].append(v)

    ordered_classes = sorted(
        class_to_verts.values(),
        key=lambda verts: _class_sort_key(U, verts, decimals),
        reverse=True,
    )

    system = GF2System(k)
    for verts in ordered_classes:
        row0 = U[verts[0]]
        support = np.round(np.abs(row0), decimals) > 0
        K = np.nonzero(support)[0]
        if K.size == 0:
            continue
        A = [(U[v, K] < 0).astype(np.uint8) for v in verts]
        a0 = A[0]
        diffs = [a ^ a0 for a in A]
        T_local = gf2_null_space(diffs, ncols=K.size)
        for row_local in T_local:
            b = int(np.dot(row_local.astype(np.int64), a0.astype(np.int64)) % 2)
            row_full = np.zeros(k, dtype=np.uint8)
            row_full[K] = row_local
            system.add_row(row_full, b)

    s0, kernel_basis = system.finalize()
    s = gf2_lex_min_in_coset(s0, kernel_basis, k)
    return s, kernel_basis


# =============================================================================
# Stage 4: Match
# =============================================================================

def match_stage(U_signed: np.ndarray, decimals: int = DECIMALS) -> Tuple[np.ndarray, np.ndarray]:
    """Sort rows in descending lexicographic order; return (matrix, row_order)."""
    n = U_signed.shape[0]
    keys = [tuple(np.round(U_signed[v], decimals).tolist()) for v in range(n)]
    order = np.array(sorted(range(n), key=lambda v: keys[v], reverse=True))
    return U_signed[order], order


# =============================================================================
# FastSign (Algorithm 2)
# =============================================================================

def is_injective_signature(U: np.ndarray, decimals: int = DECIMALS) -> bool:
    """True iff all rows of ``|U|`` are distinct after rounding."""
    sigma = np.round(np.abs(U), decimals)
    keys = {tuple(row.tolist()) for row in sigma}
    return len(keys) == U.shape[0]


def fastsign(U: np.ndarray, decimals: int = DECIMALS) -> np.ndarray:
    """FastSign: per-column sign scan, valid iff the signature is injective.

    Returns the GF(2) sign vector (1 = flip).  Raises ``ValueError`` when the
    signature is not injective.
    """
    n, k = U.shape
    sigma = np.round(np.abs(U), decimals)
    keys = [tuple(sigma[v].tolist()) for v in range(n)]
    if len(set(keys)) != n:
        raise ValueError("fastsign requires an injective abs-value signature")
    order = sorted(range(n), key=lambda v: keys[v], reverse=True)
    U_work = U.copy()
    s = np.zeros(k, dtype=np.uint8)
    for j in range(k):
        v_star = next(v for v in order if abs(U_work[v, j]) > 10 ** (-decimals))
        if U_work[v_star, j] < 0:
            s[j] = 1
            U_work[:, j] = -U_work[:, j]
    return s


# =============================================================================
# Master procedure (Algorithm 1)
# =============================================================================

@dataclass
class PrismResult:
    canonical: np.ndarray
    sign_vector: np.ndarray
    kernel_dim: int
    row_order: np.ndarray
    partition: np.ndarray
    eigenvalues: np.ndarray = field(default=None)


def sign_vector(U: np.ndarray, decimals: int = DECIMALS, fast_path: bool = False) -> np.ndarray:
    """GF(2) sign vector of Partition -> Refine -> Solve (1 = flip the column).

    With ``fast_path=True`` the FastSign shortcut is used when the signature
    is injective.
    """
    U = np.asarray(U, dtype=np.float64)
    if fast_path and is_injective_signature(U, decimals):
        return fastsign(U, decimals)
    p0 = partition_stage(U, decimals)
    classes = refine_stage(U, p0, decimals)
    s, _ = solve_stage(U, classes, decimals)
    return s


def prism_canonicalize(U: np.ndarray, decimals: int = DECIMALS) -> PrismResult:
    """Base PRiSM (Algorithm 1) for a simple-spectrum eigenvector matrix.

    ``U`` is column-wise orthonormal with columns in a fixed (eigenvalue)
    order; column order is not part of the ambiguity group.
    """
    U = np.asarray(U, dtype=np.float64)
    p0 = partition_stage(U, decimals)
    classes = refine_stage(U, p0, decimals)
    s, kernel_basis = solve_stage(U, classes, decimals)
    sign_mult = np.where(s == 1, -1.0, 1.0)
    U_signed = U * sign_mult[None, :]
    canonical, order = match_stage(U_signed, decimals)
    return PrismResult(
        canonical=canonical,
        sign_vector=s,
        kernel_dim=len(kernel_basis),
        row_order=order,
        partition=classes,
    )
