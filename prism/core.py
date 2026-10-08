"""Numpy primitives for eigenvector canonicalization (PRiSM family).

Resolves the sign (m=1), O(2) rotation/reflection (m=2), O(3)
rotation/reflection (m=3), and general O(m) ambiguity (m>=4 via QR)
in eigenvectors to produce a canonical representation.
This is a deterministic, non-differentiable preprocessing step.

Algorithm
---------
1. **Norm-value signature**: For each vertex v compute
       sig(v) = (||pr_1(e_v)||, ..., ||pr_K(e_v)||)
   Norms are invariant under sign flips (m=1), O(2) actions (m=2),
   and O(3) actions (m=3).

2. **Canonical vertex ordering**: Sort vertices lexicographically by sig(v).

3. **Eigenspace canonicalization** (per eigenspace S_i):

   *m_i = 1*  Find first vertex v_j (canonical order) with |psi_i(v_j)| > thr.
              If psi_i(v_j) < 0, flip the sign of the column.

   *m_i = 2*  (a) Rotation -- find first v_j with ||p_{v_j}|| > thr, rotate all
              rows of the 2-column block by -theta so p_{v_j} lands on the
              positive x-axis.
              (b) Reflection -- find first v_k (k > j in canonical order) with
              |p_{v_k}[1]| > thr (after rotation).  If p_{v_k}[1] < 0, negate
              the second column.

   *m_i = 3*  (a) Householder -- find first v_j with ||p_{v_j}|| > thr, apply
              Householder reflection to map p_{v_j} to the +x axis.
              (b) In-plane rotation -- find next v_k with sqrt(y^2+z^2) > thr,
              rotate about x-axis by -atan2(z,y) to place it in the xy
              half-plane (y > 0).
              (c) z-reflection -- find next v_l with |z| > thr.  If z < 0,
              negate the z-column.

   *m_i >= 4* Gram-Schmidt pivot selection, LQ factorization, Stewart sign
              convention (see :func:`canonicalize_eigenspace_qr`).

4. Return canonicalized eigenvectors.

Two numerics profiles are supported through :class:`Tolerances`.  ``adaptive``
snaps tie-detection quantities onto a grid relative to their magnitude and
adds an eps-scale floor to the eigenvalue grouping tolerance.  ``legacy_round``
rounds to a fixed number of decimals and uses no floor.  Both profiles are
used by the experiments of the paper (see :mod:`prism.presets`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


_F64_EPS = float(np.finfo(np.float64).eps)


# ---------------------------------------------------------------------------
# Numerical-tolerance profile
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Tolerances:
    """Tolerances shared by all canonicalization paths.

    Attributes
    ----------
    mult_rtol : relative tolerance for grouping eigenvalues into eigenspaces.
    mult_atol : absolute floor of the grouping tolerance (added via ``max``).
    order_decimals : fixed decimals used to snap norm signatures before the
        lexicographic sort; ``None`` selects adaptive relative snapping.
    refine_decimals : same, for the sequential ordering refinement.
    unique_decimals : same, for the unique-signature mask of the QR path.
    tie_rtol : relative grid used by adaptive snapping.
    """

    mult_rtol: float = 1e-10
    mult_atol: float = 100.0 * _F64_EPS
    order_decimals: Optional[int] = None
    refine_decimals: Optional[int] = None
    unique_decimals: Optional[int] = None
    tie_rtol: float = 1e-9


def make_tolerances(profile: str, mult_mode: str = "hybrid") -> Tolerances:
    """Return the :class:`Tolerances` of a named profile.

    ``adaptive``      relative snapping + eps floor (canonicalization-fresh).
    ``legacy_round``  fixed-decimal rounding, no floor (transformer / BREC
                      harness): 10 decimals for the base ordering, 10 (hybrid)
                      or 8 (QR) for refinement, 8 for the uniqueness mask.
    """
    if profile == "adaptive":
        return Tolerances()
    if profile == "legacy_round":
        return Tolerances(
            mult_atol=0.0,
            order_decimals=10,
            refine_decimals=10 if mult_mode == "hybrid" else 8,
            unique_decimals=8,
        )
    raise ValueError(f"unknown tol_profile {profile!r}; use 'adaptive' or 'legacy_round'")


def _adaptive_round(x: np.ndarray, rtol: float) -> np.ndarray:
    """Snap values onto a grid whose spacing scales with the data magnitude.

    The grid spacing for column ``c`` is ``rtol * max(|x[:, c]|, 1.0)``.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.copy()
    if x.ndim == 1:
        scale = max(float(np.max(np.abs(x))), 1.0)
        step = rtol * scale
        return x.copy() if step <= 0 else np.round(x / step) * step
    scale = np.maximum(np.max(np.abs(x), axis=0, keepdims=True), 1.0)
    step = rtol * scale
    step = np.where(step > 0, step, 1.0)
    return np.round(x / step) * step


def _snap(x: np.ndarray, decimals: Optional[int], rtol: float) -> np.ndarray:
    if decimals is not None:
        return np.round(x, decimals)
    return _adaptive_round(x, rtol)


# ---------------------------------------------------------------------------
# Multiplicities, signature, ordering
# ---------------------------------------------------------------------------

def detect_multiplicities(eigenvalues, rtol: float = 1e-10,
                          atol: Optional[float] = None) -> list:
    """Group *sorted* eigenvalues by approximate equality; return multiplicities.

    Two neighbours are merged when ``|a-b| <= max(rtol * max(|a|,|b|,1), atol)``.
    ``atol=None`` selects the eps-scale floor ``100 * eps``.
    """
    eigs = np.asarray(eigenvalues, dtype=np.float64)
    if len(eigs) == 0:
        return []
    if atol is None:
        atol = 100.0 * _F64_EPS

    multiplicities: list = []
    i = 0
    n = len(eigs)
    while i < n:
        j = i + 1
        while j < n:
            ref = max(abs(eigs[i]), abs(eigs[j]), 1.0)
            if abs(eigs[j] - eigs[i]) <= max(rtol * ref, atol):
                j += 1
            else:
                break
        multiplicities.append(j - i)
        i = j
    return multiplicities


def compute_norm_signature(eigenvectors: np.ndarray, multiplicities: list) -> np.ndarray:
    """Norm-value signature: ``sig[v, i] = ||pr_i(e_v)||``, shape (n, K)."""
    n = eigenvectors.shape[0]
    K = len(multiplicities)
    sig = np.zeros((n, K))

    col = 0
    for idx, m_i in enumerate(multiplicities):
        block = eigenvectors[:, col: col + m_i]
        sig[:, idx] = np.linalg.norm(block, axis=1)
        col += m_i
    return sig


def canonical_ordering(signatures: np.ndarray, tol: Tolerances) -> np.ndarray:
    """Lexicographic argsort of signature rows (stable), after tie snapping."""
    n = signatures.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    sigs = _snap(signatures, tol.order_decimals, tol.tie_rtol)

    K = sigs.shape[1]
    order = np.arange(n, dtype=np.int64)
    for col in range(K - 1, -1, -1):
        perm = np.argsort(sigs[order, col], kind="stable")
        order = order[perm]
    return order


def signature_uniqueness(signatures: np.ndarray, decimals: Optional[int],
                         rtol: float) -> np.ndarray:
    """Boolean mask: True for vertices whose (snapped) signature row is unique.

    Unique-signature vertices are safe QR pivots because their position in the
    canonical ordering is fixed by their signature, not by array index.
    """
    n = signatures.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)

    sigs = _snap(signatures, decimals, rtol)

    order = np.lexsort(sigs[:, ::-1].T)
    sorted_sigs = sigs[order]

    if sorted_sigs.shape[1] == 0:
        adj_equal = np.ones(n - 1, dtype=bool)
    else:
        adj_equal = np.all(sorted_sigs[1:] == sorted_sigs[:-1], axis=1)
    is_dup_sorted = np.zeros(n, dtype=bool)
    is_dup_sorted[1:] |= adj_equal
    is_dup_sorted[:-1] |= adj_equal

    is_unique = np.ones(n, dtype=bool)
    is_unique[order] = ~is_dup_sorted
    return is_unique


def refine_ordering(V: np.ndarray, canon_cols: list, base_order: np.ndarray,
                    decimals: Optional[int], rtol: float) -> np.ndarray:
    """Recompute the vertex ordering from already canonical columns.

    Lexicographic sort on ``V[:, canon_cols]``; ties are broken by
    ``base_order`` (the norm-signature ordering, which is permutation
    equivariant).
    """
    if not canon_cols:
        return base_order.copy()
    cols = sorted(canon_cols)
    sigs = _snap(V[:, cols], decimals, rtol)
    K = sigs.shape[1]
    order = base_order.copy()
    for c in range(K - 1, -1, -1):
        perm = np.argsort(sigs[order, c], kind="stable")
        order = order[perm]
    return order


# ---------------------------------------------------------------------------
# m=3 helpers: Householder reflection and rotation about x-axis
# ---------------------------------------------------------------------------

def rotation_to_ex(u):
    """Householder reflection R (3x3) with ``R @ u/||u|| = e_x``."""
    ex = np.array([1.0, 0.0, 0.0])
    u_hat = u / np.linalg.norm(u)
    if np.abs(np.dot(u_hat, ex) - 1.0) < 1e-12:
        return np.eye(3)
    if np.abs(np.dot(u_hat, ex) + 1.0) < 1e-12:
        return np.diag([-1.0, 1.0, 1.0])
    w = u_hat - ex
    v = w / np.linalg.norm(w)
    R = np.eye(3) - 2.0 * np.outer(v, v)
    return R


def rotation_about_x(theta):
    """3x3 rotation matrix about the x-axis by angle ``theta``."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0,   c,  -s],
                     [0.0,   s,   c]])


# ---------------------------------------------------------------------------
# Per-eigenspace resolution
# ---------------------------------------------------------------------------

def canonicalize_eigenspace_qr(V, c_start, c_end, order, n, threshold,
                               is_unique=None):
    """Canonicalize one eigenspace block ``V[:, c_start:c_end]`` via QR in place.

    1. Select m linearly independent pivot rows via Gram-Schmidt rejection,
       iterating through vertices in canonical order.  When *is_unique* is
       given, vertices with unique signatures are tried first and tied
       vertices are used only as fallback.
    2. Form the m x m pivot matrix M and factorize ``M^T = Q R``.
    3. Stewart sign convention: flip Q columns so that ``diag(R) > 0``.
    4. Apply Q to all rows: ``V[:, block] = V[:, block] @ Q``.

    If fewer than m independent pivots exist (numerically degenerate), the
    basis is extended with standard basis vectors via Gram-Schmidt.
    """
    m = c_end - c_start
    if m <= 0:
        return

    basis = np.zeros((m, m), dtype=np.float64)
    pivot_rows = np.zeros((m, m), dtype=np.float64)
    r = 0

    def _try_pivot(v_idx):
        nonlocal r
        v = order[v_idx]
        row = V[v, c_start:c_end].copy()
        residual = row.copy()
        for j in range(r):
            residual -= np.dot(residual, basis[j]) * basis[j]
        norm_res = np.linalg.norm(residual)
        if norm_res > threshold:
            basis[r] = residual / norm_res
            pivot_rows[r] = row
            r += 1
            return True
        return False

    if is_unique is not None:
        for idx in range(n):
            if r == m:
                break
            if is_unique[order[idx]]:
                _try_pivot(idx)
        for idx in range(n):
            if r == m:
                break
            if not is_unique[order[idx]]:
                _try_pivot(idx)
    else:
        for idx in range(n):
            if r == m:
                break
            _try_pivot(idx)

    if r == 0:
        return

    if r < m:
        for i in range(m):
            if r == m:
                break
            e_i = np.zeros(m, dtype=np.float64)
            e_i[i] = 1.0
            residual = e_i.copy()
            for j in range(r):
                residual -= np.dot(residual, basis[j]) * basis[j]
            norm_res = np.linalg.norm(residual)
            if norm_res > threshold:
                basis[r] = residual / norm_res
                pivot_rows[r] = e_i
                r += 1

    M = pivot_rows[:m, :]
    Q_t, R_t = np.linalg.qr(M.T)

    signs = np.sign(np.diag(R_t))
    signs[signs == 0] = 1.0
    Q_t = Q_t * signs[np.newaxis, :]

    V[:, c_start:c_end] = V[:, c_start:c_end] @ Q_t


def resolve_m1(V, c, order, n, threshold):
    """m=1: flip the column so its first significant entry (in order) is positive."""
    for idx in range(n):
        v = order[idx]
        val = V[v, c]
        if abs(val) > threshold:
            if val < 0:
                V[:, c] = -V[:, c]
            break


def resolve_m2(V, c_start, c_end, order, n, threshold):
    """m=2: rotate the first pivot to +x, then fix the reflection from the next pivot."""
    pivot_j = -1
    for idx in range(n):
        v = order[idx]
        r = np.linalg.norm(V[v, c_start:c_end])
        if r > threshold:
            pivot_j = idx
            x_val = V[v, c_start]
            y_val = V[v, c_start + 1]
            theta = np.arctan2(y_val, x_val)
            cos_t = np.cos(-theta)
            sin_t = np.sin(-theta)
            R = np.array([[cos_t, -sin_t],
                          [sin_t,  cos_t]])
            block = V[:, c_start:c_end].copy()
            V[:, c_start:c_end] = block @ R.T
            break

    if pivot_j >= 0:
        for idx in range(pivot_j + 1, n):
            v = order[idx]
            y_after = V[v, c_start + 1]
            if abs(y_after) > threshold:
                if y_after < 0:
                    V[:, c_start + 1] = -V[:, c_start + 1]
                break


def resolve_m3(V, c_start, c_end, order, n, threshold):
    """m=3: Householder to +x, rotation about x into the y>0 half-plane, z-flip."""
    pivot_j = -1
    for idx in range(n):
        v = order[idx]
        p = V[v, c_start:c_end]
        r = np.linalg.norm(p)
        if r > threshold:
            pivot_j = idx
            R1 = rotation_to_ex(p)
            V[:, c_start:c_end] = V[:, c_start:c_end] @ R1.T
            break

    if pivot_j < 0:
        return  # degenerate: all vectors too small

    pivot_k = -1
    for idx in range(pivot_j + 1, n):
        v = order[idx]
        y_val = V[v, c_start + 1]
        z_val = V[v, c_start + 2]
        rho = np.sqrt(y_val ** 2 + z_val ** 2)
        if rho > threshold:
            pivot_k = idx
            theta = np.arctan2(z_val, y_val)
            R2 = rotation_about_x(-theta)
            V[:, c_start:c_end] = V[:, c_start:c_end] @ R2.T
            break

    if pivot_k < 0:
        return  # degenerate: all vertices collinear with the x-axis

    for idx in range(pivot_k + 1, n):
        v = order[idx]
        z_val = V[v, c_start + 2]
        if abs(z_val) > threshold:
            if z_val < 0:
                V[:, c_start + 2] = -V[:, c_start + 2]
            break


def _resolve_hybrid(V, m_i, c_start, c_end, order, n, threshold):
    """Dispatch the hybrid per-eigenspace rule (Householder/Givens, QR for m>=4)."""
    if m_i == 1:
        resolve_m1(V, c_start, order, n, threshold)
    elif m_i == 2:
        resolve_m2(V, c_start, c_end, order, n, threshold)
    elif m_i == 3:
        resolve_m3(V, c_start, c_end, order, n, threshold)
    else:
        canonicalize_eigenspace_qr(V, c_start, c_end, order, n, threshold)


# ---------------------------------------------------------------------------
# Whole-spectrum drivers
# ---------------------------------------------------------------------------

def _eigenspace_info(multiplicities: list) -> list:
    """(m_i, c_start, c_end, index) sorted by multiplicity (stable)."""
    info = []
    col = 0
    for idx, m_i in enumerate(multiplicities):
        info.append((m_i, col, col + m_i, idx))
        col += m_i
    info.sort(key=lambda x: x[0])
    return info


def canonicalize_hybrid(eigenvectors, multiplicities, threshold, tol: Tolerances):
    """Hybrid canonicalization, one fixed norm-signature ordering.

    Eigenspaces are processed in eigenvalue order.  m=1,2,3 use the
    specialized paths, m>=4 uses QR.  Returns ``(V, order)``.
    """
    V = eigenvectors.copy()
    n, k = V.shape
    if n == 0 or k == 0:
        return V, np.arange(n, dtype=np.int64)

    sig = compute_norm_signature(V, multiplicities)
    order = canonical_ordering(sig, tol)

    col = 0
    for m_i in multiplicities:
        _resolve_hybrid(V, m_i, col, col + m_i, order, n, threshold)
        col += m_i
    return V, order


def canonicalize_hybrid_refined(eigenvectors, multiplicities, threshold,
                                tol: Tolerances, gf2_solver=None):
    """Hybrid canonicalization with sequential ordering refinement.

    Eigenspaces are processed in order of multiplicity (m=1 first).  After
    each eigenspace the vertex ordering is recomputed from the canonical
    values of all processed columns, with ties broken by the norm-signature
    order.  Returns ``(V, final_order)``.

    If *gf2_solver* is given (``sign_rule="gf2"``), all m=1 eigenspaces are
    treated as one block whose signs are produced by that callable
    ``solver(U1) -> bool mask of columns to flip`` (Refine + Solve of
    Algorithm 1) instead of the per-column first-pivot rule.
    """
    V = eigenvectors.copy()
    n, k = V.shape
    if n == 0 or k == 0:
        return V, np.arange(n, dtype=np.int64)

    eigenspace_info = _eigenspace_info(multiplicities)

    sig = compute_norm_signature(V, multiplicities)
    base_order = canonical_ordering(sig, tol)
    order = base_order.copy()

    canonicalized_cols: list = []

    def _refined():
        return refine_ordering(V, canonicalized_cols, base_order,
                               tol.refine_decimals, tol.tie_rtol)

    if gf2_solver is not None:
        cols1 = [c0 for (m_i, c0, _c1, _i) in eigenspace_info if m_i == 1]
        if cols1:
            flip = np.asarray(gf2_solver(V[:, cols1]), dtype=bool)
            for c, f in zip(cols1, flip):
                if f:
                    V[:, c] = -V[:, c]
            canonicalized_cols.extend(cols1)
        eigenspace_info = [e for e in eigenspace_info if e[0] != 1]

    for m_i, c_start, c_end, _orig in eigenspace_info:
        if canonicalized_cols:
            order = _refined()
        _resolve_hybrid(V, m_i, c_start, c_end, order, n, threshold)
        canonicalized_cols.extend(range(c_start, c_end))

    final_order = _refined() if canonicalized_cols else base_order
    return V, final_order


def canonicalize_qr_refined(eigenvectors, multiplicities, threshold, tol: Tolerances):
    """QR canonicalization for every multiplicity with ordering refinement.

    Eigenspaces are processed in order of multiplicity (m=1 first).  Pivots are
    chosen among vertices with a unique signature when ties exist, and the
    ordering is refined only while ties remain.  Returns ``(V, final_order)``;
    the final order is not used by any harness.
    """
    V = eigenvectors.copy()
    n, k = V.shape
    if n == 0 or k == 0:
        return V, np.arange(n, dtype=np.int64)

    eigenspace_info = _eigenspace_info(multiplicities)

    sig = compute_norm_signature(V, multiplicities)
    base_order = canonical_ordering(sig, tol)
    order = base_order.copy()
    is_unique = signature_uniqueness(sig, tol.unique_decimals, tol.tie_rtol)
    has_ties = not is_unique.all()

    canonicalized_cols: list = []

    for m_i, c_start, c_end, _orig in eigenspace_info:
        if has_ties and canonicalized_cols:
            order = refine_ordering(V, canonicalized_cols, base_order,
                                    tol.refine_decimals, tol.tie_rtol)
            cols = sorted(canonicalized_cols)
            if tol.unique_decimals is not None:
                refined_sig = np.round(V[:, cols], tol.unique_decimals)
                is_unique = signature_uniqueness(refined_sig, tol.unique_decimals,
                                                 tol.tie_rtol)
            else:
                is_unique = signature_uniqueness(V[:, cols], None, tol.tie_rtol)
            has_ties = not is_unique.all()

        canonicalize_eigenspace_qr(
            V, c_start, c_end, order, n, threshold,
            is_unique=is_unique if has_ties else None,
        )
        canonicalized_cols.extend(range(c_start, c_end))

    final_order = (refine_ordering(V, canonicalized_cols, base_order,
                                   tol.refine_decimals, tol.tie_rtol)
                   if canonicalized_cols else base_order)
    return V, final_order
