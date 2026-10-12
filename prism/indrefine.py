"""Individualization-refinement ordering with nauty (``ordering="indrefine_nauty"``).

This is the vertex-ordering layer of :class:`prism.PRiSM` that produced the
PRiSM row of the BREC table.  It replaces the norm-signature ordering of
:mod:`prism.core` by an ordering built from the basis-invariant eigenspace
projection matrices ``P_i = U_i U_i^T`` and a canonical labelling of the graph
computed by nauty (through ``pynauty``), then resolves every eigenspace by
pivoted QR.  It is not a separate canonicalizer: the :class:`~prism.PRiSM`
constructor selects it with ``ordering="indrefine_nauty"`` and calls
:func:`canonicalize` below.

Procedure for eigenvectors ``V`` (n, c) with eigenvalues ``lam`` of a graph
with adjacency ``A``:

1. Group the eigenvalues into eigenspaces (relative tolerance 1e-10) and form
   the projection matrices ``P_i`` of the supplied columns, rounded to
   ``ir_proj_decimals``.
2. Colour every vertex by the tuple ``(P_1[v, v], ..., P_K[v, v])`` rounded to
   ``ir_order_decimals`` (colour classes sorted by that tuple) and compute the
   canonical labelling of the coloured graph with nauty.  ``nauty_rank[v]`` is
   the position of ``v`` in it.
3. For every eigenspace order the vertices by the key
   ``(-P_i[v, v], sorted |P_i[v, :]| descending, nauty_rank[v])``.
4. Eigenspace of multiplicity one: flip the column so that its first entry of
   magnitude above the pivot threshold, in that order, is positive.
   Multiplicity m > 1: take the first m vertices of that order whose rows are
   linearly independent as pivots, QR-factorize the transposed (m, m) pivot
   block with a positive diagonal of R (Stewart convention) and rotate all
   rows by Q.  An eigenspace with fewer than m independent pivot rows is left
   unchanged.

None of the three keys of step 3 depends on the basis chosen inside an eigenspace.
The nauty rank is the only part that depends on how the graph is labelled, and
nauty's canonical form is unique only up to automorphisms of the coloured graph.
For a graph with a trivial automorphism group the output is therefore equivariant
under vertex permutations (up to the rounding of the keys); for a graph with
non-trivial automorphisms, different labellings can lead to different
representatives, which may flip the signs of antisymmetric eigenvectors.

Eigenvectors are returned in the original vertex order.  The second return
value is the nauty canonical order (``order[i]`` = vertex at position i).

The code is the ``indrefine_nauty_canonicalize`` path of the research code
that produced the BREC row, with the pynauty import made lazy and with the
eigenvector-based reconstruction of the adjacency (a fallback of that code
when no edge list is given) replaced by a required ``adjacency`` argument.
``pynauty`` (GPL-3.0-or-later, installed with ``pip install
'prism-canonicalization[nauty]'``) is imported only when this ordering is used.
"""

from __future__ import annotations

import numpy as np

PYNAUTY_HINT = ("ordering='indrefine_nauty' needs the optional dependency pynauty; "
                "install it with: pip install 'prism-canonicalization[nauty]' "
                "(or: pip install pynauty)")


def require_pynauty():
    """Import and return ``pynauty``; raise a descriptive ImportError if it is missing."""
    try:
        import pynauty
    except ImportError as exc:
        raise ImportError(PYNAUTY_HINT) from exc
    return pynauty


def detect_multiplicities(evals, rtol: float = 1e-10) -> list:
    """Group sorted eigenvalues; two are merged when ``|a - b| / max(|first of group|, 1e-14) < rtol``.

    This is the grouping of the research code of the BREC row (relative to the first
    eigenvalue of the group, floor 1e-14), which differs from
    :func:`prism.core.detect_multiplicities`: there is no absolute tolerance, so the
    numerically zero eigenvalues (about 1e-16) of a disconnected graph are not merged
    into one eigenspace in float64.
    """
    mults = []
    if len(evals) == 0:
        return mults
    curr = 1
    for i in range(1, len(evals)):
        ref = max(abs(evals[i - curr]), 1e-14)
        if abs(evals[i] - evals[i - curr]) / ref < rtol:
            curr += 1
        else:
            mults.append(curr)
            curr = 1
    mults.append(curr)
    return mults


def nauty_canonical_order(adj, vertex_coloring=None) -> np.ndarray:
    """Canonical labelling of a (coloured) graph by nauty: ``order[i]`` = vertex at position i."""
    pynauty = require_pynauty()
    n = adj.shape[0]
    kwargs = {}
    if vertex_coloring:
        kwargs["vertex_coloring"] = vertex_coloring
    g = pynauty.Graph(n, **kwargs)
    for v in range(n):
        nbrs = [int(u) for u in np.where(adj[v] > 0)[0]]
        if nbrs:
            g.connect_vertex(v, nbrs)
    return np.array(pynauty.canon_label(g), dtype=np.int64)


def spectral_node_ordering(P, nauty_rank, n, decimals: int = 8) -> np.ndarray:
    """Vertex order of one eigenspace from its projection matrix ``P`` and the nauty rank."""
    keys = []
    for v in range(n):
        diag_val = -round(float(P[v, v]), decimals)
        row = np.abs(P[v, :])
        sorted_row = tuple(-round(float(x), decimals) for x in sorted(row, reverse=True))
        keys.append((diag_val, sorted_row, int(nauty_rank[v])))
    return np.array(sorted(range(n), key=lambda v: keys[v]), dtype=np.int64)


def _canonicalize_eigenspace(V, c_start, c_end, P, nauty_rank, n, threshold, decimals):
    """Resolve one eigenspace of ``V`` in place (sign for m = 1, pivoted QR for m > 1)."""
    m = c_end - c_start
    if m == 0:
        return
    order = spectral_node_ordering(P, nauty_rank, n, decimals)

    if m == 1:
        for idx in range(n):
            v = order[idx]
            val = V[v, c_start]
            if abs(val) > threshold:
                if val < 0:
                    V[:, c_start] = -V[:, c_start]
                break
        return

    pivot_rows = []
    for idx in range(n):
        v = order[idx]
        row = V[v, c_start:c_end]
        if np.linalg.norm(row) < threshold:
            continue
        test = list(pivot_rows) + [v]
        M = V[test, c_start:c_end]
        if np.linalg.matrix_rank(M, tol=threshold) == len(test):
            pivot_rows.append(v)
            if len(pivot_rows) == m:
                break

    if len(pivot_rows) < m:
        return  # degenerate: not enough independent pivot rows

    M = V[pivot_rows, c_start:c_end]
    Q, R = np.linalg.qr(M.T)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1
    Q = Q * signs
    V[:, c_start:c_end] = V[:, c_start:c_end] @ Q


def canonicalize(V, lam, adjacency, threshold: float = 1e-8, order_decimals: int = 8,
                 proj_decimals: int = 10):
    """Canonicalize eigenvectors by the nauty ordering and pivoted QR.

    Parameters
    ----------
    V : (n, c) float array, columns are eigenvectors in eigenvalue order.
    lam : (c,) float array, eigenvalues in non-decreasing order.
    adjacency : (n, n) array; its support (entries > 0) is the edge set given to nauty.
    threshold : minimum magnitude of a pivot entry / residual (``pivot_threshold``).
    order_decimals, proj_decimals : rounding of the vertex colours and ordering keys
        (``ir_order_decimals``) and of the projection matrices (``ir_proj_decimals``).

    Returns
    -------
    ``(Vc, order, mults)``: canonical eigenvectors in the original vertex order
    (float64, (n, c)), the nauty canonical vertex order (``order[i]`` = vertex at
    position i) and the eigenspace multiplicities used.
    """
    V = np.array(V, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    adjacency = np.asarray(adjacency, dtype=np.float64)
    n = V.shape[0]
    if adjacency.shape != (n, n):
        raise ValueError(f"adjacency must have shape ({n}, {n}), got {adjacency.shape}")
    if V.size == 0:
        return V, np.arange(n, dtype=np.int64), []

    mults = detect_multiplicities(lam)

    P_list = []
    col = 0
    for m in mults:
        U = V[:, col:col + m]
        P_list.append(np.round(U @ U.T, proj_decimals))
        col += m

    # Vertex colouring of nauty: the diagonals of the projection matrices.  Colour
    # classes are listed by increasing signature so the list is permutation-equivariant.
    K = len(P_list)
    sig_groups = {}
    for v in range(n):
        sig = tuple(round(float(P_list[i][v, v]), order_decimals) for i in range(K))
        sig_groups.setdefault(sig, set()).add(v)
    coloring = [grp for _, grp in sorted(sig_groups.items())]
    if len(coloring) <= 1:
        coloring = []

    nauty_order = nauty_canonical_order(adjacency, vertex_coloring=coloring)
    nauty_rank = np.zeros(n, dtype=np.int64)
    nauty_rank[nauty_order] = np.arange(n)

    col = 0
    for i, m in enumerate(mults):
        _canonicalize_eigenspace(V, col, col + m, P_list[i], nauty_rank, n, threshold,
                                 order_decimals)
        col += m
    return V, nauty_order, mults
