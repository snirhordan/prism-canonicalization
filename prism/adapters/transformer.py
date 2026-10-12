"""Adapter for the transformer / BREC harness.

The harness stores, per graph, the lowest ``num_vecs + buffer`` Laplacian
eigenpairs in float32 (NaN padded when the graph is smaller).  PRiSM
canonicalizes the valid ones and the harness additionally needs the canonical
vertex order, its rank (for rotary or sinusoidal rank embeddings) and the
norm-value signature.
"""

from __future__ import annotations

import numpy as np


def canonicalize_eigenpairs(eigvals, eigvecs, canon, num_vecs: int, adjacency=None) -> dict:
    """Canonicalize NaN-padded eigenpairs and return the harness features.

    Parameters
    ----------
    eigvals : (c,) array, ascending, NaN for missing eigenpairs.
    eigvecs : (n, c) array, NaN columns for missing eigenpairs.
    canon : :class:`prism.PRiSM`.
    num_vecs : width of the padded signature.
    adjacency : (n, n) array, only needed when ``canon.needs_adjacency`` (the
        ``indrefine_nauty`` ordering).

    Returns a dict with ``eigvecs`` (float32, input shape), ``norm_sig``
    (float32, (n, num_vecs)), ``canon_rank`` and ``canon_order`` (int64, (n,)).
    """
    eigvals = np.asarray(eigvals, dtype=np.float64)
    eigvecs = np.asarray(eigvecs, dtype=np.float64).copy()
    n = eigvecs.shape[0]
    valid = ~np.isnan(eigvals)
    out = {"eigvecs": eigvecs.astype(np.float32)}
    if not valid.any():
        return out
    info = canon.canonicalize(eigvecs[:, valid], eigvals[valid], return_info=True,
                              adjacency=adjacency)
    eigvecs[:, valid] = info.V
    out["eigvecs"] = eigvecs.astype(np.float32)
    sig = np.zeros((n, num_vecs), dtype=np.float64)
    cols = min(info.norm_sig.shape[1], num_vecs)
    sig[:, :cols] = info.norm_sig[:, :cols]
    out["norm_sig"] = sig.astype(np.float32)
    out["canon_rank"] = info.rank
    out["canon_order"] = info.order
    return out
