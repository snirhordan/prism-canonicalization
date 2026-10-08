"""Graph preprocessing of the transformer / BREC harness.

``Transform`` computes the lowest Laplacian eigenpairs of a graph, canonicalizes
them with a canonicalizer from the :mod:`prism` package and attaches the token
structure used by the pair tokenizer.

Parts of this module (eigenpair padding, token index, degree features) are
adapted from "Aligning Transformers with Weisfeiler-Leman" (Mueller and Morris,
ICML 2024), https://github.com/luis-mueller/wl-transformers, MIT License,
Copyright (c) 2024 Luis Mueller.  See ``THIRD_PARTY_NOTICES``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch_geometric.utils import add_self_loops

from prism import PRiSM
from prism.adapters.transformer import canonicalize_eigenpairs


def transform_dataset(dataset, transform):
    """Apply ``transform`` to every graph once and keep the result in memory."""
    data_list = []
    for data in tqdm.tqdm(dataset, miniters=len(dataset) / 50):
        data_list.append(transform(data))
    data_list = list(filter(None, data_list))
    dataset._indices = None
    dataset._data_list = data_list
    dataset._data, dataset.slices = dataset.collate(data_list)
    return dataset


def _sparse_laplacian(edge_index, num_nodes, normalized):
    import scipy.sparse as sp

    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()
    data = np.ones(row.shape[0], dtype=np.float64)
    A = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    deg = np.asarray(A.sum(axis=1)).flatten()
    if normalized:
        d12 = 1.0 / np.sqrt(np.maximum(deg, 1.0))
        D12 = sp.diags(d12)
        L = sp.eye(num_nodes, format="csr") - D12 @ A @ D12
    else:
        L = sp.diags(deg) - A
    return L.astype(np.float64)


def _partial_eigh(edge_index, num_nodes, max_freq, normalized, large_graph):
    """Compute only the eigenpairs that are used, via ``scipy.sparse.linalg.eigsh``.

    * ``large_graph=True``: the lowest ``max_freq // 2`` and the highest
      ``max_freq // 2`` eigenpairs (two ``eigsh`` calls).
    * ``large_graph=False``: the lowest ``max_freq`` eigenpairs (one call).

    Falls back to a dense ``eigh`` when the requested count is within one
    eigenpair of the full spectrum (``eigsh`` requires ``k < n - 1``), which
    only happens for graphs smaller than the requested spectrum.
    """
    import scipy.sparse.linalg as spla

    low_k = (max_freq // 2) if large_graph else max_freq
    high_k = (max_freq // 2) if large_graph else 0
    needed_total = low_k + high_k
    if needed_total >= num_nodes - 1:
        L = _sparse_laplacian(edge_index, num_nodes, normalized).toarray()
        vals, vecs = np.linalg.eigh(L)
        vals = np.clip(vals, 0.0, None)
        if large_graph and vals.shape[0] >= 2 * high_k:
            idx_low = np.arange(low_k)
            idx_high = np.arange(vals.shape[0] - 1, vals.shape[0] - 1 - high_k, -1)
            idx = np.concatenate([idx_low, idx_high])
        else:
            idx = np.arange(min(max_freq, vals.shape[0]))
        return vals[idx], vecs[:, idx]

    L = _sparse_laplacian(edge_index, num_nodes, normalized)
    pieces_vals, pieces_vecs = [], []

    if low_k > 0:
        # Shift-invert slightly below zero avoids the singularity of the null
        # eigenvalue (multiplicity = number of connected components).
        try:
            vals_lo, vecs_lo = spla.eigsh(L, k=low_k, sigma=-1e-10, which="LM")
        except Exception:
            vals_lo, vecs_lo = spla.eigsh(L, k=low_k, which="SA", tol=1e-6)
        order = np.argsort(vals_lo)
        pieces_vals.append(vals_lo[order])
        pieces_vecs.append(vecs_lo[:, order])

    if high_k > 0:
        vals_hi, vecs_hi = spla.eigsh(L, k=high_k, which="LA", tol=1e-6)
        order = np.argsort(vals_hi)[::-1]
        pieces_vals.append(vals_hi[order])
        pieces_vecs.append(vecs_hi[:, order])

    vals = np.concatenate(pieces_vals).astype(np.float64)
    vecs = np.concatenate(pieces_vecs, axis=1).astype(np.float64)
    return np.clip(vals, 0.0, None), vecs


def compute_laplacian_eigen(edge_index, num_nodes, max_freq, normalized=False,
                            normalize=False, large_graph=False):
    """Lowest (and, for ``large_graph``, highest) ``max_freq`` Laplacian eigenpairs.

    Returns float32 ``(eigvals, eigvecs)`` with ``eigvals`` repeated per node,
    shape ``(num_nodes, max_freq)``, and both NaN-padded when the graph has
    fewer than ``max_freq`` nodes.
    """
    vals_np, vecs_np = _partial_eigh(edge_index, num_nodes, max_freq, normalized, large_graph)

    eigvals = torch.from_numpy(vals_np).to(torch.float32)
    eigvecs = torch.from_numpy(vecs_np).to(torch.float32)

    if normalize:
        eignorm = eigvecs.norm(p=2, dim=0, keepdim=True)
        eigvecs = eigvecs / eignorm.clamp_min(1e-12).expand_as(eigvecs)

    if num_nodes < max_freq:
        eigvals = F.pad(eigvals, (0, max_freq - num_nodes), value=float("nan"))
        eigvecs = F.pad(eigvecs, (0, max_freq - num_nodes), value=float("nan"))
    eigvals = eigvals.unsqueeze(0).repeat(num_nodes, 1)
    return eigvals, eigvecs


def _drop_canon_buffer(eigvals, eigvecs, num_nodes, max_freq, large_graph, canon_buffer):
    """Drop the safety buffer after canonicalization.

    The caller computes ``max_freq + (2 if large_graph else 1) * canon_buffer``
    eigenpairs so that canonicalization sees any multiplicity block straddling
    the ``num_vecs`` cutoff.  This strips the buffer elements closest to the
    boundary and leaves exactly ``max_freq`` eigenpairs.

    * Non-large layout: input is ascending; keep the first ``max_freq``.
    * Large-graph layout: input is ``[half+buf low ascending | half+buf high
      descending]``; keep ``[0..half-1]`` of the low side and
      ``[half+buf..max_freq+buf-1]`` of the high side.
    """
    eigvals_1d = eigvals[0] if eigvals.dim() == 2 else eigvals
    n = eigvals_1d.size(0)

    if large_graph:
        half = max_freq // 2
        idx_low = torch.arange(half)
        idx_high = torch.arange(half + canon_buffer, max_freq + canon_buffer)
        idx = torch.cat([idx_low, idx_high])
    else:
        idx = torch.arange(max_freq)

    idx = idx[idx < n]

    eigvals_sel = eigvals_1d[idx]
    eigvecs_sel = eigvecs[:, idx]

    kept = eigvals_sel.size(0)
    if kept < max_freq:
        eigvals_sel = F.pad(eigvals_sel, (0, max_freq - kept), value=float("nan"))
        eigvecs_sel = F.pad(eigvecs_sel, (0, max_freq - kept), value=float("nan"))

    eigvals_sel = eigvals_sel.unsqueeze(0).repeat(num_nodes, 1)
    return eigvals_sel, eigvecs_sel


def _effective_max_freq(max_freq, canon_buffer, large_graph):
    """Number of eigenpairs to compute when canonicalizing with a buffer."""
    return max_freq + (2 * canon_buffer if large_graph else canon_buffer)


def compute_degrees(edge_index, num_nodes):
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.long)
    adj[edge_index[0], edge_index[1]] = 1
    return adj.sum(-1)


def compute_token_index(edge_index, num_nodes, order):
    """Tokens of the pair tokenizer: all edges and self loops (``order == 2``)."""
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    assert edge_index.size(1) > 0
    if order == 2:
        return edge_index
    raise ValueError("Only order 2 (pair tokens) is supported.")


def compute_real_edges(edge_index, num_nodes):
    _, attr = add_self_loops(
        edge_index, torch.ones(edge_index.size(1)), fill_value=0, num_nodes=num_nodes
    )
    return attr.to(bool)


class Transform:
    """Per-graph preprocessing: eigenpairs, canonicalization, token index.

    Parameters
    ----------
    canonicalizer : ``None`` (random-sign LapPE), a :class:`prism.PRiSM`
        instance, or a :class:`prism.baselines.MAP` / ``OAP`` instance.  Use
        :func:`wl_transformers.pe_types.make_canonicalizer` to build it from a
        ``pe_type``.
    canon_buffer : extra eigenpairs canonicalized beyond ``num_vecs`` (and
        dropped afterwards), so a multiplicity block that straddles the cutoff
        is resolved whole.  Only active when a canonicalizer is given.

    For a PRiSM canonicalizer the transform also stores ``canon_rank`` (rank of
    each vertex in the canonical order) and ``canon_order`` on the graph.
    """

    def __init__(self, num_vecs, order, normalized_laplacian, normalize_eigenvecs,
                 large_graph=False, canonicalizer=None, canon_buffer=3):
        self.num_vecs = num_vecs
        self.order = order
        self.normalized = normalized_laplacian
        self.normalize = normalize_eigenvecs
        self.large_graph = large_graph
        self.canonicalizer = canonicalizer
        self.canon_buffer = canon_buffer if canonicalizer is not None else 0

    def __call__(self, data):
        eff_freq = _effective_max_freq(self.num_vecs, self.canon_buffer, self.large_graph)
        data.eigvals, data.eigvecs = compute_laplacian_eigen(
            data.edge_index,
            data.num_nodes,
            eff_freq,
            self.normalized,
            self.normalize,
            self.large_graph,
        )

        if self.canonicalizer is not None:
            self._canonicalize(data)
            # Drop the buffer eigenpairs closest to the spectrum cutoff.
            data.eigvals, data.eigvecs = _drop_canon_buffer(
                data.eigvals, data.eigvecs,
                data.num_nodes, self.num_vecs, self.large_graph, self.canon_buffer,
            )

        data.degrees = compute_degrees(data.edge_index, data.num_nodes)
        data.token_index = compute_token_index(
            data.edge_index, data.num_nodes, self.order
        )
        data.real_edges = compute_real_edges(data.edge_index, data.num_nodes)
        return data

    def _canonicalize(self, data):
        canon = self.canonicalizer
        if isinstance(canon, PRiSM):
            out = canonicalize_eigenpairs(
                data.eigvals[0].numpy(), data.eigvecs.numpy(), canon, self.num_vecs)
            data.eigvecs = torch.from_numpy(out["eigvecs"]).float()
            if "canon_rank" in out:
                data.canon_rank = torch.from_numpy(out["canon_rank"]).long()
                data.canon_order = torch.from_numpy(out["canon_order"]).long()
            return
        # MAP / OAP baselines: canonical eigenvectors only, no vertex order.
        eigvecs_np = data.eigvecs.numpy().astype(np.float64)
        eigvals_np = data.eigvals[0].numpy().astype(np.float64)
        valid = ~np.isnan(eigvals_np)
        if valid.any():
            eigvecs_np[:, valid] = canon.canonicalize(eigvecs_np[:, valid], eigvals_np[valid])
            data.eigvecs = torch.from_numpy(eigvecs_np).float()
