"""DGL adapter for the molecular GNN harness."""

from __future__ import annotations

from typing import Optional


def add_pos_enc(g, canon, k: Optional[int] = None, key: str = "pos_enc"):
    """Store the PRiSM positional encoding of a DGL graph in ``g.ndata[key]``.

    ``canon`` is a :class:`prism.PRiSM` (or a MAP/OAP baseline); ``k`` is the
    number of columns.  The encoding is a float32 tensor of shape (n, k).
    """
    import torch

    A = g.adj_external(scipy_fmt="csr")
    g.ndata[key] = torch.from_numpy(canon.pos_enc(A, k))
    return g
