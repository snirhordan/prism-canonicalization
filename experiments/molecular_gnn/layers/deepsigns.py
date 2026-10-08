"""Network that processes a positional encoding of shape (nodes, k, 1) before it is fused
with the node features.

``use_signnet=True`` is the sign-invariant SignNet map ``enc(x) + enc(-x)``; for
canonicalized encodings (PRiSM, MAP, OAP) the same network is applied once, without
the symmetrization.  Column indices at or beyond the number of nodes of a graph are
padding and are masked out before the sum over the k axis.

Adapted from SignNet-BasisNet (Lim et al., MIT License); see THIRD_PARTY_NOTICES.md.
"""

import torch
import torch.nn as nn

from layers.mlp import MLP
from layers.gnns import GIN


class MaskedGINDeepSigns(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, k, use_bn=False, use_ln=False,
                 dropout=0.5, activation='relu', use_signnet=False):
        super(MaskedGINDeepSigns, self).__init__()
        self.enc = GIN(in_channels, hidden_channels, out_channels, num_layers, use_bn=use_bn, dropout=dropout,
                       activation=activation)
        self.rho = MLP(out_channels, hidden_channels, k, num_layers, use_bn=use_bn, dropout=dropout,
                       activation=activation)
        self.k = k
        self.use_signnet = use_signnet

    def forward(self, g, x):
        if self.use_signnet:
            x = self.enc(g, x) + self.enc(g, -x)
        else:
            x = self.enc(g, x)
        orig_shape = x.shape  # (total nodes, k, out_channels)
        batch_num_nodes = g.batch_num_nodes()
        # number of nodes of the graph that each node belongs to
        n_nodes = torch.repeat_interleave(batch_num_nodes, batch_num_nodes)
        mask = torch.arange(orig_shape[1], device=x.device).unsqueeze(0) < n_nodes.unsqueeze(1)
        x[~mask] = 0
        x = x.sum(dim=1)
        x = self.rho(x)
        x = x.reshape(orig_shape[0], self.k, 1)
        return x
