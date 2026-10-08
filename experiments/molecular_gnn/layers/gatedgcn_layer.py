"""GatedGCN layer.

Residual Gated Graph ConvNets, Bresson and Laurent, "An Experimental Study of
Neural Networks for Variable Graphs", https://arxiv.org/abs/1711.07553.

Adapted from benchmarking-gnns (Dwivedi et al., MIT License) via the
canonicalization code of Ma et al. (PKU-ML/canonicalization, MIT License); see
THIRD_PARTY_NOTICES.md.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


class GatedGCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False, graph_norm=True):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.graph_norm = graph_norm
        self.residual = residual
        if input_dim != output_dim:
            self.residual = False
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.C = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)

    def forward(self, g, h, p=None, e=None, snorm_n=None):
        h_in = h  # for residual connection
        e_in = e  # for residual connection
        g.ndata['h'] = h
        g.ndata['Ah'] = self.A(h)
        g.ndata['Bh'] = self.B(h)
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h)
        g.edata['e'] = e
        g.edata['Ce'] = self.C(e)
        g.apply_edges(fn.u_add_v('Dh', 'Eh', 'DEh'))
        g.edata['e'] = g.edata['DEh'] + g.edata['Ce']
        g.edata['sigma'] = torch.sigmoid(g.edata['e'])
        g.update_all(fn.u_mul_e('Bh', 'sigma', 'm'), fn.sum('m', 'sum_sigma_h'))
        g.update_all(fn.copy_e('sigma', 'm'), fn.sum('m', 'sum_sigma'))
        g.ndata['h'] = g.ndata['Ah'] + g.ndata['sum_sigma_h'] / (g.ndata['sum_sigma'] + 1e-6)
        h = g.ndata['h']  # result of graph convolution
        e = g.edata['e']  # result of graph convolution
        if self.graph_norm:
            h = h * snorm_n
        if self.batch_norm:
            h = self.bn_node_h(h)
            e = self.bn_node_e(e)
        h = F.relu(h)
        e = F.relu(e)
        if self.residual:
            h = h_in + h
            e = e_in + e
        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)
        return h, p, e

    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                                            self.in_channels,
                                                            self.out_channels)
