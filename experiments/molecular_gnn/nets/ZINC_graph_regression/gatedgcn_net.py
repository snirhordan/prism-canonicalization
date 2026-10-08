import torch
import torch.nn as nn

from layers.gatedgcn_layer import GatedGCNLayer
from layers.mlp_readout_layer import MLPReadout
from nets.common import check_pe_init, get_sign_inv_net, readout, uses_sign_inv_net


class GatedGCNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_atom_type = net_params['num_atom_type']
        num_bond_type = net_params['num_bond_type']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        self.n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        self.pe_init = net_params['pe_init']
        self.pos_enc_dim = net_params['pos_enc_dim']
        check_pe_init(self.pe_init)
        self.embedding_p = nn.Linear(self.pos_enc_dim, hidden_dim)
        self.embedding_h = nn.Embedding(num_atom_type, hidden_dim)
        if self.edge_feat:
            self.embedding_e = nn.Embedding(num_bond_type, hidden_dim)
        else:
            self.embedding_e = nn.Linear(1, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.layers = nn.ModuleList([GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                   self.batch_norm, residual=self.residual, graph_norm=False)
                                     for _ in range(self.n_layers - 1)])
        self.layers.append(
            GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, residual=self.residual, graph_norm=False))
        self.MLP_layer = MLPReadout(out_dim, 1)  # 1 out dim since regression problem
        if uses_sign_inv_net(self.pe_init):
            self.sign_inv_net = get_sign_inv_net(net_params)
            self.pe_proj = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, g, h, p, e, snorm_n):
        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        p = self.embedding_p(p)
        if self.pe_init == 'lap_pe':
            h = h + p
        else:
            h = torch.cat([h, p], dim=1)
            h = self.pe_proj(h)
        p = None
        if not self.edge_feat:  # edge feature set to 1
            e = torch.ones(e.size(0), 1).to(self.device)
        e = self.embedding_e(e)
        # convnets
        for conv in self.layers:
            h, p, e = conv(g, h, p, e, snorm_n)
        hg = readout(g, h, self.readout)
        return self.MLP_layer(hg)

    def loss(self, scores, targets):
        return nn.L1Loss()(scores, targets)
