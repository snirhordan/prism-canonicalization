import torch
import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from layers.gatedgcn_layer import GatedGCNLayer
from layers.mlp_readout_layer import MLPReadout
from nets.common import check_pe_init, get_sign_inv_net, readout, uses_sign_inv_net


class GatedGCNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.device = net_params['device']
        self.pe_init = net_params['pe_init']
        self.n_classes = n_classes
        self.pos_enc_dim = net_params['pos_enc_dim']
        check_pe_init(self.pe_init)
        self.embedding_p = nn.Linear(self.pos_enc_dim, hidden_dim)
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bond_encoder = BondEncoder(hidden_dim)
        self.layers = nn.ModuleList([GatedGCNLayer(hidden_dim, hidden_dim, dropout, self.batch_norm, self.residual)
                                     for _ in range(n_layers - 1)])
        self.layers.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes)
        if uses_sign_inv_net(self.pe_init):
            self.sign_inv_net = get_sign_inv_net(net_params)
            self.pe_proj = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, g, h, p, e, snorm_n):
        h = self.atom_encoder(h)
        e = self.bond_encoder(e)
        p = self.embedding_p(p)
        if self.pe_init == 'lap_pe':
            h = h + p
        else:
            h = torch.cat([h, p], dim=1)
            h = self.pe_proj(h)
        p = None
        for conv in self.layers:
            h, p, e = conv(g, h, p, e, snorm_n)
        hg = readout(g, h, self.readout)
        return self.MLP_layer(hg)

    def loss(self, pred, labels):
        return nn.BCEWithLogitsLoss()(pred, labels)
