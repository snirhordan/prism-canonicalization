import torch
import torch.nn as nn

from layers.mlp_readout_layer import MLPReadout
from layers.pna_layer import PNALayer
from nets.common import check_pe_init, get_sign_inv_net, readout, uses_sign_inv_net


class PNANet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_atom_type = net_params['num_atom_type']
        num_bond_type = net_params['num_bond_type']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.graph_norm = net_params['graph_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.aggregators = net_params['aggregators']
        self.scalers = net_params['scalers']
        self.avg_d = net_params['avg_d']
        self.towers = net_params['towers']
        self.divide_input_first = net_params['divide_input_first']
        self.divide_input_last = net_params['divide_input_last']
        self.edge_feat = net_params['edge_feat']
        edge_dim = net_params['edge_dim']
        pretrans_layers = net_params['pretrans_layers']
        posttrans_layers = net_params['posttrans_layers']
        self.device = net_params['device']
        self.pe_init = net_params['pe_init']
        self.pos_enc_dim = net_params['pos_enc_dim']
        check_pe_init(self.pe_init)
        self.embedding_p = nn.Linear(self.pos_enc_dim, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.embedding_h = nn.Embedding(num_atom_type, hidden_dim)
        if self.edge_feat:
            self.embedding_e = nn.Embedding(num_bond_type, edge_dim)
        self.layers = nn.ModuleList([PNALayer(in_dim=hidden_dim, out_dim=hidden_dim, dropout=dropout,
                                              graph_norm=self.graph_norm, batch_norm=self.batch_norm,
                                              residual=self.residual, aggregators=self.aggregators,
                                              scalers=self.scalers,
                                              avg_d=self.avg_d, towers=self.towers, edge_features=self.edge_feat,
                                              edge_dim=edge_dim, divide_input=self.divide_input_first,
                                              pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers)
                                     for _ in range(n_layers - 1)])
        self.layers.append(PNALayer(in_dim=hidden_dim, out_dim=out_dim, dropout=dropout,
                                    graph_norm=self.graph_norm, batch_norm=self.batch_norm,
                                    residual=self.residual, aggregators=self.aggregators, scalers=self.scalers,
                                    avg_d=self.avg_d, towers=self.towers, divide_input=self.divide_input_last,
                                    edge_features=self.edge_feat, edge_dim=edge_dim,
                                    pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers))
        self.MLP_layer = MLPReadout(out_dim, 1)  # 1 out dim since regression problem
        if uses_sign_inv_net(self.pe_init):
            self.sign_inv_net = get_sign_inv_net(net_params)
            self.pe_proj = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, g, h, p, e, snorm_n):
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        p = self.embedding_p(p)
        if self.pe_init == 'lap_pe':
            h = h + p
        else:
            h = torch.cat([h, p], dim=1)
            h = self.pe_proj(h)
        p = None
        if self.edge_feat:
            e = self.embedding_e(e)
        for conv in self.layers:
            h, p = conv(g, h, p, e, snorm_n)
        hg = readout(g, h, self.readout)
        return self.MLP_layer(hg)

    def loss(self, scores, targets):
        return nn.L1Loss()(scores, targets)
