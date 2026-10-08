import torch
import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from layers.mlp_readout_layer import MLPReadout2 as MLPReadout
from layers.pna_layer import PNANoTowersLayer as PNALayer
from nets.common import check_pe_init, get_sign_inv_net, readout, uses_sign_inv_net


class PNANet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        self.dropout_2 = net_params['dropout_2']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.graph_norm = net_params['graph_norm']
        self.batch_norm = net_params['batch_norm']
        self.aggregators = net_params['aggregators']
        self.scalers = net_params['scalers']
        self.avg_d = net_params['avg_d']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        edge_dim = net_params['edge_dim']
        pretrans_layers = net_params['pretrans_layers']
        posttrans_layers = net_params['posttrans_layers']
        self.device = net_params['device']
        self.pe_init = net_params['pe_init']
        self.n_classes = n_classes
        self.pos_enc_dim = net_params['pos_enc_dim']
        check_pe_init(self.pe_init)
        self.embedding_p = nn.Linear(self.pos_enc_dim, hidden_dim)
        self.embedding_h = AtomEncoder(emb_dim=hidden_dim)
        if self.edge_feat:
            self.embedding_e = BondEncoder(emb_dim=edge_dim)
        self.layers = nn.ModuleList(
            [PNALayer(in_dim=hidden_dim, out_dim=hidden_dim, dropout=dropout, graph_norm=self.graph_norm,
                      batch_norm=self.batch_norm, aggregators=self.aggregators, scalers=self.scalers,
                      avg_d=self.avg_d,
                      pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers, residual=self.residual,
                      edge_features=self.edge_feat, edge_dim=edge_dim)
             for _ in range(n_layers - 1)])
        self.layers.append(PNALayer(in_dim=hidden_dim, out_dim=out_dim, dropout=dropout, graph_norm=self.graph_norm,
                                    batch_norm=self.batch_norm, aggregators=self.aggregators, scalers=self.scalers,
                                    avg_d=self.avg_d,
                                    pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers,
                                    residual=self.residual, edge_features=self.edge_feat, edge_dim=edge_dim))
        self.MLP_layer = MLPReadout(out_dim, n_classes, self.dropout_2)
        if uses_sign_inv_net(self.pe_init):
            self.sign_inv_net = get_sign_inv_net(net_params)
            self.pe_proj = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, g, h, p, e, snorm_n):
        h = self.embedding_h(h)
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

    def loss(self, pred, labels):
        return nn.BCEWithLogitsLoss()(pred, labels)
