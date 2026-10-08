from nets.ZINC_graph_regression.gatedgcn_net import GatedGCNNet
from nets.ZINC_graph_regression.pna_net import PNANet


def gnn_model(MODEL_NAME, net_params):
    models = {
        'GatedGCN': GatedGCNNet,
        'PNA': PNANet,
    }
    return models[MODEL_NAME](net_params)
