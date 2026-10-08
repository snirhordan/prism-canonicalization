from nets.OGBMOL_graph_classification.gatedgcn_net import GatedGCNNet
from nets.OGBMOL_graph_classification.pna_net import PNANet


def gnn_model(MODEL_NAME, net_params):
    models = {
        'GatedGCN': GatedGCNNet,
        'PNA': PNANet,
    }
    return models[MODEL_NAME](net_params)
