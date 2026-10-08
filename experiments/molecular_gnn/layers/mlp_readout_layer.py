"""MLP readouts applied to the pooled graph representation.

Adapted from benchmarking-gnns (Dwivedi et al., MIT License); see
THIRD_PARTY_NOTICES.md.
"""

import torch.nn as nn
import torch.nn.functional as F


class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):  # L = number of hidden layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y


class MLPReadout2(nn.Module):
    """MLPReadout with dropout before every hidden layer (used by PNA on the OGB tasks)."""

    def __init__(self, input_dim, output_dim, dropout_2=0.0, L=2):
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        self.dropout_2 = dropout_2

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = F.dropout(y, self.dropout_2, training=self.training)
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y
