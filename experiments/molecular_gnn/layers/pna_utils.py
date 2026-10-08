"""Aggregators, scalers and dense building blocks of PNA.

PNA: Principal Neighbourhood Aggregation, Corso, Cavalleri, Beaini, Lio and Velickovic,
https://arxiv.org/abs/2004.05718.  Adapted from the reference implementation
(https://github.com/lukecavabarrett/pna, MIT License) as shipped in gnn-lspe (Dwivedi et
al., MIT License); see THIRD_PARTY_NOTICES.md.
"""

import numpy as np
import torch
import torch.nn as nn

# PNA aggregators -----------------------------------------------------------------------------

EPS = 1e-5


def aggregate_mean(h):
    return torch.mean(h, dim=1)


def aggregate_max(h):
    return torch.max(h, dim=1)[0]


def aggregate_min(h):
    return torch.min(h, dim=1)[0]


def aggregate_var(h):
    h_mean_squares = torch.mean(h * h, dim=-2)
    h_mean = torch.mean(h, dim=-2)
    var = torch.relu(h_mean_squares - h_mean * h_mean)
    return var


def aggregate_std(h):
    return torch.sqrt(aggregate_var(h) + EPS)


def aggregate_sum(h):
    return torch.sum(h, dim=1)


AGGREGATORS = {'mean': aggregate_mean, 'sum': aggregate_sum, 'max': aggregate_max, 'min': aggregate_min,
               'std': aggregate_std}

# PNA scalers ---------------------------------------------------------------------------------
# each scaler takes the aggregated features, the degree D and avg_d (a dictionary of averages
# over the training set) and returns the scaled features


def scale_identity(h, D=None, avg_d=None):
    return h


def scale_amplification(h, D, avg_d):
    # log(D + 1) / d * h     where d is the average of the ``log(D + 1)`` in the training set
    return h * (np.log(D + 1) / avg_d["log"])


def scale_attenuation(h, D, avg_d):
    # (log(D + 1))^-1 / d * X     where d is the average of the ``log(D + 1))^-1`` in the training set
    return h * (avg_d["log"] / np.log(D + 1))


SCALERS = {'identity': scale_identity, 'amplification': scale_amplification, 'attenuation': scale_attenuation}

# Dense layers --------------------------------------------------------------------------------

SUPPORTED_ACTIVATION_MAP = {'ReLU', 'Sigmoid', 'Tanh', 'ELU', 'SELU', 'GLU', 'LeakyReLU', 'Softplus', 'None'}


def get_activation(activation):
    """ returns the activation function represented by the input string """
    if activation and callable(activation):
        # activation is already a function
        return activation
    # search in SUPPORTED_ACTIVATION_MAP a torch.nn.modules.activation
    activation = [x for x in SUPPORTED_ACTIVATION_MAP if activation.lower() == x.lower()]
    assert len(activation) == 1 and isinstance(activation[0], str), 'Unhandled activation function'
    activation = activation[0]
    if activation.lower() == 'none':
        return None
    return vars(torch.nn.modules.activation)[activation]()


class FCLayer(nn.Module):
    r"""
    A fully connected layer: dense layer, activation, dropout (if any), batch normalization (if any).
    The weight is initialised with ``xavier_uniform_`` (gain ``1 / in_size``) and the bias with zeros.
    """

    def __init__(self, in_size, out_size, activation='relu', dropout=0., b_norm=False, bias=True, init_fn=None,
                 device='cpu'):
        super(FCLayer, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.bias = bias
        self.linear = nn.Linear(in_size, out_size, bias=bias).to(device)
        self.dropout = None
        self.b_norm = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        if b_norm:
            self.b_norm = nn.BatchNorm1d(out_size).to(device)
        self.activation = get_activation(activation)
        self.init_fn = nn.init.xavier_uniform_

        self.reset_parameters()

    def reset_parameters(self, init_fn=None):
        init_fn = init_fn or self.init_fn
        if init_fn is not None:
            init_fn(self.linear.weight, 1 / self.in_size)
        if self.bias:
            self.linear.bias.data.zero_()

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation(h)
        if self.dropout is not None:
            h = self.dropout(h)
        if self.b_norm is not None:
            if h.shape[1] != self.out_size:
                h = self.b_norm(h.transpose(1, 2)).transpose(1, 2)
            else:
                h = self.b_norm(h)
        return h

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_size) + ' -> ' \
               + str(self.out_size) + ')'


class MLP(nn.Module):
    """Simple multi-layer perceptron, built of a series of FCLayers."""

    def __init__(self, in_size, hidden_size, out_size, layers, mid_activation='relu', last_activation='none',
                 dropout=0., mid_b_norm=False, last_b_norm=False, device='cpu'):
        super(MLP, self).__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size

        self.fully_connected = nn.ModuleList()
        if layers <= 1:
            self.fully_connected.append(FCLayer(in_size, out_size, activation=last_activation, b_norm=last_b_norm,
                                                device=device, dropout=dropout))
        else:
            self.fully_connected.append(FCLayer(in_size, hidden_size, activation=mid_activation, b_norm=mid_b_norm,
                                                device=device, dropout=dropout))
            for _ in range(layers - 2):
                self.fully_connected.append(FCLayer(hidden_size, hidden_size, activation=mid_activation,
                                                    b_norm=mid_b_norm, device=device, dropout=dropout))
            self.fully_connected.append(FCLayer(hidden_size, out_size, activation=last_activation, b_norm=last_b_norm,
                                                device=device, dropout=dropout))

    def forward(self, x):
        for fc in self.fully_connected:
            x = fc(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_size) + ' -> ' \
               + str(self.out_size) + ')'
