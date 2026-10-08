"""Runtime processing of the positional encoding of a batch."""

import torch

from nets.common import uses_sign_inv_net


def get_batch_pos_enc(model, batch_graphs, device, training):
    """Positional encoding of a batch (nodes, k), float32, ready for ``model.forward``.

    ``lap_pe``: random sign flip of every column (training only; one sign per column and batch).
    ``signnet``, ``map``, ``oap``, ``prism``: the sign-invariant network of the model is applied
    to the (nodes, k, 1) encoding and returns a (nodes, k, 1) tensor.
    """
    pos_enc = batch_graphs.ndata['pos_enc'].to(device).float()
    if model.pe_init == 'lap_pe':
        if training:
            sign_flip = torch.rand(pos_enc.size(1)).to(device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            pos_enc = pos_enc * sign_flip.unsqueeze(0)
    elif uses_sign_inv_net(model.pe_init):
        pos_enc = pos_enc.unsqueeze(-1)  # n x k -> n x k x 1
        pos_enc = model.sign_inv_net(batch_graphs, pos_enc)
        pos_enc = pos_enc.squeeze(-1)  # n x k x 1 -> n x k
    return pos_enc.float()
