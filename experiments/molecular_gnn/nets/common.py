"""Pieces shared by the ZINC and OGB networks: positional-encoding modes, sign-invariant network, readout.

``pe_init`` selects how the positional encoding enters the model.

``lap_pe``    raw Laplacian eigenvectors, random sign flip per column and batch during
              training, added to the node embedding.
``signnet``   raw Laplacian eigenvectors through the sign-invariant network enc(x) + enc(-x),
              concatenated with the node embedding and projected.
``map``, ``oap``, ``prism``
              canonicalized eigenvectors (MAP, OAP, PRiSM) through the same network without
              the symmetrization, concatenated with the node embedding and projected.

For ``prism`` the preset in ``net_params['prism_preset']`` (see ``prism.presets``) picks the
canonicalization variant.
"""

import dgl

from layers.deepsigns import MaskedGINDeepSigns

PE_INITS = ('lap_pe', 'signnet', 'map', 'oap', 'prism')
SIGN_INV_PE_INITS = ('signnet', 'map', 'oap', 'prism')


def check_pe_init(pe_init):
    if pe_init not in PE_INITS:
        raise ValueError("pe_init must be one of %s, got %r" % (PE_INITS, pe_init))


def uses_sign_inv_net(pe_init):
    check_pe_init(pe_init)
    return pe_init in SIGN_INV_PE_INITS


def get_sign_inv_net(net_params):
    if net_params.get('sign_inv_net') != 'masked_gin':
        raise ValueError("sign_inv_net must be 'masked_gin'")
    use_signnet = net_params['pe_init'] == 'signnet'
    return MaskedGINDeepSigns(1, net_params['hidden_dim'], net_params['phi_out_dim'],
                              net_params['sign_inv_layers'], net_params['pos_enc_dim'],
                              use_bn=True, dropout=net_params['dropout'],
                              activation=net_params['sign_inv_activation'], use_signnet=use_signnet)


def readout(g, h, kind):
    g.ndata['h'] = h
    if kind == 'sum':
        return dgl.sum_nodes(g, 'h')
    if kind == 'max':
        return dgl.max_nodes(g, 'h')
    return dgl.mean_nodes(g, 'h')  # default readout is mean nodes
