import copy
from types import SimpleNamespace

import networkx as nx
import pytest
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx

from wl_transformers import (Transform, count_parameters, create_gnn, create_transformer,
                             make_canonicalizer)

E = 32


def tiny_batch(pe_type, refine, num_vecs, node_dim=None):
    tr = Transform(num_vecs, 2, True, False, canonicalizer=make_canonicalizer(pe_type, refine))
    graphs = []
    for G in (nx.cycle_graph(6), nx.path_graph(7), nx.petersen_graph()):
        d = from_networkx(G)
        d.x = torch.ones(G.number_of_nodes(), 1)
        d.edge_attr = torch.ones(d.edge_index.size(1), dtype=torch.long)
        graphs.append(tr(d))
    return next(iter(DataLoader(graphs, batch_size=3)))


def cfg(pe_type, num_vecs, **kw):
    base = dict(num_phi_layers=2, num_vecs=num_vecs, max_degree=100, eigen_dim=E, degree_dim=16,
                num_layers=2, embed_dim=E, num_heads=4, dropout=0.1, attention_dropout=0.1,
                bias=True, stochastic_depth=True, token_ln=True, pe_type=pe_type, pooling="mean")
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.mark.parametrize("pe_type,refine", [
    ("LPE", False), ("CanonLPE_min_rope", True), ("CanonLPE_min_sinid", True),
    ("CanonLPE_min", True), ("MAP_min", False), ("OAP_min", False),
])
def test_transformer_forward(pe_type, refine):
    batch = tiny_batch(pe_type, refine, 96)
    atom = torch.nn.Linear(1, E)
    bond = torch.nn.Embedding(2, E)
    model = create_transformer(cfg(pe_type, 96), atom, bond, output_dim=3, device="cpu")
    model.eval()
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (3, 3) and torch.isfinite(out).all()


def test_rank_embeddings_change_the_output():
    batch = tiny_batch("CanonLPE_min_rope", True, 96)
    atom, bond = torch.nn.Linear(1, E), torch.nn.Embedding(2, E)
    model = create_transformer(cfg("CanonLPE_min_rope", 96), atom, bond, 3, "cpu").eval()
    with torch.no_grad():
        with_rank = model(copy.deepcopy(batch))
        no_rank = copy.deepcopy(batch)
        del no_rank.canon_rank
        without = model(no_rank)
    assert not torch.allclose(with_rank, without)


@pytest.mark.parametrize("pe_type,expected", [
    ("MAP", 12656), ("OAP", 12656), ("CanonLPE_noeps", 12656), ("CanonLPE", 12664),
    ("QRRefinedLPE", 12664), ("MAP_min", 9520),
])
def test_matched_brec_gnn_parameter_counts(pe_type, expected):
    c = cfg(pe_type, 8, num_layers=2, embed_dim=32)
    model = create_gnn(c, torch.nn.Linear(1, 32), torch.nn.Embedding(2, 32), 16, "cpu")
    assert count_parameters(model) == expected


def test_gnn_forward_and_rank_types_rejected():
    batch = tiny_batch("CanonLPE_noeps", False, 8)
    model = create_gnn(cfg("CanonLPE_noeps", 8), torch.nn.Linear(1, E), torch.nn.Embedding(2, E),
                       16, "cpu").eval()
    with torch.no_grad():
        assert model(batch).shape == (3, 16)
    with pytest.raises(ValueError):
        create_gnn(cfg("CanonLPE_min_rope", 8), torch.nn.Linear(1, E), torch.nn.Embedding(2, E),
                   16, "cpu")


def test_rope_forward_under_autocast():
    # Alchemy-style linear atom and bond encoders, bfloat16 autocast on CPU
    batch = tiny_batch("CanonLPE_min_rope", True, 96)
    batch.edge_attr = torch.ones(batch.edge_index.size(1), 1)
    atom, bond = torch.nn.Linear(1, E), torch.nn.Linear(1, E)
    model = create_transformer(cfg("CanonLPE_min_rope", 96), atom, bond, 3, "cpu").eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(batch)
    assert torch.isfinite(out.float()).all()
