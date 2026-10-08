import networkx as nx
import numpy as np
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

from wl_transformers import Transform, make_canonicalizer

NUM_VECS = 96


def make_graph(G):
    data = from_networkx(G)
    data.x = torch.ones(G.number_of_nodes(), 1)
    return data


def transform(pe_type, refine=False, num_vecs=NUM_VECS):
    return Transform(num_vecs, 2, True, False,
                     canonicalizer=make_canonicalizer(pe_type, refine))


def generic_graph(n=12, seed=3):
    G = nx.gnp_random_graph(n, 0.4, seed=seed)
    while not nx.is_connected(G):
        seed += 1
        G = nx.gnp_random_graph(n, 0.4, seed=seed)
    return G


@pytest.mark.parametrize("pe_type,refine", [
    ("LPE", False), ("CanonLPE_min_rope", True), ("CanonLPE_min", False),
    ("QRRefinedLPE", False), ("MAP_min", False), ("OAP_min", False),
])
def test_shapes_and_padding(pe_type, refine):
    G = nx.petersen_graph()  # eigenvalue multiplicities 4 and 5
    d = transform(pe_type, refine)(make_graph(G))
    n = G.number_of_nodes()
    assert d.eigvecs.shape == (n, NUM_VECS)
    assert d.eigvals.shape == (n, NUM_VECS)
    assert torch.isnan(d.eigvecs[:, n:]).all() and not torch.isnan(d.eigvecs[:, :n]).any()
    assert d.token_index.shape[0] == 2 and d.real_edges.dtype == torch.bool


def test_prism_attaches_canonical_rank_and_baselines_do_not():
    g = make_graph(nx.petersen_graph())
    d = transform("CanonLPE_min_rope", True)(g.clone())
    assert sorted(d.canon_order.tolist()) == list(range(10))
    assert torch.equal(d.canon_rank[d.canon_order], torch.arange(10))
    for pe in ("LPE", "MAP_min", "OAP_min"):
        assert not hasattr(transform(pe)(g.clone()), "canon_rank")


@pytest.mark.parametrize("pe_type,refine", [("CanonLPE_min_rope", True), ("CanonLPE_min", False)])
def test_prism_pe_is_permutation_equivariant(pe_type, refine):
    G = generic_graph()
    n = G.number_of_nodes()
    perm = np.random.RandomState(0).permutation(n)
    new_of = {int(old): new for new, old in enumerate(perm)}
    H = nx.Graph()
    H.add_nodes_from(range(n))  # keep the node order 0..n-1
    H.add_edges_from((new_of[u], new_of[v]) for u, v in G.edges)
    t = transform(pe_type, refine)
    a, b = t(make_graph(G)), t(make_graph(H))
    # vertex `perm[i]` of G is vertex i of H
    va = a.eigvecs.numpy()[perm][:, :n]
    vb = b.eigvecs.numpy()[:, :n]
    assert np.allclose(va, vb, atol=1e-4)
    assert np.array_equal(a.canon_rank.numpy()[perm], b.canon_rank.numpy())


def test_lpe_has_no_canonicalization_buffer():
    G = generic_graph(n=120, seed=1)
    d = transform("LPE")(make_graph(G))  # n > num_vecs: partial eigendecomposition
    assert d.eigvecs.shape == (120, NUM_VECS)
    d2 = transform("CanonLPE_min", False)(make_graph(G))
    assert d2.eigvecs.shape == (120, NUM_VECS)  # canon_buffer eigenpairs dropped again
