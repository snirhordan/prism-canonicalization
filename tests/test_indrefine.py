"""The ``indrefine_nauty`` ordering of PRiSM (preset ``brec_indrefine``).

The comparison with the research code is in ``test_golden.py``; these tests check
the properties of the ordering itself.  They need ``pynauty``.
"""

import numpy as np
import pytest

pynauty = pytest.importorskip("pynauty")

from prism import PRiSM
from prism.adapters.transformer import canonicalize_eigenpairs
from prism.indrefine import detect_multiplicities

from conftest import graph_id_names
from helpers import er_adjacency, is_connected, permute

NAMES = graph_id_names()


def trivial_automorphism_group(A):
    g = pynauty.Graph(A.shape[0])
    for v in range(A.shape[0]):
        nbrs = [int(u) for u in np.nonzero(A[v])[0]]
        if nbrs:
            g.connect_vertex(v, nbrs)
    return pynauty.autgrp(g)[1] == 1.0


@pytest.fixture(scope="module")
def asymmetric_graph():
    """A connected graph with trivial automorphism group and a simple spectrum."""
    canon = PRiSM(operator="norm_lap", drop_trivial=False)
    for seed in range(200):
        A = er_adjacency(13, 0.4, seed)
        if not is_connected(A) or not trivial_automorphism_group(A):
            continue
        lam, _ = canon.eigendecompose(A)
        if np.diff(lam).min() > 1e-6:
            return A
    raise RuntimeError("no asymmetric graph found")


def scramble_eigenspaces(V, lam, rng):
    """Haar-random orthogonal transformation inside every eigenspace (grouped as the ordering groups)."""
    V = V.copy()
    col = 0
    for m in detect_multiplicities(lam):
        if m > 1:
            Q, R = np.linalg.qr(rng.standard_normal((m, m)))
            V[:, col:col + m] = V[:, col:col + m] @ (Q * np.sign(np.diag(R)))
        col += m
    return V


def make(**kw):
    """Float64 instance without the float32 storage step of the preset."""
    return PRiSM(operator="norm_lap", drop_trivial=False, ordering="indrefine_nauty", **kw)


def test_sign_flip_invariance(asymmetric_graph):
    A = asymmetric_graph
    canon = make()
    lam, V = canon.eigendecompose(A)
    base = canon.canonicalize(V, lam, adjacency=A)
    rng = np.random.default_rng(1)
    for _ in range(5):
        s = rng.choice([-1.0, 1.0], size=V.shape[1])
        np.testing.assert_allclose(canon.canonicalize(V * s, lam, adjacency=A), base,
                                   rtol=0, atol=1e-8)


def test_permutation_equivariance_on_an_asymmetric_graph(asymmetric_graph):
    """For a trivial automorphism group nauty's labelling is unique: relabeling the
    vertices permutes the rows of the canonical eigenvectors and of the vertex order."""
    A = asymmetric_graph
    canon = make()
    lam, V = canon.eigendecompose(A)
    base = canon.canonicalize(V, lam, return_info=True, adjacency=A)
    rng = np.random.default_rng(2)
    for _ in range(5):
        p = rng.permutation(A.shape[0])
        Ap = permute(A, p)
        lam_p, V_p = canon.eigendecompose(Ap)
        got = canon.canonicalize(V_p, lam_p, return_info=True, adjacency=Ap)
        np.testing.assert_allclose(got.V, base.V[p], rtol=0, atol=1e-8)
        # vertex i of the permuted graph is vertex p[i] of the original one
        np.testing.assert_array_equal(p[got.order], base.order)


@pytest.mark.parametrize("i", range(len(NAMES)), ids=NAMES)
def test_eigenspace_rotation_invariance(golden, i):
    """The output depends on the eigenspaces only, not on the basis chosen in each,
    for every graph (the colouring and the keys use only the projection matrices)."""
    A = golden.adj(i)
    canon = make()
    lam, V = canon.eigendecompose(A)
    base = canon.canonicalize(V, lam, adjacency=A)
    Vs = scramble_eigenspaces(V, lam, np.random.default_rng(10 + i))
    np.testing.assert_allclose(canon.canonicalize(Vs, lam, adjacency=A), base, rtol=0, atol=1e-7)


@pytest.mark.parametrize("i", range(len(NAMES)), ids=NAMES)
def test_canonical_columns_span_the_same_eigenspaces(golden, i):
    A = golden.adj(i)
    canon = make()
    lam, V = canon.eigendecompose(A)
    Vc = canon.canonicalize(V, lam, adjacency=A)
    np.testing.assert_allclose(Vc.T @ Vc, np.eye(Vc.shape[1]), atol=1e-10)
    col = 0
    for m in detect_multiplicities(lam):
        P0 = V[:, col:col + m] @ V[:, col:col + m].T
        P1 = Vc[:, col:col + m] @ Vc[:, col:col + m].T
        np.testing.assert_allclose(P1, P0, atol=1e-10)
        col += m


def test_info_fields(asymmetric_graph):
    A = asymmetric_graph
    canon = PRiSM.from_preset("brec_indrefine")
    lam, V = canon.eigendecompose(A)
    info = canon.canonicalize(V, lam, return_info=True, adjacency=A)
    n = A.shape[0]
    assert sorted(info.order.tolist()) == list(range(n))
    np.testing.assert_array_equal(info.rank[info.order], np.arange(n))
    assert sum(info.multiplicities) == lam.shape[0] and info.norm_sig.shape[0] == n
    np.testing.assert_array_equal(canon.canonicalize(V, lam, adjacency=A), info.V)


def test_pos_enc_and_spectrum_pass_their_adjacency(asymmetric_graph):
    sp = pytest.importorskip("scipy.sparse")
    A = asymmetric_graph
    canon = PRiSM.from_preset("brec_indrefine")
    pe = canon.pos_enc(A, k=8)
    assert pe.shape == (A.shape[0], 8) and pe.dtype == np.float32
    np.testing.assert_array_equal(canon.pos_enc(sp.csr_matrix(A), k=8), pe)
    lam, V = canon.eigendecompose(A)
    w = 8 + canon.window_buffer
    expected = canon.canonicalize(V[:, :w], lam[:w], adjacency=A)[:, :8]
    lam_k, V_k = canon.spectrum(A, k=8)
    np.testing.assert_array_equal(lam_k, lam[:8])
    np.testing.assert_array_equal(V_k, expected)


def test_transformer_adapter_passes_the_adjacency(asymmetric_graph):
    A = asymmetric_graph
    canon = PRiSM.from_preset("brec_indrefine")
    lam, V = canon.eigendecompose(A)
    num_vecs = 15
    pad = num_vecs - lam.shape[0]
    lam_p = np.concatenate([lam, np.full(pad, np.nan)]) if pad > 0 else lam[:num_vecs]
    V_p = (np.concatenate([V, np.full((V.shape[0], pad), np.nan)], axis=1)
           if pad > 0 else V[:, :num_vecs])
    out = canonicalize_eigenpairs(lam_p, V_p, canon, num_vecs, adjacency=A)
    info = canon.canonicalize(V, lam, return_info=True, adjacency=A)
    c = lam.shape[0]
    np.testing.assert_allclose(out["eigvecs"][:, :c], info.V.astype(np.float32))
    np.testing.assert_array_equal(out["canon_order"], info.order)
    np.testing.assert_array_equal(out["canon_rank"], info.rank)
    with pytest.raises(ValueError, match="adjacency"):
        canonicalize_eigenpairs(lam_p, V_p, canon, num_vecs)


def test_edgeless_and_tiny_graphs_do_not_crash():
    canon = PRiSM.from_preset("brec_indrefine")
    assert canon.pos_enc(np.zeros((1, 1)), k=4).shape == (1, 4)
    assert canon.pos_enc(np.zeros((3, 3)), k=4).shape == (3, 4)
    assert canon.pos_enc(np.array([[0.0, 1.0], [1.0, 0.0]]), k=4).shape == (2, 4)
