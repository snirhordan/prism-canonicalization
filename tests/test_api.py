"""Constructor validation, presets, selection logic and adapters."""

import subprocess
import sys

import numpy as np
import pytest

from prism import PRESETS, PRiSM, operators
from prism.adapters.transformer import canonicalize_eigenpairs

from helpers import er_adjacency


def test_every_preset_builds_and_roundtrips():
    for name in PRESETS:
        canon = PRiSM.from_preset(name)
        assert PRiSM(**canon.get_config()).get_config() == canon.get_config()
        assert repr(canon).startswith("PRiSM(")


def test_paper_cells_have_presets():
    for name in ["zinc_gatedgcn", "zinc_pna", "moltox21_gatedgcn", "moltox21_pna",
                 "molpcba_gatedgcn", "molpcba_pna", "transformer_refined", "brec",
                 "rebuttal_comb_lap", "qr_refined_full", "prism_gf2"]:
        assert name in PRESETS


def test_overrides_and_unknown_preset():
    assert PRiSM.from_preset("zinc_pna", k=4).k == 4
    with pytest.raises(KeyError):
        PRiSM.from_preset("no_such_preset")


@pytest.mark.parametrize("kwargs", [
    dict(operator="laplace"),
    dict(scope="partial"),
    dict(keep="middle"),
    dict(sign_rule="random"),
    dict(mult_mode="svd"),
    dict(weight="log"),
    dict(mult_mode="qr", refine_ordering=False),
    dict(sign_rule="gf2", refine_ordering=False),
    dict(sign_rule="gf2", mult_mode="qr", refine_ordering=True),
    dict(operator="comb_lap", weight="eigval"),
    dict(scope="full", window_buffer=3),
    dict(k=-1),
])
def test_invalid_configurations_are_rejected(kwargs):
    with pytest.raises(ValueError):
        PRiSM(**kwargs)


def test_import_does_not_pull_torch():
    code = ("import sys, prism; from prism import PRiSM; "
            "PRiSM.from_preset('zinc_pna').pos_enc([[0,1],[1,0]], 3); "
            "assert 'torch' not in sys.modules; assert 'dgl' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_pos_enc_shape_dtype_and_padding():
    A = er_adjacency(7, 0.5, 0)
    canon = PRiSM.from_preset("zinc_pna")
    pe = canon.pos_enc(A, k=10)
    assert pe.shape == (7, 10) and pe.dtype == np.float32
    assert np.all(pe[:, 6:] == 0)  # n - 1 = 6 non-trivial eigenpairs
    assert np.any(pe[:, :6] != 0)
    assert PRiSM.from_preset("zinc_pna").pos_enc(A, k=3).shape == (7, 3)
    with pytest.raises(ValueError):
        PRiSM().pos_enc(A)  # no k anywhere


def test_pos_enc_accepts_sparse_input():
    sp = pytest.importorskip("scipy.sparse")
    A = er_adjacency(9, 0.4, 3)
    canon = PRiSM.from_preset("zinc_gatedgcn")
    np.testing.assert_array_equal(canon.pos_enc(sp.csr_matrix(A)), canon.pos_enc(A))


def test_single_node_and_edgeless_graphs_do_not_crash():
    canon = PRiSM.from_preset("zinc_pna")
    assert canon.pos_enc(np.zeros((1, 1)), k=4).shape == (1, 4)
    assert canon.pos_enc(np.zeros((3, 3)), k=4).shape == (3, 4)


def test_spectrum_selection_full_scope_keeps_trailing_columns():
    A = er_adjacency(12, 0.4, 1)
    canon = PRiSM.from_preset("qr_refined_full")
    lam, V = canon.eigendecompose(A)
    full = canon.canonicalize(V, lam)
    lam_k, V_k = canon.spectrum(A, k=4)
    np.testing.assert_array_equal(lam_k, lam[-4:])
    np.testing.assert_array_equal(V_k, full[:, -4:])


def test_spectrum_selection_window_canonicalizes_only_the_window():
    A = er_adjacency(12, 0.4, 1)
    canon = PRiSM.from_preset("moltox21_pna")
    lam, V = canon.eigendecompose(A)
    lam_k, V_k = canon.spectrum(A, k=5)
    np.testing.assert_array_equal(lam_k, lam[-5:])
    np.testing.assert_array_equal(V_k, canon.canonicalize(V[:, -5:], lam[-5:]))


def test_spectrum_selection_window_with_buffer_keeps_first_columns():
    A = er_adjacency(14, 0.4, 2)
    canon = PRiSM.from_preset("transformer_refined", k=4)
    lam, V = canon.eigendecompose(A)
    assert lam.shape[0] == 14  # the trivial Laplacian eigenpair is kept
    lam_k, V_k = canon.spectrum(A)
    np.testing.assert_array_equal(lam_k, lam[:4])
    np.testing.assert_array_equal(V_k, canon.canonicalize(V[:, :7], lam[:7])[:, :4])


def test_weights_follow_the_adjacency_eigenvalue():
    A = er_adjacency(12, 0.4, 1)
    plain = PRiSM(operator="norm_adj", scope="full", k=5)
    lam, V = plain.spectrum(A)
    for weight, param, expected in [
        ("eigval", None, np.sqrt(np.maximum(lam, 0))),
        ("inv_sqrt_lap", 1e-6, 1 / np.sqrt(np.maximum(1 - lam, 1e-6))),
        ("heat", 2.0, np.exp(-2.0 * (1 - lam))),
    ]:
        w = PRiSM(operator="norm_adj", scope="full", k=5, weight=weight, weight_param=param)
        np.testing.assert_allclose(w.pos_enc(A), (V * expected).astype(np.float32),
                                   rtol=1e-6, atol=1e-7)


def test_canonicalize_info_fields():
    A = er_adjacency(10, 0.5, 4)
    canon = PRiSM.from_preset("transformer_refined")
    lam, V = canon.eigendecompose(A)
    info = canon.canonicalize(V, lam, return_info=True)
    n = A.shape[0]
    assert sorted(info.order.tolist()) == list(range(n))
    np.testing.assert_array_equal(info.rank[info.order], np.arange(n))
    assert info.norm_sig.shape[0] == n and sum(info.multiplicities) == lam.shape[0]


def test_transformer_adapter_handles_nan_padding():
    A = er_adjacency(9, 0.5, 5)
    canon = PRiSM.from_preset("transformer_refined")
    lam, V = canon.eigendecompose(A)
    num_vecs = 12
    pad = num_vecs - lam.shape[0]
    lam_p = np.concatenate([lam, np.full(pad, np.nan)])
    V_p = np.concatenate([V, np.full((V.shape[0], pad), np.nan)], axis=1)
    out = canonicalize_eigenpairs(lam_p, V_p, canon, num_vecs)
    info = canon.canonicalize(V, lam, return_info=True)
    np.testing.assert_allclose(out["eigvecs"][:, :lam.shape[0]], info.V.astype(np.float32))
    assert np.all(np.isnan(out["eigvecs"][:, lam.shape[0]:]))
    assert out["norm_sig"].shape == (9, num_vecs)
    np.testing.assert_array_equal(out["canon_rank"], info.rank)
    np.testing.assert_array_equal(out["canon_order"], info.order)


def test_dgl_adapter():
    dgl = pytest.importorskip("dgl")
    torch = pytest.importorskip("torch")
    from prism.adapters.dgl import add_pos_enc

    A = er_adjacency(9, 0.5, 6)
    src, dst = np.nonzero(A)
    g = dgl.graph((torch.from_numpy(src), torch.from_numpy(dst)), num_nodes=9)
    canon = PRiSM.from_preset("zinc_pna")
    add_pos_enc(g, canon, k=8)
    np.testing.assert_array_equal(g.ndata["pos_enc"].numpy(), canon.pos_enc(A, k=8))


def test_operators_are_symmetric_psd_where_expected():
    A = er_adjacency(10, 0.5, 7)
    Ln = operators.normalized_laplacian(A)
    Lc = operators.combinatorial_laplacian(A)
    assert np.all(np.linalg.eigvalsh(Ln) > -1e-12)
    np.testing.assert_allclose(Lc.sum(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(operators.normalized_adjacency(A) + Ln, np.eye(10), atol=1e-14)
