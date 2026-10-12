"""Golden equivalence: every preset reproduces the original (live) implementation.

``tests/fixtures/golden.npz`` stores small graphs, their eigendecompositions and
the outputs of the original code of each variant (see ``dev/make_golden.py``).

* Function-level tests feed the stored eigendecomposition to the preset and
  compare canonical eigenvectors (atol 1e-8) and vertex orders (exact).  They
  do not depend on the LAPACK build.
* PE-level tests run the whole pipeline (eigendecomposition, selection,
  weighting, float32 cast) and compare against the original DGL wrappers up to
  float32 resolution.
"""

import numpy as np
import pytest

from prism import PRiSM
from prism.baselines import lap_pe
from prism import gf2

from conftest import graph_id_names

NAMES = graph_id_names()
IDS = list(range(len(NAMES)))
ATOL = 1e-8


def _id(i):
    return NAMES[i]


# --------------------------------------------------------------- function level

CANON_CASES = [
    # (label, constructor, operator of the stored spectrum, golden key)
    ("zinc_gatedgcn", lambda: PRiSM.from_preset("zinc_gatedgcn"), "norm_adj", "fresh_m3"),
    ("hybrid_refined_adaptive",
     lambda: PRiSM(operator="norm_adj", mult_mode="hybrid", refine_ordering=True),
     "norm_adj", "fresh_m3_refined"),
    ("zinc_pna", lambda: PRiSM.from_preset("zinc_pna"), "norm_adj", "fresh_qr_refined"),
    ("qr_refined_full", lambda: PRiSM.from_preset("qr_refined_full"), "norm_adj",
     "fresh_qr_refined"),
    ("molpcba_pna", lambda: PRiSM.from_preset("molpcba_pna"), "norm_adj", "fresh_qr_refined"),
    ("brec", lambda: PRiSM.from_preset("brec"), "norm_lap_f32", "brec_m3"),
    ("transformer_refined", lambda: PRiSM.from_preset("transformer_refined"),
     "norm_lap_f32", "brec_m3_refined"),
    ("brec_qr_refined", lambda: PRiSM.from_preset("brec_qr_refined"), "norm_lap_f32",
     "brec_qr_refined"),
    ("rebuttal_comb_lap", lambda: PRiSM.from_preset("rebuttal_comb_lap"), "comb_lap",
     "brec_m3_refined_comb"),
]


@pytest.mark.parametrize("i", IDS, ids=_id)
@pytest.mark.parametrize("label,make,op,key", CANON_CASES, ids=[c[0] for c in CANON_CASES])
def test_canonicalize_matches_live(golden, label, make, op, key, i):
    lam, V = golden.spec(op, i)
    expected = golden.fn(key, i, op)
    got = make().canonicalize(V, lam)
    np.testing.assert_allclose(got, expected, rtol=0, atol=ATOL)


@pytest.mark.parametrize("i", IDS, ids=_id)
def test_transformer_order_rank_and_signature(golden, i):
    """Vertex order, rank and norm signature of the transformer harness."""
    lam, V = golden.spec("norm_lap_f32", i)
    info = PRiSM.from_preset("transformer_refined").canonicalize(V, lam, return_info=True)
    order = golden.fn_order("brec_m3_refined", i)
    np.testing.assert_array_equal(info.order, order)
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    np.testing.assert_array_equal(info.rank, rank)
    np.testing.assert_allclose(info.norm_sig, golden.fn_shape("brec_norm_sig", i, info.norm_sig.shape),
                               rtol=0, atol=ATOL)


@pytest.mark.parametrize("i", IDS, ids=_id)
def test_transformer_window(golden, i):
    """num_vecs + buffer eigenpairs canonicalized, the first num_vecs kept."""
    lam, V = golden.spec("norm_lap_f32", i)
    canon = PRiSM.from_preset("transformer_refined", k=6)
    w = min(lam.shape[0], 6 + canon.window_buffer)
    got = canon.canonicalize(V[:, :w], lam[:w])[:, :min(lam.shape[0], 6)]
    np.testing.assert_allclose(got, golden.fn_shape("brec_window_k6", i, got.shape),
                               rtol=0, atol=ATOL)


# ------------------------------------- pre-fix rank (MolClinTox cell of the transformer table)

PREFIX_KEY = "prefix_m3_refined"


def _need_prefix_keys(golden):
    if f"fn__{PREFIX_KEY}__V" not in golden.z.files:
        pytest.skip("golden.npz has no pre-fix keys; run dev/make_golden.py --only-prefix")


@pytest.mark.parametrize("i", IDS, ids=_id)
def test_prefix_rank_matches_live(golden, i):
    """Preset ``transformer_refined_prefix`` against the research code before commit f5e4255:
    eigenvectors (atol 1e-8) and the stored vertex order, which is the norm-signature order
    and not the refined one (exact)."""
    _need_prefix_keys(golden)
    lam, V = golden.spec("norm_lap_f32", i)
    info = PRiSM.from_preset("transformer_refined_prefix").canonicalize(V, lam, return_info=True)
    np.testing.assert_allclose(info.V, golden.fn(PREFIX_KEY, i, "norm_lap_f32"), rtol=0, atol=ATOL)
    order = golden.fn_order(PREFIX_KEY, i)
    np.testing.assert_array_equal(info.order, order)
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    np.testing.assert_array_equal(info.rank, rank)


@pytest.mark.parametrize("i", IDS, ids=_id)
def test_prefix_rank_only_changes_the_reported_order(golden, i):
    """The knob does not touch the eigenvectors or the signature; the fixed preset still
    reports the refined order of the post-fix research code."""
    lam, V = golden.spec("norm_lap_f32", i)
    pre = PRiSM.from_preset("transformer_refined_prefix").canonicalize(V, lam, return_info=True)
    fixed = PRiSM.from_preset("transformer_refined").canonicalize(V, lam, return_info=True)
    np.testing.assert_array_equal(pre.V, fixed.V)
    np.testing.assert_array_equal(pre.norm_sig, fixed.norm_sig)
    np.testing.assert_array_equal(fixed.order, golden.fn_order("brec_m3_refined", i))


def test_prefix_and_fixed_ranks_differ_on_the_fixture(golden):
    """The knob is not vacuous: the two reported orders differ on the stored graphs."""
    _need_prefix_keys(golden)
    differing = sum(
        not np.array_equal(golden.fn_order(PREFIX_KEY, i), golden.fn_order("brec_m3_refined", i))
        for i in IDS)
    assert differing > len(IDS) // 2


# ----------------------------------------------- IndRefineNauty (BREC PRiSM row)

IR_KEYS = [("brec_indrefine", None), ("brec_indrefine_window_k8", 8)]


@pytest.mark.parametrize("i", IDS, ids=_id)
@pytest.mark.parametrize("key,num_vecs", IR_KEYS, ids=[k for k, _ in IR_KEYS])
def test_indrefine_nauty_matches_live(golden, key, num_vecs, i):
    """Preset ``brec_indrefine`` against the live ``indrefine_nauty_canonicalize``: eigenvectors
    (atol 1e-8) and the nauty vertex order (exact), on the full stored spectrum and, as the BREC
    harness does, on the first ``num_vecs + 3`` eigenpairs."""
    pytest.importorskip("pynauty")
    if f"fn__{key}__V" not in golden.z.files:
        pytest.skip("golden.npz has no IndRefineNauty keys; run dev/make_golden.py --only-indrefine")
    lam, V = golden.spec("norm_lap_f32", i)
    canon = PRiSM.from_preset("brec_indrefine")
    if num_vecs is None:
        w, keep = lam.shape[0], lam.shape[0]
    else:
        w, keep = min(lam.shape[0], num_vecs + canon.window_buffer), min(lam.shape[0], num_vecs)
    info = canon.canonicalize(V[:, :w], lam[:w], return_info=True, adjacency=golden.adj(i))
    np.testing.assert_allclose(info.V[:, :keep],
                               golden.fn_shape(key, i, (V.shape[0], keep)), rtol=0, atol=ATOL)
    order = golden.fn_order(key, i)
    np.testing.assert_array_equal(info.order, order)
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    np.testing.assert_array_equal(info.rank, rank)


# --------------------------------------------------------------------- PE level

PE_PRESETS = ["zinc_gatedgcn", "zinc_gatedgcn_heat_t2", "zinc_pna", "moltox21_gatedgcn",
              "moltox21_pna", "molpcba_gatedgcn", "molpcba_gatedgcn_plain", "molpcba_pna"]


def _pe_ks():
    try:
        from conftest import Golden
        g = Golden()
    except Exception:
        return []
    out = []
    for key in g.keys("pe__"):
        _, name, kk = key.split("__")
        if name in PE_PRESETS:
            out.append((name, int(kk[1:])))
    return out


@pytest.mark.parametrize("i", IDS, ids=_id)
@pytest.mark.parametrize("preset,k", _pe_ks(), ids=lambda v: str(v))
def test_pos_enc_matches_live(golden, preset, k, i):
    got = PRiSM.from_preset(preset).pos_enc(golden.adj(i), k=k)
    expected = golden.pe(f"pe__{preset}__k{k}", i)
    assert got.dtype == np.float32 and got.shape == expected.shape
    # float32 resolution: 8 ulps relative, 1e-7 absolute
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-7)


# -------------------------------------------------------------------- baselines

torch = pytest.importorskip("torch")


def _baseline_ks(name):
    try:
        from conftest import Golden
        g = Golden()
    except Exception:
        return []
    return [int(key.split("__")[2][1:]) for key in g.keys(f"pe__{name}__")]


@pytest.mark.parametrize("i", IDS, ids=_id)
@pytest.mark.parametrize("k", _baseline_ks("map"))
def test_map_gnn_matches_live(golden, k, i):
    from prism.baselines import MAP

    got = MAP.from_profile("gnn").pos_enc(golden.adj(i), k)
    np.testing.assert_allclose(got, golden.pe(f"pe__map__k{k}", i), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("i", IDS, ids=_id)
@pytest.mark.parametrize("k", _baseline_ks("oap"))
def test_oap_gnn_matches_live(golden, k, i):
    from prism.baselines import OAP

    got = OAP.from_profile("gnn").pos_enc(golden.adj(i), k)
    np.testing.assert_allclose(got, golden.pe(f"pe__oap__k{k}", i), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("i", IDS, ids=_id)
@pytest.mark.parametrize("op,key,cls", [
    ("norm_lap_f32", "brec_map", "MAP"), ("norm_lap_f32", "brec_oap", "OAP"),
    ("comb_lap", "brec_map_comb", "MAP"), ("comb_lap", "brec_oap_comb", "OAP"),
])
def test_map_oap_brec_profile_matches_live(golden, op, key, cls, i):
    from prism import baselines

    lam, V = golden.spec(op, i)
    got = getattr(baselines, cls).from_profile("brec").canonicalize(V, lam)
    np.testing.assert_allclose(got, golden.fn(key, i, op), rtol=0, atol=ATOL)


@pytest.mark.parametrize("i", IDS, ids=_id)
@pytest.mark.parametrize("k", _baseline_ks("lap"))
def test_lap_pe_matches_live_up_to_sign(golden, k, i):
    """LapPE leaves signs to the eigensolver; compare magnitudes where the
    spectrum is simple (a repeated eigenvalue has no canonical basis)."""
    A = golden.adj(i)
    lam, _ = golden.spec("norm_adj", i)
    from prism.core import detect_multiplicities

    if max(detect_multiplicities(lam)) > 1:
        pytest.skip("repeated eigenvalue: eigenvectors are basis dependent")
    got = lap_pe(A, k)
    np.testing.assert_allclose(np.abs(got), np.abs(golden.pe(f"pe__lap__k{k}", i)),
                               rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------- GF(2) reference

def _gf2_cases():
    try:
        from conftest import Golden
        return list(range(len(Golden().z["gf2__sizes"])))
    except Exception:
        return []


@pytest.mark.parametrize("t", _gf2_cases())
def test_gf2_algorithm1_matches_reference(golden, t):
    z = golden.z
    n = int(z["gf2__sizes"][t])
    U = z["gf2__U"][z["gf2__U_off"][t]:z["gf2__U_off"][t + 1]].reshape(n, n)
    res = gf2.prism_canonicalize(U)
    s = z["gf2__s"][z["gf2__s_off"][t]:z["gf2__s_off"][t + 1]]
    canon = z["gf2__canon"][z["gf2__canon_off"][t]:z["gf2__canon_off"][t + 1]].reshape(n, n)
    part = z["gf2__part"][z["gf2__part_off"][t]:z["gf2__part_off"][t + 1]]
    np.testing.assert_array_equal(res.sign_vector, s)
    np.testing.assert_array_equal(res.partition, part)
    np.testing.assert_allclose(res.canonical, canon, rtol=0, atol=ATOL)


def test_gf2_fastsign_matches_reference(golden):
    z = golden.z
    for slot, t in enumerate(z["gf2__fast_idx"]):
        n = int(z["gf2__sizes"][t])
        U = z["gf2__U"][z["gf2__U_off"][t]:z["gf2__U_off"][t + 1]].reshape(n, n)
        ref = z["gf2__fast"][z["gf2__fast_off"][slot]:z["gf2__fast_off"][slot + 1]]
        np.testing.assert_array_equal(gf2.fastsign(U), ref)
