"""Sign-flip, permutation and O(m) invariance of the canonicalizers."""

import numpy as np
import pytest

from prism import PRiSM, gf2
from prism.core import detect_multiplicities

from helpers import generic_graph, permute, scramble_eigenspaces
from conftest import graph_id_names

CONFIGS = {
    "zinc_gatedgcn": lambda: PRiSM.from_preset("zinc_gatedgcn"),
    "hybrid_refined": lambda: PRiSM(operator="norm_adj", mult_mode="hybrid",
                                    refine_ordering=True),
    "qr_refined_full": lambda: PRiSM.from_preset("qr_refined_full"),
    "brec": lambda: PRiSM.from_preset("brec"),
    "transformer_refined": lambda: PRiSM.from_preset("transformer_refined"),
    "brec_qr_refined": lambda: PRiSM.from_preset("brec_qr_refined"),
    "rebuttal_comb_lap": lambda: PRiSM.from_preset("rebuttal_comb_lap"),
    "prism_gf2": lambda: PRiSM.from_preset("prism_gf2"),
}


def _tol(canon):
    return 1e-5 if canon.quantize_float32 else 1e-8


@pytest.fixture(scope="module")
def graph():
    return generic_graph([make() for make in CONFIGS.values()])


@pytest.mark.parametrize("name", CONFIGS)
def test_sign_flip_invariance(name, graph):
    canon = CONFIGS[name]()
    lam, V = canon.eigendecompose(graph)
    base = canon.canonicalize(V, lam)
    rng = np.random.default_rng(1)
    for _ in range(5):
        s = rng.choice([-1.0, 1.0], size=V.shape[1])
        np.testing.assert_allclose(canon.canonicalize(V * s, lam), base, rtol=0, atol=_tol(canon))


@pytest.mark.parametrize("name", CONFIGS)
def test_permutation_equivariance(name, graph):
    """Relabeling the vertices permutes the rows of the canonical eigenvectors."""
    canon = CONFIGS[name]()
    lam, V = canon.eigendecompose(graph)
    base = canon.canonicalize(V, lam)
    rng = np.random.default_rng(2)
    for _ in range(5):
        p = rng.permutation(graph.shape[0])
        lam_p, V_p = canon.eigendecompose(permute(graph, p))
        np.testing.assert_allclose(canon.canonicalize(V_p, lam_p), base[p], rtol=0,
                                   atol=_tol(canon))


@pytest.mark.parametrize("name", ["zinc_pna", "molpcba_gatedgcn", "molpcba_pna", "zinc_gatedgcn",
                                  "moltox21_gatedgcn", "moltox21_pna"])
def test_pos_enc_permutation_equivariance(name, graph):
    canon = PRiSM.from_preset(name)
    base = canon.pos_enc(graph)
    rng = np.random.default_rng(3)
    for _ in range(3):
        p = rng.permutation(graph.shape[0])
        np.testing.assert_allclose(canon.pos_enc(permute(graph, p)), base[p], rtol=1e-4, atol=1e-5)


# ----------------------------------------------------- O(m) invariance, any graph

# Algorithm-level check on float64 eigendecompositions of the molecular graphs:
# the legacy-rounding presets are used without their float32 storage step, since
# a float32 round trip breaks the exact eigenspace structure the test relies on.
SCRAMBLE = {
    "zinc_gatedgcn": lambda: PRiSM.from_preset("zinc_gatedgcn"),
    "hybrid_refined": CONFIGS["hybrid_refined"],
    "qr_refined_full": lambda: PRiSM.from_preset("qr_refined_full"),
    "legacy_hybrid": lambda: PRiSM(operator="norm_adj", tol_profile="legacy_round",
                                   mult_mode="hybrid", refine_ordering=False),
    "legacy_hybrid_refined": lambda: PRiSM(operator="norm_adj", tol_profile="legacy_round",
                                           mult_mode="hybrid", refine_ordering=True),
    "legacy_qr_refined": lambda: PRiSM(operator="norm_adj", tol_profile="legacy_round",
                                       mult_mode="qr", refine_ordering=True),
}
NAMES = graph_id_names()


@pytest.mark.parametrize("i", range(len(NAMES)), ids=NAMES)
@pytest.mark.parametrize("name", SCRAMBLE)
def test_eigenspace_rotation_invariance(golden, name, i):
    """The output depends on the eigenspaces only, not on the basis chosen in each.

    Holds for every graph, including those with large multiplicities; ties
    between vertices are broken by array index, which a basis change keeps.
    """
    canon = SCRAMBLE[name]()
    lam, V = golden.spec("norm_adj", i)
    base = canon.canonicalize(V, lam)
    Vs = scramble_eigenspaces(V, lam, np.random.default_rng(10 + i))
    np.testing.assert_allclose(canon.canonicalize(Vs, lam), base, rtol=0, atol=1e-7)


def test_canonical_columns_stay_orthonormal(golden):
    canon = PRiSM.from_preset("qr_refined_full")
    for i in range(len(NAMES)):
        lam, V = golden.spec("norm_adj", i)
        Vc = canon.canonicalize(V, lam)
        np.testing.assert_allclose(Vc.T @ Vc, np.eye(Vc.shape[1]), atol=1e-10)
        col = 0
        for m in detect_multiplicities(lam):
            P0 = V[:, col:col + m] @ V[:, col:col + m].T
            P1 = Vc[:, col:col + m] @ Vc[:, col:col + m].T
            np.testing.assert_allclose(P1, P0, atol=1e-10)  # same eigenspace
            col += m
