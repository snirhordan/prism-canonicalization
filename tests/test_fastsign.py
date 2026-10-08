"""FastSign (Algorithm 2) and its relation to Algorithm 1 and to the hybrid."""

import numpy as np
import pytest

from prism import PRiSM, gf2

from helpers import generic_graph, permute


def random_orthogonal_eigvecs(n, seed):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    return np.linalg.eigh((M + M.T) / 2)[1]


@pytest.mark.parametrize("n,seed", [(5, 0), (8, 1), (11, 2), (16, 3)])
def test_fastsign_is_sign_and_permutation_invariant(n, seed):
    U = random_orthogonal_eigvecs(n, seed)
    assert gf2.is_injective_signature(U)
    canon = PRiSM()
    base, _ = canon.fastsign(U)
    rng = np.random.default_rng(seed + 100)
    for _ in range(5):
        t = rng.choice([-1.0, 1.0], size=n)
        p = rng.permutation(n)
        got, _ = canon.fastsign((U * t)[p])
        np.testing.assert_allclose(got, base, rtol=0, atol=1e-10)


@pytest.mark.parametrize("n,seed", [(5, 0), (8, 1), (11, 2), (16, 3), (20, 4)])
def test_fastsign_equals_solve_when_signature_is_injective(n, seed):
    """Algorithm 1 reduces to FastSign on injective signatures (paper, Algorithm 1 note)."""
    U = random_orthogonal_eigvecs(n, seed)
    np.testing.assert_array_equal(gf2.fastsign(U), gf2.sign_vector(U, fast_path=False))
    np.testing.assert_array_equal(gf2.sign_vector(U, fast_path=True),
                                  gf2.sign_vector(U, fast_path=False))


def test_fastsign_rejects_non_injective_signature():
    U = np.array([[1.0, 2.0], [-1.0, 2.0], [0.5, 1.0]])  # rows 0 and 1 share |.|
    assert not gf2.is_injective_signature(U)
    with pytest.raises(ValueError):
        gf2.fastsign(U)
    with pytest.raises(ValueError):
        PRiSM().fastsign(U)


def test_first_pivot_rule_is_fastsign_on_injective_signature():
    """With an injective signature the hybrid's m=1 rule is a FastSign scan.

    Reference: order vertices by ascending lexicographic ``|V|``, flip each
    column so that its first significant entry in that order is positive.
    """
    canon = PRiSM(operator="comb_lap", drop_trivial=False, mult_mode="hybrid",
                  refine_ordering=False)
    A = generic_graph([canon])
    lam, V = canon.eigendecompose(A)
    order = np.lexsort(np.round(np.abs(V), 10).T[::-1])  # ascending lexicographic
    expected = V.copy()
    for j in range(V.shape[1]):
        v = next(v for v in order if abs(expected[v, j]) > 1e-8)
        if expected[v, j] < 0:
            expected[:, j] *= -1
    np.testing.assert_allclose(canon.canonicalize(V, lam), expected, rtol=0, atol=1e-12)

    info = canon.canonicalize(V, lam, return_info=True)
    np.testing.assert_array_equal(info.order, order)


def test_gf2_sign_rule_uses_fastsign_fast_path():
    """On an injective signature ``sign_rule='gf2'`` flips exactly the FastSign columns."""
    canon = PRiSM.from_preset("prism_gf2")
    A = generic_graph([canon])
    lam, V = canon.eigendecompose(A)
    s = gf2.fastsign(V)
    expected = V * np.where(s == 1, -1.0, 1.0)[None, :]
    np.testing.assert_allclose(canon.canonicalize(V, lam), expected, rtol=0, atol=1e-12)


def test_canonical_form_agrees_across_vertex_relabelings():
    canon = PRiSM.from_preset("prism_gf2")
    A = generic_graph([canon])
    lam, V = canon.eigendecompose(A)
    rows = lambda M: M[np.lexsort(np.round(M, 8).T[::-1])]
    base = rows(canon.canonicalize(V, lam))
    p = np.random.default_rng(7).permutation(A.shape[0])
    lam_p, V_p = canon.eigendecompose(permute(A, p))
    np.testing.assert_allclose(rows(canon.canonicalize(V_p, lam_p)), base, rtol=0, atol=1e-8)
