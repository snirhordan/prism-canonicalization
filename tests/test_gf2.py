"""Algorithm 1 (Partition, Refine, Solve, Match) and the worked example of the paper."""

import numpy as np
import pytest

from prism import PRiSM, gf2

# Section 4 of the paper: the eigenvector matrix of the running example.
U_EXAMPLE = np.array([
    [+1, +1, +2, +1],
    [-1, 0, -1, -2],
    [-1, -1, -2, -1],
    [-1, 0, +1, +2],
], dtype=np.float64)

# Partition output: rows regrouped into the classes C1 = {1, 2}, C2 = {3, 4}.
PARTITION_ROWS = [0, 2, 1, 3]
# Solve output as printed in the paper (same row grouping).
SOLVE_PRINTED = np.array([
    [-1, -1, -2, -1],
    [+1, +1, +2, +1],
    [+1, 0, +1, +2],
    [+1, 0, -1, -2],
], dtype=np.float64)
# Match output as printed in the paper.  Rows 1 and 3 are consistent with a
# descending lexicographic sort of the Solve output; in rows 2 and 4 the last
# two entries appear transposed in the printed matrix (typo in the paper), so
# those rows are compared with the transposition undone.
MATCH_PRINTED = np.array([
    [+1, +1, +2, +1],
    [+1, 0, +2, +1],
    [+1, 0, -1, -2],
    [-1, -1, -1, -2],
], dtype=np.float64)


def test_partition_matches_worked_example():
    p0 = gf2.partition_stage(U_EXAMPLE)
    assert p0[0] == p0[2] and p0[1] == p0[3] and p0[0] != p0[1]
    # the class with signature (1, 1, 2, 1) comes first, as printed
    assert p0[0] == 0 and p0[1] == 1


def test_refine_leaves_example_partition_unchanged():
    p0 = gf2.partition_stage(U_EXAMPLE)
    np.testing.assert_array_equal(gf2.refine_stage(U_EXAMPLE, p0), p0)


def test_solve_flips_all_four_columns():
    """Paper: the requirement on C1 and C2 forces s1 = s2 = s3 = s4 = -1."""
    p0 = gf2.partition_stage(U_EXAMPLE)
    classes = gf2.refine_stage(U_EXAMPLE, p0)
    s, kernel = gf2.solve_stage(U_EXAMPLE, classes)
    np.testing.assert_array_equal(s, [1, 1, 1, 1])
    assert len(kernel) == 0  # the solution is unique
    signed = U_EXAMPLE * np.where(s == 1, -1.0, 1.0)[None, :]
    np.testing.assert_array_equal(signed[PARTITION_ROWS], SOLVE_PRINTED)


def test_match_matches_worked_example():
    res = PRiSM().canonicalize_simple(U_EXAMPLE)
    canonical = res.canonical
    # rows 1 and 3 of the printed Match matrix
    np.testing.assert_array_equal(canonical[0], MATCH_PRINTED[0])
    np.testing.assert_array_equal(canonical[2], MATCH_PRINTED[2])
    # rows 2 and 4 of the printed matrix, with the transposed last two entries undone
    swap = lambda r: r[[0, 1, 3, 2]]
    np.testing.assert_array_equal(canonical[1], swap(MATCH_PRINTED[1]))
    np.testing.assert_array_equal(canonical[3], swap(MATCH_PRINTED[3]))
    # and the rows are sorted in descending lexicographic order
    keys = [tuple(r) for r in canonical]
    assert keys == sorted(keys, reverse=True)


def test_example_is_invariant_to_signs_and_row_permutations():
    base = gf2.prism_canonicalize(U_EXAMPLE).canonical
    rng = np.random.default_rng(0)
    for _ in range(10):
        t = rng.choice([-1.0, 1.0], size=4)
        p = rng.permutation(4)
        np.testing.assert_array_equal(gf2.prism_canonicalize((U_EXAMPLE * t)[p]).canonical, base)


def _path_laplacian_eigvecs(n):
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = A[i + 1, i] = 1.0
    return np.linalg.eigh(np.diag(A.sum(1)) - A)[1]


@pytest.mark.parametrize("U", [_path_laplacian_eigvecs(5), _path_laplacian_eigvecs(8),
                               np.linalg.eigh(np.random.default_rng(5).standard_normal((9, 9))
                                              + np.random.default_rng(5).standard_normal((9, 9)).T)[1]],
                         ids=["path5", "path8", "random9"])
def test_algorithm1_is_a_canonicalization(U):
    """Same canonical matrix for every sign pattern and vertex relabeling.

    The path graphs have a nontrivial automorphism, so only the matrix (not
    the labeling of its rows) is canonical.
    """
    base = gf2.prism_canonicalize(U).canonical
    rng = np.random.default_rng(1)
    for _ in range(8):
        t = rng.choice([-1.0, 1.0], size=U.shape[1])
        p = rng.permutation(U.shape[0])
        got = gf2.prism_canonicalize((U * t)[p]).canonical
        np.testing.assert_allclose(got, base, rtol=0, atol=1e-8)


def brute_force_solve(U, decimals=gf2.DECIMALS):
    """Reference Solve by candidate-set filtering (independent of Gaussian elimination).

    Maintain the set of all 2^k sign vectors.  Visit the constraint rows in the
    canonical class order; a row whose value is constant on the current set
    carries no new information and is dropped, otherwise the set is filtered
    to the vectors satisfying it.  Return the lex-smallest survivor.
    """
    n, k = U.shape
    p0 = gf2.partition_stage(U, decimals)
    classes = gf2.refine_stage(U, p0, decimals)
    by_class = {}
    for v in range(n):
        by_class.setdefault(int(classes[v]), []).append(v)
    ordered = sorted(by_class.values(), key=lambda vs: gf2._class_sort_key(U, vs, decimals),
                     reverse=True)
    bits = {t: np.array([(t >> j) & 1 for j in range(k)], dtype=np.uint8) for t in range(2 ** k)}
    candidates = list(range(2 ** k))
    dot = lambda row, t: int(np.dot(row.astype(np.int64), bits[t].astype(np.int64)) % 2)
    for verts in ordered:
        K = np.nonzero(np.round(np.abs(U[verts[0]]), decimals) > 0)[0]
        if K.size == 0:
            continue
        A = [(U[v, K] < 0).astype(np.uint8) for v in verts]
        a0 = A[0]
        for row_local in gf2.gf2_null_space([a ^ a0 for a in A], ncols=K.size):
            b = int(np.dot(row_local.astype(np.int64), a0.astype(np.int64)) % 2)
            row = np.zeros(k, dtype=np.uint8)
            row[K] = row_local
            if len({dot(row, t) for t in candidates}) <= 1:
                continue
            candidates = [t for t in candidates if dot(row, t) == b]
    return bits[min(candidates, key=lambda t: tuple(bits[t].tolist()))]


@pytest.mark.parametrize("n,seed", [(4, 0), (5, 1), (6, 2), (7, 3), (8, 4)])
def test_solve_agrees_with_enumeration(n, seed):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    U = np.linalg.eigh((M + M.T) / 2)[1]
    p0 = gf2.partition_stage(U)
    s, _ = gf2.solve_stage(U, gf2.refine_stage(U, p0))
    np.testing.assert_array_equal(s, brute_force_solve(U))


def test_solve_agrees_with_enumeration_on_example_and_path():
    for U in (U_EXAMPLE, _path_laplacian_eigvecs(5)):
        p0 = gf2.partition_stage(U)
        s, _ = gf2.solve_stage(U, gf2.refine_stage(U, p0))
        np.testing.assert_array_equal(s, brute_force_solve(U))


def test_hybrid_gf2_matches_algorithm1_on_simple_spectrum():
    """On a simple-spectrum graph ``sign_rule='gf2'`` applies the sign vector of Algorithm 1."""
    from helpers import generic_graph

    canon = PRiSM.from_preset("prism_gf2")
    A = generic_graph([canon])
    lam, V = canon.eigendecompose(A)
    s = gf2.sign_vector(V, fast_path=False)
    expected = V * np.where(s == 1, -1.0, 1.0)[None, :]
    np.testing.assert_allclose(canon.canonicalize(V, lam), expected, rtol=0, atol=1e-12)
