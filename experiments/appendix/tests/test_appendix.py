"""Tests of the appendix scripts on small synthetic graphs (no dataset download).

Run from the repository root::

    python -m pytest experiments/appendix/tests
"""

import os
import sys

import networkx as nx
import numpy as np
import pytest

APPENDIX = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, APPENDIX)

import common  # noqa: E402
import invariance_test  # noqa: E402
import multiplicity_stats  # noqa: E402
import noise_sensitivity  # noqa: E402
import rbound_stats  # noqa: E402
import runtime_scaling  # noqa: E402

torch = pytest.importorskip("torch")


def _adjacency(G):
    return nx.to_numpy_array(G, nodelist=sorted(G.nodes), dtype=np.float64)


def _random_graphs(count, n, p, seed):
    """Connected Erdos-Renyi graphs (trivial automorphism group with high probability)."""
    out, s = [], seed
    while len(out) < count:
        G = nx.gnp_random_graph(n, p, seed=s)
        s += 1
        if nx.is_connected(G):
            out.append(_adjacency(G))
    return out


# ------------------------------------------------------------------- common

def test_dense_adjacency_symmetrizes():
    A = common.dense_adjacency(np.array([[0, 1], [1, 2]]), 3)
    assert A.tolist() == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]


def test_percent_rounds_half_up():
    assert common.percent(0.2115) == "21.2"
    assert common.percent(0.228) == "22.8"
    assert common.percent(1.0, 0) == "100"
    assert common.half_up(11.95) == "12.0" and common.half_up(23.45) == "23.5"
    assert common.half_up(0.2, 2) == "0.20"


def test_row_multiset_equal_ignores_row_order_only():
    M = np.array([[0.0, 1.0], [1.0, 0.5], [0.3, 0.3]])
    assert common.row_multiset_equal(M, M[[2, 0, 1]], 10, 1e-6)
    assert not common.row_multiset_equal(M, M * np.array([-1.0, 1.0]), 10, 1e-6)
    assert not common.row_multiset_equal(M, M[:2], 10, 1e-6)


def test_write_rows_csv_and_json(tmp_path):
    rows = [{"a": 1, "b": 2.5}]
    common.write_rows(str(tmp_path / "x" / "t.csv"), rows, ["a", "b"])
    common.write_rows(str(tmp_path / "x" / "t.json"), rows, ["a", "b"])
    assert (tmp_path / "x" / "t.csv").read_text().split() == ["a,b", "1,2.5"]
    assert '"b": 2.5' in (tmp_path / "x" / "t.json").read_text()


# ------------------------------------------------------------ r-bound table

@pytest.mark.parametrize("n", [5, 6, 9])
def test_rbound_path_graph(n):
    """A path has a simple spectrum and the reflection pairs mirrored vertices:
    ceil(n / 2) classes, so r = n // 2."""
    n_, L, nonsingleton = rbound_stats.partition_stats(_adjacency(nx.path_graph(n)), 6)
    assert n_ == n
    assert L == (n + 1) // 2
    assert nonsingleton == n // 2


def test_rbound_row_statistics():
    graphs = [_adjacency(nx.path_graph(n)) for n in (6, 5, 2, 1)]  # r = 3, 2, 1, 0
    row = rbound_stats.dataset_row("toy", graphs, 6)
    assert row["val_graphs"] == 4
    assert row["max_r"] == 3 and row["median_r"] == 1.5
    assert row["frac_r0"] == 0.25          # the single vertex
    assert row["frac_one_block"] == 0.25   # K2: one block of two mirrored vertices
    assert row["frac_r_le_4"] == 1.0
    assert row["mean_nodes"] == 3.5
    assert set(row) == set(rbound_stats.FIELDS)


# ----------------------------------------------------- multiplicity table

def test_count_multiplicities():
    # eigenvalues 1 (x2), 0.5 (x1), 0 (x3), -1 (x4), -2 (x5)
    lam = np.array([-2] * 5 + [-1] * 4 + [0] * 3 + [0.5] + [1] * 2, dtype=float)
    assert multiplicity_stats.count_multiplicities(lam) == (1, 1, 1, 2)
    assert multiplicity_stats.count_multiplicities(np.array([0.0, 1e-7])) == (0, 1, 0, 0)


@pytest.mark.parametrize("G, expected", [
    (nx.complete_graph(4), (1, 0, 1, 0)),     # spectrum 1, -1/3 (x3)
    (nx.cycle_graph(6), (2, 2, 0, 0)),        # cos(2 pi j / 6): 1, 1/2 (x2), -1/2 (x2), -1
    (nx.path_graph(5), (5, 0, 0, 0)),         # simple
])
def test_multiplicities_of_known_graphs(G, expected):
    from prism import operators
    lam = np.linalg.eigvalsh(operators.normalized_adjacency(_adjacency(G)))
    assert multiplicity_stats.count_multiplicities(lam) == expected


def test_multiplicity_dataset_row_samples_reproducibly():
    graphs = _random_graphs(12, 9, 0.4, seed=0)
    a = multiplicity_stats.dataset_row("toy", graphs, 5, 42, 1e-6)
    b = multiplicity_stats.dataset_row("toy", graphs, 5, 42, 1e-6)
    assert a == b and a["sampled_graphs"] == 5
    assert a["mean_m1"] + 2 * a["mean_m2"] + 3 * a["mean_m3"] <= a["mean_nodes"] + 1e-9


# --------------------------------------------------------- invariance table

def test_invariance_prism_is_equivariant_on_simple_spectrum_graphs():
    graphs = _random_graphs(4, 11, 0.4, seed=3)
    res = invariance_test.run_dataset(graphs, ["prism"], n_perms=3)
    assert res["prism"]["trials"] == 12
    assert res["prism"]["failures"] == 0


def test_invariance_reports_all_requested_methods():
    graphs = _random_graphs(2, 8, 0.5, seed=7)
    res = invariance_test.run_dataset(graphs, ["prism", "map", "oap"], n_perms=2)
    assert set(res) == {"prism", "map", "oap"}
    assert all(r["trials"] == 4 for r in res.values())
    assert res["prism"]["failures"] == 0


def test_invariance_detects_a_broken_canonicalizer(monkeypatch):
    """The identity map is not permutation equivariant: every trial must fail."""
    monkeypatch.setattr(invariance_test, "make_canonicalizers",
                        lambda names: {m: (lambda V, lam: V) for m in names})
    graphs = _random_graphs(2, 8, 0.5, seed=5)
    res = invariance_test.run_dataset(graphs, ["prism"], n_perms=3)
    assert res["prism"]["failures"] >= 5  # a permutation can leave the eigenvectors' signs alone


# -------------------------------------------------------------- noise table

def test_noise_run_seed_counts_and_determinism():
    graphs = _random_graphs(2, 9, 0.5, seed=11)
    kw = dict(methods=["prism", "map"], sigmas=[0.0, 1e-2], n_perms=2, seed=1)
    a = noise_sensitivity.run_seed(graphs, **kw)
    b = noise_sensitivity.run_seed(graphs, **kw)
    assert a == b
    assert a["trials"] == {"sigma_0e+00": 4, "sigma_1e-02": 4}
    assert a["failures"]["prism"] == {"sigma_0e+00": 0, "sigma_1e-02": 0}
    assert a["errors"] == 0


def test_noise_aggregate_uses_sample_standard_deviation():
    per_seed = [{"trials": {"sigma_0e+00": 10}, "failures": {"prism": {"sigma_0e+00": c}}}
                for c in (0, 2, 4)]
    agg = noise_sensitivity.aggregate(per_seed, ["prism"], [0.0])["prism"]["sigma_0e+00"]
    assert agg["fail_count_mean"] == 2.0 and agg["fail_count_std"] == pytest.approx(2.0)


def test_noise_symmetric_noise_is_symmetric_and_zero_for_sigma_zero():
    rng = np.random.default_rng(0)
    assert not noise_sensitivity.symmetric_noise(5, rng, 0.0).any()
    N = noise_sensitivity.symmetric_noise(5, rng, 1e-3)
    assert np.array_equal(N, N.T) and N.any()


# ------------------------------------------------------------ runtime table

def test_runtime_er_graphs_are_connected_and_reproducible():
    A = runtime_scaling.er_connected(24, np.random.default_rng(42))
    B = runtime_scaling.er_connected(24, np.random.default_rng(42))
    assert np.array_equal(A, B) and np.array_equal(A, A.T)
    assert nx.is_connected(nx.from_numpy_array(A))


def test_runtime_run_and_summary():
    times = runtime_scaling.run([8, 12], reps=2, seed=0, methods=["prism", "map", "oap"])
    summary = runtime_scaling.summarize(times, ["prism", "map", "oap"])
    assert set(summary) == {8, 12}
    for n in (8, 12):
        assert summary[n]["n_reps"] == 2
        assert all(summary[n][m]["median"] > 0 for m in ("prism", "map", "oap"))


def test_runtime_threads_are_pinned():
    assert os.environ["OMP_NUM_THREADS"] == "1" and os.environ["MKL_NUM_THREADS"] == "1"
