#!/usr/bin/env python3
"""Empirical permutation-equivariance test (table ``tab:invariance``).

For each graph, take the full eigendecomposition of the normalized adjacency
D^-1/2 A D^-1/2 without its trivial (largest) eigenpair, so that k = n - 1,
canonicalize it, relabel the vertices by random permutations, recompute the
eigendecomposition of the relabelled graph, canonicalize again and compare.  A
trial fails when the canonical eigenvector matrix of the relabelled graph is not
a row permutation of the original one (rows rounded to 10 decimals, then
``numpy.allclose`` with atol 1e-6 after lexicographic sorting).

Methods
    prism   ``PRiSM.from_preset("qr_refined_full")``
    map     ``MAP.from_profile("gnn")``
    oap     ``OAP.from_profile("gnn")``

Graphs
    zinc      the first ``--n-graphs`` molecules of the full ZINC training file
              (~220k molecules), molecules with at most 5 atoms dropped
    moltox21  the first ``--n-graphs`` graphs of the ogbg-moltox21 training
              split with more than 5 nodes

Permutations of graph number g (in the filtered list) are drawn from
``numpy.random.default_rng(3000 + g)``.

    python invariance_test.py --data-dir ./data --out results/invariance.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

from prism import PRiSM  # noqa: E402

ATOL = 1e-6
DECIMALS = 10
PRESET = "qr_refined_full"
DATASETS = {"zinc": "ZINC", "moltox21": "MolTox21"}
METHODS = {"prism": "PRiSM", "map": "MAP", "oap": "OAP"}


def make_canonicalizers(names):
    """Name -> function ``(V, lam) -> canonical V`` for the requested methods."""
    out = {}
    for name in names:
        if name == "prism":
            out[name] = PRiSM.from_preset(PRESET).canonicalize
        elif name == "map":
            from prism.baselines import MAP
            out[name] = MAP.from_profile("gnn").canonicalize
        elif name == "oap":
            from prism.baselines import OAP
            out[name] = OAP.from_profile("gnn").canonicalize
        else:
            raise ValueError(f"unknown method {name!r}")
    return out


def run_dataset(adjacencies, methods, n_perms: int, rng_offset: int = 3000, log=None):
    """Failures of each method over all graphs and permutations.

    Returns ``{method: {"failures": int, "trials": int, "failing_graphs": [...]}}``.
    """
    spectrum = PRiSM.from_preset(PRESET)  # eigendecomposition with the trivial pair dropped
    canon = make_canonicalizers(methods)
    result = {m: {"failures": 0, "trials": 0, "failing_graphs": []} for m in methods}
    t0 = time.time()
    for g, A in enumerate(adjacencies):
        n = A.shape[0]
        lam, V = spectrum.eigendecompose(A)
        reference = {m: canon[m](V.copy(), lam) for m in methods}
        rng = np.random.default_rng(rng_offset + g)
        failed = set()
        for _ in range(n_perms):
            perm = rng.permutation(n)
            lam_p, V_p = spectrum.eigendecompose(A[np.ix_(perm, perm)])
            for m in methods:
                result[m]["trials"] += 1
                if not common.row_multiset_equal(reference[m], canon[m](V_p.copy(), lam_p),
                                                 DECIMALS, ATOL):
                    result[m]["failures"] += 1
                    failed.add((m, g))
        for m, graph in sorted(failed):
            result[m]["failing_graphs"].append(graph)
        if log and (g + 1) % 50 == 0:
            log(f"    {g + 1}/{len(adjacencies)} graphs, {time.time() - t0:.0f}s, "
                + ", ".join(f"{m} {result[m]['failures']}" for m in methods))
    return result


def load_adjacencies(dataset: str, data_dir: str, n_graphs: int):
    if dataset == "zinc":
        return common.zinc_full_train_adjacencies(data_dir, n_graphs)
    ds = common.load_ogb("ogbg-moltox21", data_dir)
    out = []
    for i in common.ogb_split(ds, "train"):
        if len(out) >= n_graphs:
            break
        data = ds[i]
        if data.num_nodes > 5:
            out.append(common.dense_adjacency(data.edge_index.numpy(), int(data.num_nodes)))
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-dir", default="./data", help="dataset root (default ./data)")
    p.add_argument("--out", default="results/invariance.json", help="output .json")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    p.add_argument("--n-graphs", type=int, default=200, help="graphs per dataset")
    p.add_argument("--n-perms", type=int, default=5, help="random permutations per graph")
    args = p.parse_args(argv)

    results = {}
    for dataset in args.datasets:
        adjacencies = load_adjacencies(dataset, args.data_dir, args.n_graphs)
        print(f"{DATASETS[dataset]}: {len(adjacencies)} graphs x {args.n_perms} permutations",
              flush=True)
        results[dataset] = run_dataset(adjacencies, args.methods, args.n_perms,
                                       log=lambda s: print(s, flush=True))
    payload = {
        "config": {"preset": PRESET, "n_graphs": args.n_graphs, "n_perms": args.n_perms,
                   "atol": ATOL, "decimals": DECIMALS, "methods": args.methods},
        "results": results,
    }
    common.write_json(args.out, payload)

    print(f"\n{'dataset':<10}{'method':<8}{'failures':>10}{'trials':>8}")
    for dataset in args.datasets:
        for m in args.methods:
            r = results[dataset][m]
            print(f"{DATASETS[dataset]:<10}{METHODS[m]:<8}{r['failures']:>10}{r['trials']:>8}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
