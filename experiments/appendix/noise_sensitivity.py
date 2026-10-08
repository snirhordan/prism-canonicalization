#!/usr/bin/env python3
"""Permutation invariance under symmetric matrix noise (table ``tab:noise-sensitivity``).

For each of the first ``--n-graphs`` ZINC validation graphs and each noise level
sigma, draw one symmetric Gaussian perturbation of the adjacency matrix,
A_sigma = A + sigma (Xi + Xi^T) / 2 with zero diagonal, Xi_ij ~ N(0, 1).  For
each of ``--n-perms`` random permutations P, compute the eigendecomposition of
the combinatorial Laplacian D - A of A_sigma and of P A_sigma P^T, canonicalize
both and compare the canonical eigenvector matrices as row multisets (rows
rounded to 6 decimals, ``numpy.allclose`` with atol 1e-6).  Both arms see the
same noisy matrix up to relabelling, so a failure is a permutation-invariance
failure at that noise level.  The experiment is repeated for several seeds; the
table reports the mean and standard deviation (ddof 1) of the failure counts
across seeds.

Methods
    prism   ``PRiSM.from_preset("rebuttal_comb_lap")``
    map     ``MAP.from_profile("brec")``
    oap     ``OAP.from_profile("brec")``

    python noise_sensitivity.py --data-dir ./data --out results/noise_sensitivity.json
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

PRESET = "rebuttal_comb_lap"
METHODS = {"prism": "PRiSM", "map": "MAP", "oap": "OAP"}
SIGMAS = [0.0, 1e-12, 1e-8, 1e-6, 1e-4, 1e-2]
SEEDS = [42, 43, 44]
ATOL = 1e-6
DECIMALS = 6


def sigma_key(sigma: float) -> str:
    return f"sigma_{sigma:.0e}"


def make_canonicalizers(names):
    """Name -> function ``(V, lam) -> canonical V``, rows then sorted lexicographically."""
    raw = {}
    for name in names:
        if name == "prism":
            raw[name] = PRiSM.from_preset(PRESET).canonicalize
        elif name == "map":
            from prism.baselines import MAP
            raw[name] = MAP.from_profile("brec").canonicalize
        elif name == "oap":
            from prism.baselines import OAP
            raw[name] = OAP.from_profile("brec").canonicalize
        else:
            raise ValueError(f"unknown method {name!r}")

    def wrap(fn):
        def canon(V, lam):
            Vc = fn(V.copy(), lam.copy())
            return Vc[np.lexsort(Vc.T[::-1])]
        return canon

    return {name: wrap(fn) for name, fn in raw.items()}


def symmetric_noise(n: int, rng: np.random.Generator, sigma: float) -> np.ndarray:
    if sigma == 0.0:
        return np.zeros((n, n), dtype=np.float64)
    G = rng.standard_normal((n, n)) * sigma
    return 0.5 * (G + G.T)


def run_seed(adjacencies, methods, sigmas, n_perms: int, seed: int, log=None):
    """One seed: trial counts, failure counts and canonicalization errors per sigma."""
    spectrum = PRiSM.from_preset(PRESET)  # eigh(D - A), nothing dropped
    canon = make_canonicalizers(methods)
    keys = [sigma_key(s) for s in sigmas]
    trials = {k: 0 for k in keys}
    fails = {m: {k: 0 for k in keys} for m in methods}
    errors = 0
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for i, A in enumerate(adjacencies):
        n = A.shape[0]
        if n < 2:
            continue
        for sigma, key in zip(sigmas, keys):
            A_noisy = A + symmetric_noise(n, rng, sigma)
            np.fill_diagonal(A_noisy, 0.0)
            lam, V = spectrum.eigendecompose(A_noisy)
            try:
                ref = {m: canon[m](V, lam) for m in methods}
            except Exception as exc:
                errors += 1
                print(f"  seed {seed} graph {i} sigma {sigma}: canonicalization failed: {exc}",
                      flush=True)
                continue
            for p in range(n_perms):
                perm = rng.permutation(n)
                lam_p, V_p = spectrum.eigendecompose(A_noisy[np.ix_(perm, perm)])
                try:
                    got = {m: canon[m](V_p, lam_p) for m in methods}
                except Exception as exc:
                    errors += 1
                    print(f"  seed {seed} graph {i} sigma {sigma} perm {p}: "
                          f"canonicalization failed: {exc}", flush=True)
                    continue
                trials[key] += 1
                for m in methods:
                    if not common.row_multiset_equal(ref[m], got[m], DECIMALS, ATOL):
                        fails[m][key] += 1
        if log and (i + 1) % 50 == 0:
            log(f"  seed {seed} [{i + 1}/{len(adjacencies)}] {time.time() - t0:.0f}s")
    return {"seed": seed, "trials": trials, "failures": fails, "errors": errors}


def aggregate(per_seed, methods, sigmas):
    """Mean and standard deviation (ddof 1) of the per-seed failure counts."""
    agg = {m: {} for m in methods}
    for m in methods:
        for sigma in sigmas:
            k = sigma_key(sigma)
            counts = [s["failures"][m][k] for s in per_seed]
            trial_counts = [s["trials"][k] for s in per_seed]
            rates = [c / t if t > 0 else 0.0 for c, t in zip(counts, trial_counts)]
            agg[m][k] = {
                "fail_count_per_seed": counts,
                "trials_per_seed": trial_counts,
                "fail_rate_per_seed": rates,
                "fail_rate_mean": float(np.mean(rates)),
                "fail_rate_std": float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0,
                "fail_count_mean": float(np.mean(counts)),
                "fail_count_std": float(np.std(counts, ddof=1)) if len(counts) > 1 else 0.0,
            }
    return agg


def print_table(agg, methods, sigmas) -> None:
    print(f"\n{'sigma':>10} | " + " | ".join(f"{METHODS[m] + ' (mean+-std)':>22}" for m in methods))
    for sigma in sigmas:
        k = sigma_key(sigma)
        cells = [f"{agg[m][k]['fail_count_mean']:>8.1f} +- {agg[m][k]['fail_count_std']:>6.1f}"
                 for m in methods]
        print(f"{sigma:>10.0e} | " + " | ".join(f"{c:>22}" for c in cells))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-dir", default="./data", help="dataset root (default ./data)")
    p.add_argument("--out", default="results/noise_sensitivity.json", help="output .json")
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    p.add_argument("--n-graphs", type=int, default=200, help="ZINC validation graphs")
    p.add_argument("--n-perms", type=int, default=5, help="random permutations per graph and sigma")
    p.add_argument("--sigmas", nargs="+", type=float, default=SIGMAS)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = p.parse_args(argv)

    ds = common.load_zinc(args.data_dir, "val")
    n_total = min(args.n_graphs, len(ds))
    adjacencies = list(common.pyg_adjacencies(ds, range(n_total)))
    print(f"ZINC validation subset: {n_total} of {len(ds)} graphs; seeds {args.seeds}", flush=True)

    per_seed = []
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        per_seed.append(run_seed(adjacencies, args.methods, args.sigmas, args.n_perms, seed,
                                 log=lambda s: print(s, flush=True)))
    agg = aggregate(per_seed, args.methods, args.sigmas)
    payload = {
        "config": {"preset": PRESET, "n_graphs": n_total, "n_perms": args.n_perms,
                   "sigmas": args.sigmas, "seeds": args.seeds, "atol": ATOL,
                   "decimals": DECIMALS, "methods": args.methods,
                   "metric": "row-multiset equality"},
        "per_seed": per_seed,
        "aggregate": agg,
    }
    common.write_json(args.out, payload)
    print_table(agg, args.methods, args.sigmas)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
