#!/usr/bin/env python3
"""Wall-clock canonicalization time on random graphs (table ``tab:runtime-scaling``).

For every size n, ``--reps`` connected Erdos-Renyi graphs G(n, p) with
p = max(log(n) / n, 0.1) are drawn (``networkx.fast_gnp_random_graph``, redrawn
until connected; the draws come from ``numpy.random.default_rng(--seed)``).  For
each graph the eigendecomposition of the combinatorial Laplacian D - A is
computed once (not timed), every method is called once untimed (warm-up) and
once timed.  The timed region is the canonicalization of the eigenvector matrix
plus the lexicographic sort of its rows.  The table lists the median in
milliseconds and the ratios MAP / PRiSM and OAP / PRiSM of the medians.

Methods
    prism   ``PRiSM.from_preset("rebuttal_comb_lap")``
    map     ``MAP.from_profile("brec")``
    oap     ``OAP.from_profile("brec")``

The measurement is for a single CPU core: the threads of OpenMP, MKL, OpenBLAS
and PyTorch are pinned to one at import time of this script.  For a stricter
setting also restrict the process to one core, e.g. ``taskset -c 0``.

    python runtime_scaling.py --out results/runtime_scaling.json
"""

from __future__ import annotations

import os

# One core: set before numpy / torch are imported.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

from prism import PRiSM  # noqa: E402

PRESET = "rebuttal_comb_lap"
METHODS = {"prism": "PRiSM", "map": "MAP", "oap": "OAP"}
SIZES = [16, 32, 64, 128, 256, 512]


def er_connected(n: int, rng: np.random.Generator, max_tries: int = 100) -> np.ndarray:
    """Connected Erdos-Renyi graph G(n, max(log(n)/n, 0.1)) as a dense adjacency matrix."""
    p = max(np.log(max(n, 2)) / max(n, 2), 0.1)
    for _ in range(max_tries):
        seed = int(rng.integers(0, 2 ** 31 - 1))
        G = nx.fast_gnp_random_graph(n, p, seed=seed)
        if nx.is_connected(G):
            return nx.to_numpy_array(G, dtype=np.float64)
    return nx.to_numpy_array(nx.path_graph(n), dtype=np.float64)  # fallback, essentially never used


def make_canonicalizers(names):
    out = {}
    for name in names:
        if name == "prism":
            out[name] = PRiSM.from_preset(PRESET).canonicalize
        elif name == "map":
            from prism.baselines import MAP
            out[name] = MAP.from_profile("brec").canonicalize
        elif name == "oap":
            from prism.baselines import OAP
            out[name] = OAP.from_profile("brec").canonicalize
        else:
            raise ValueError(f"unknown method {name!r}")
    return out


def time_call(fn, V, lam) -> float:
    """Seconds for ``fn(V, lam)`` followed by the lexicographic row sort."""
    t0 = time.perf_counter()
    Vc = fn(V, lam)
    np.lexsort(Vc.T[::-1])
    return time.perf_counter() - t0


def run(sizes, reps: int, seed: int, methods, log=None):
    """Timings in seconds: ``{n: {method: [per graph]}}``."""
    spectrum = PRiSM.from_preset(PRESET)
    canon = make_canonicalizers(methods)
    rng = np.random.default_rng(seed)
    times = {n: {m: [] for m in methods} for n in sizes}
    for n in sizes:
        if log:
            log(f"\n=== n = {n} ({reps} graphs) ===")
        for r in range(reps):
            lam, V = spectrum.eigendecompose(er_connected(n, rng))
            try:
                for m in methods:  # warm-up
                    canon[m](V, lam)
            except Exception as exc:
                if log:
                    log(f"  rep {r}: warm-up failed: {exc}")
                continue
            for m in methods:
                times[n][m].append(time_call(canon[m], V, lam))
        if log:
            log("  median (s): " + "  ".join(f"{m}={np.median(times[n][m]):.4f}" for m in methods))
    return times


def summarize(times, methods):
    summary = {}
    for n, per_method in times.items():
        summary[n] = {"n_reps": len(per_method[methods[0]])}
        for m in methods:
            t = np.array(per_method[m])
            summary[n][m] = {"median": float(np.median(t)), "mean": float(np.mean(t)),
                             "p90": float(np.percentile(t, 90))}
    return summary


def machine_info() -> dict:
    info = {"python": platform.python_version(), "numpy": np.__version__,
            "platform": platform.platform(),
            "threads_env": {v: os.environ.get(v) for v in
                            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
    try:
        info["cpus_available"] = len(os.sched_getaffinity(0))
    except AttributeError:
        pass
    try:
        with open("/proc/cpuinfo") as f:
            info["cpu"] = next(l.split(":", 1)[1].strip() for l in f if l.startswith("model name"))
    except (OSError, StopIteration):
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
    except ImportError:
        pass
    return info


def print_table(summary, methods) -> None:
    head = f"{'n':>5} | " + " | ".join(f"{METHODS[m] + ' (ms)':>11}" for m in methods)
    if {"prism", "map", "oap"} <= set(methods):
        head += f" | {'vs MAP':>7} | {'vs OAP':>7}"
    print("\n" + head)
    for n, s in summary.items():
        line = f"{n:>5} | " + " | ".join(f"{s[m]['median'] * 1000:>11.2f}" for m in methods)
        if {"prism", "map", "oap"} <= set(methods):
            line += (f" | {s['map']['median'] / s['prism']['median']:>6.1f}x"
                     f" | {s['oap']['median'] / s['prism']['median']:>6.1f}x")
        print(line)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", default="results/runtime_scaling.json", help="output .json")
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    p.add_argument("--sizes", nargs="+", type=int, default=SIZES)
    p.add_argument("--reps", type=int, default=50, help="graphs per size")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except ImportError:
        pass  # PRiSM alone does not need torch
    except RuntimeError:
        pass  # inter-op pool already started

    times = run(args.sizes, args.reps, args.seed, args.methods, log=lambda s: print(s, flush=True))
    summary = summarize(times, args.methods)
    payload = {
        "config": {"preset": PRESET, "sizes": args.sizes, "reps": args.reps, "seed": args.seed,
                   "methods": args.methods,
                   "graph_model": "connected Erdos-Renyi G(n, max(log(n)/n, 0.1))"},
        "machine": machine_info(),
        "summary": summary,
    }
    common.write_json(args.out, payload)
    print_table(summary, args.methods)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
