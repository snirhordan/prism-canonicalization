#!/usr/bin/env python3
"""Generate ``tests/fixtures/golden.npz`` from the live research code.

The golden file stores a set of small graphs together with the outputs of the
original (pre-refactoring) implementations of every PRiSM variant, MAP, OAP,
LapPE and the GF(2) reference.  ``tests/test_golden.py`` checks that the
presets of :class:`prism.PRiSM` reproduce them.

Usage::

    LIVE_SRC=/path/containing/the/checkouts \\
        python dev/make_golden.py [--out tests/fixtures/golden.npz]

    # add only the IndRefineNauty keys to an existing fixture (graphs, spectra and
    # all other keys are left as they are; needs only numpy, scipy and pynauty)
    LIVE_SRC=... python dev/make_golden.py --only-indrefine

``LIVE_SRC`` must contain these checkouts (read-only; nothing is written there)::

    $LIVE_SRC/canonicalization-fresh      molecular GNN harness
    $LIVE_SRC/brec_pe_canon               transformer / BREC harness (spectral_m3.py,
                                          map_oap.py, indrefine.py)
    $LIVE_SRC/WL-Simple/experiments/thm1_demo/prism_signsolve.py   GF(2) reference

The environment must provide numpy, scipy, torch, dgl, networkx and pynauty (the
molecular GNN environment of the paper plus pynauty does).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import pickle
import subprocess
import sys

sys.dont_write_bytecode = True

import numpy as np

LIVE_SRC = os.environ.get("LIVE_SRC")
if not LIVE_SRC:
    sys.exit("set LIVE_SRC to the directory holding the live checkouts")
FRESH = os.path.join(LIVE_SRC, "canonicalization-fresh")
BREC = os.path.join(LIVE_SRC, "brec_pe_canon")
GF2_DIR = os.path.join(LIVE_SRC, "WL-Simple", "experiments", "thm1_demo")

MAX_NODES_ZINC = 26
N_ZINC = 10
PE_EXTRA_KS = (2, 20)


# ------------------------------------------------------------------ loading

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def git_rev(path):
    try:
        rev = subprocess.check_output(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(["git", "-C", path, "status", "--porcelain",
                                         "--untracked-files=no"],
                                        stderr=subprocess.DEVNULL).decode().strip()
        return rev + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


# ------------------------------------------------------------------- graphs

def _adj_from_edges(n, edges):
    A = np.zeros((n, n))
    for u, v in edges:
        A[u, v] = A[v, u] = 1.0
    return A


def synthetic_graphs():
    import networkx as nx

    def A_of(G):
        return nx.to_numpy_array(nx.convert_node_labels_to_integers(G), dtype=np.float64)

    graphs = {
        "edge": A_of(nx.path_graph(2)),
        "path3": A_of(nx.path_graph(3)),
        "path8": A_of(nx.path_graph(8)),
        "cycle4": A_of(nx.cycle_graph(4)),
        "cycle6": A_of(nx.cycle_graph(6)),
        "cycle8": A_of(nx.cycle_graph(8)),
        "cycle12": A_of(nx.cycle_graph(12)),
        "complete4": A_of(nx.complete_graph(4)),
        "complete6": A_of(nx.complete_graph(6)),
        "k33": A_of(nx.complete_bipartite_graph(3, 3)),
        "k24": A_of(nx.complete_bipartite_graph(2, 4)),
        "star7": A_of(nx.star_graph(6)),
        "petersen": A_of(nx.petersen_graph()),
        "cube": A_of(nx.hypercube_graph(3)),
        "hypercube4": A_of(nx.hypercube_graph(4)),
        "wheel7": A_of(nx.wheel_graph(7)),
        "grid3x4": A_of(nx.grid_2d_graph(3, 4)),
        "ladder6": A_of(nx.ladder_graph(6)),
        "prism_graph": A_of(nx.circular_ladder_graph(4)),
        "regular3_n12": A_of(nx.random_regular_graph(3, 12, seed=1)),
        "regular4_n10": A_of(nx.random_regular_graph(4, 10, seed=2)),
        "er_n12": A_of(nx.gnp_random_graph(12, 0.3, seed=3)),
        "er_n14": A_of(nx.gnp_random_graph(14, 0.35, seed=4)),
        "er_n10_dense": A_of(nx.gnp_random_graph(10, 0.6, seed=5)),
        "binary_tree": A_of(nx.balanced_tree(2, 3)),
        "two_triangles": A_of(nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(3))),
    }
    iso = _adj_from_edges(7, [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5)])  # isolated vertex 6
    graphs["with_isolated"] = iso
    return graphs


def zinc_graphs():
    path = os.path.join(FRESH, "data", "molecules")
    val = pickle.load(open(os.path.join(path, "val.pickle"), "rb"))
    idx = [int(i) for i in open(os.path.join(path, "val.index")).read().strip().split(",")]
    out = {}
    for i in idx:
        mol = val[i]
        n = int(mol["num_atom"])
        if n < 6 or n > MAX_NODES_ZINC:
            continue
        B = mol["bond_type"].numpy()
        A = (B != 0).astype(np.float64)
        A = np.maximum(A, A.T)
        np.fill_diagonal(A, 0.0)
        out[f"zinc_val{i}"] = A
        if len(out) == N_ZINC:
            break
    return out


def all_graphs():
    graphs = dict(zinc_graphs())
    graphs.update(synthetic_graphs())
    return graphs


def pack(arrays):
    """Concatenate flattened arrays; return (flat, offsets)."""
    flat = np.concatenate([np.asarray(a).ravel() for a in arrays])
    off = np.concatenate([[0], np.cumsum([np.asarray(a).size for a in arrays])]).astype(np.int64)
    return flat, off


def dgl_graph(A):
    import dgl
    import torch

    src, dst = np.nonzero(A)
    return dgl.graph((torch.from_numpy(src), torch.from_numpy(dst)), num_nodes=A.shape[0])


# ------------------------------------------------------------------- spectra

def spectra(A):
    """The three eigendecompositions the harnesses feed to the canonicalizers."""
    n = A.shape[0]
    # molecular GNN: D^-1/2 A D^-1/2 via scipy sparse, trivial eigenpair dropped
    from scipy import sparse as sp
    As = sp.csr_matrix(A)
    deg = np.asarray(A.sum(axis=0)).astype(np.int64)
    N = sp.diags(deg.clip(1) ** -0.5, dtype=np.double)
    lam, V = np.linalg.eigh((N * As * N).toarray())
    na = (lam[:-1], V[:, :-1])
    # transformer: I - D^-1/2 A D^-1/2 via scipy sparse, clipped, float32 storage
    d12 = 1.0 / np.sqrt(np.maximum(np.asarray(As.sum(axis=1)).flatten(), 1.0))
    D12 = sp.diags(d12)
    L = (sp.eye(n, format="csr") - D12 @ As @ D12).astype(np.float64)
    lam, V = np.linalg.eigh(L.toarray())
    nl = (np.clip(lam, 0.0, None).astype(np.float32), V.astype(np.float32))
    # rebuttal: D - A
    lam, V = np.linalg.eigh(np.diag(A.sum(1)) - A)
    cl = (lam, V)
    return {"norm_adj": na, "norm_lap_f32": nl, "comb_lap": cl}


# ---------------------------------------------------- IndRefineNauty (BREC row)

IR_NUM_VECS = 8      # BREC launch: num_vecs 8 ...
IR_BUFFER = 3        # ... and canon_buffer 3 -> the first 11 eigenpairs are canonicalized


def indrefine_goldens(out, mats, norm_lap_specs):
    """Add the ``fn__brec_indrefine*`` keys: the live ``indrefine_nauty_canonicalize`` applied to
    the stored float32 normalized-Laplacian eigenpairs, the way ``Transform`` of the live
    harness calls it (``edge_index`` of the graph, all valid columns), on the full spectrum and
    on the first ``num_vecs + buffer`` columns."""
    indrefine = load_module("live_brec_indrefine", os.path.join(BREC, "indrefine.py"))

    def live(A, lam, V):
        edge_index = np.stack(np.nonzero(A))
        Vc, order, rank = indrefine.indrefine_nauty_canonicalize(
            V, lam, edge_index=edge_index, num_nodes=A.shape[0])
        assert np.array_equal(rank[order], np.arange(A.shape[0]))
        return Vc, order

    full_V, full_order, win_V, win_order = [], [], [], []
    for A, (lam, V) in zip(mats, norm_lap_specs):
        lam = np.asarray(lam, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        Vc, order = live(A, lam, V)
        full_V.append(Vc)
        full_order.append(order)
        w = min(lam.shape[0], IR_NUM_VECS + IR_BUFFER)
        Vw, order_w = live(A, lam[:w], V[:, :w])
        win_V.append(Vw[:, :min(lam.shape[0], IR_NUM_VECS)])
        win_order.append(order_w)
    for key, Vs, orders in (("brec_indrefine", full_V, full_order),
                            ("brec_indrefine_window_k8", win_V, win_order)):
        out[f"fn__{key}__V"], out[f"fn__{key}__off"] = pack(Vs)
        out[f"fn__{key}__order"], out[f"fn__{key}__order_off"] = pack(orders)
    import pynauty

    try:
        blob = subprocess.check_output(
            ["git", "hash-object", os.path.join(BREC, "indrefine.py")],
            stderr=subprocess.DEVNULL).decode().strip()[:12]
    except Exception:
        blob = "unknown"
    try:
        last = subprocess.check_output(
            ["git", "-C", BREC, "log", "-1", "--format=%h", "--", "indrefine.py"],
            stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(
            ["git", "-C", BREC, "status", "--porcelain", "--", "indrefine.py"],
            stderr=subprocess.DEVNULL).decode().strip()
        last += "+modified" if dirty else ""
    except Exception:
        last = "unknown"
    out["meta__live_rev_indrefine"] = np.array([
        f"brec_pe_canon indrefine.py: last commit {last}, blob {blob}",
        f"pynauty {pynauty.__version__}, numpy {np.__version__}",
    ])


def stored_spectra(z, op):
    """The ``(lam, V)`` lists of the stored spectra of ``op`` in an existing fixture."""
    sizes = z["meta__sizes"]
    lam_flat, lam_off = z[f"spec__{op}__lam"], z[f"spec__{op}__lam_off"]
    V_flat, V_off = z[f"spec__{op}__V"], z[f"spec__{op}__V_off"]
    out = []
    for i, n in enumerate(sizes):
        n = int(n)
        c = n - 1 if op == "norm_adj" else n
        out.append((lam_flat[lam_off[i]:lam_off[i + 1]].reshape(c),
                    V_flat[V_off[i]:V_off[i + 1]].reshape(n, c)))
    return out


def stored_adjacencies(z):
    return [z["meta__adj"][z["meta__adj_off"][i]:z["meta__adj_off"][i + 1]]
            .reshape(int(n), int(n)).astype(np.float64) for i, n in enumerate(z["meta__sizes"])]


def add_indrefine_to_fixture(path):
    """Add only the IndRefineNauty keys to an existing fixture; every other key is kept as is."""
    z = dict(np.load(path))
    old_keys = set(z)
    indrefine_goldens(z, stored_adjacencies(z), stored_spectra(z, "norm_lap_f32"))
    np.savez_compressed(path, **z)
    new = sorted(set(z) - old_keys)
    print(f"added {len(new)} keys to {path}: {os.path.getsize(path) / 1e6:.2f} MB")
    for k in new:
        print("  ", k)


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "tests",
                                                  "fixtures", "golden.npz"))
    ap.add_argument("--only-indrefine", action="store_true",
                    help="add only the IndRefineNauty keys to the existing --out fixture")
    args = ap.parse_args()

    if args.only_indrefine:
        add_indrefine_to_fixture(os.path.abspath(args.out))
        return

    import torch

    torch.set_default_dtype(torch.float64)  # as in the molecular harness
    fresh = load_module("live_fresh_spectral", os.path.join(FRESH, "data", "spectral_m3.py"))
    brec = load_module("live_brec_spectral", os.path.join(BREC, "spectral_m3.py"))
    brec_mo = load_module("live_brec_map_oap", os.path.join(BREC, "map_oap.py"))
    gf2ref = load_module("live_gf2_ref", os.path.join(GF2_DIR, "prism_signsolve.py"))
    old_cwd = os.getcwd()
    os.chdir(FRESH)
    sys.path.insert(0, FRESH)
    mol = importlib.import_module("data.molecules")
    os.chdir(old_cwd)

    graphs = all_graphs()
    names = list(graphs)
    mats = [graphs[k] for k in names]
    sizes = np.array([A.shape[0] for A in mats], dtype=np.int64)
    out = {}
    out["meta__names"] = np.array(names)
    out["meta__sizes"] = sizes
    out["meta__adj"], out["meta__adj_off"] = pack([A.astype(np.uint8) for A in mats])
    out["meta__live_rev"] = np.array([
        f"canonicalization-fresh@{git_rev(FRESH)}",
        f"brec_pe_canon@{git_rev(BREC)}",
        f"WL-Simple@{git_rev(os.path.join(LIVE_SRC, 'WL-Simple'))}",
        f"numpy {np.__version__}, torch {torch.__version__}",
    ])

    specs = [spectra(A) for A in mats]
    for op in ("norm_adj", "norm_lap_f32", "comb_lap"):
        lam_flat, lam_off = pack([s[op][0] for s in specs])
        V_flat, V_off = pack([s[op][1] for s in specs])
        out[f"spec__{op}__lam"], out[f"spec__{op}__lam_off"] = lam_flat, lam_off
        out[f"spec__{op}__V"], out[f"spec__{op}__V_off"] = V_flat, V_off

    # ---- function-level goldens -------------------------------------------
    def run_all(op, fn_dict):
        for key, fn in fn_dict.items():
            Vs, extras = [], []
            for s in specs:
                lam, V = s[op]
                lam = np.asarray(lam, dtype=np.float64)
                V = np.asarray(V, dtype=np.float64)
                res = fn(V, lam)
                if isinstance(res, tuple):
                    Vs.append(res[0])
                    extras.append(np.asarray(res[1]))
                else:
                    Vs.append(res)
            out[f"fn__{key}__V"], out[f"fn__{key}__off"] = pack(Vs)
            if extras:
                out[f"fn__{key}__order"], out[f"fn__{key}__order_off"] = pack(extras)

    run_all("norm_adj", {
        "fresh_m3": lambda V, l: fresh.spectral_m3_canonicalize(V, l),
        "fresh_m3_refined": lambda V, l: fresh.spectral_m3_refined_canonicalize(V, l),
        "fresh_qr_refined": lambda V, l: fresh.spectral_qr_refined_canonicalize(V, l),
    })
    run_all("norm_lap_f32", {
        "brec_m3": lambda V, l: brec.spectral_m3_canonicalize(V, l),
        "brec_m3_refined": lambda V, l: brec.spectral_m3_canonicalize(
            V, l, refine_ordering=True, return_refined_ordering=True),
        "brec_qr_refined": lambda V, l: brec.spectral_qr_refined_canonicalize(V, l),
        "brec_map": lambda V, l: brec_mo.map_canonicalize(V, l),
        "brec_oap": lambda V, l: brec_mo.oap_canonicalize(V, l),
    })
    run_all("comb_lap", {
        "brec_m3_refined_comb": lambda V, l: brec.spectral_m3_canonicalize(
            V, l, refine_ordering=True),
        "brec_map_comb": lambda V, l: brec_mo.map_canonicalize(V, l),
        "brec_oap_comb": lambda V, l: brec_mo.oap_canonicalize(V, l),
    })

    # norm-value signatures of the transformer harness
    sigs = []
    for s in specs:
        lam, V = s["norm_lap_f32"]
        lam = lam.astype(np.float64)
        V = V.astype(np.float64)
        mults = brec.detect_multiplicities(lam)
        sigs.append(brec.compute_norm_signature(V, mults))
    out["fn__brec_norm_sig__V"], out["fn__brec_norm_sig__off"] = pack(sigs)

    # transformer window: lowest num_vecs + buffer columns canonicalized, first num_vecs kept
    k_win, buf = 6, 3
    wins = []
    for s in specs:
        lam, V = s["norm_lap_f32"]
        lam = lam.astype(np.float64)
        V = V.astype(np.float64)
        w = min(lam.shape[0], k_win + buf)
        Vc = brec.spectral_m3_canonicalize(V[:, :w], lam[:w], refine_ordering=True)
        wins.append(Vc[:, :min(lam.shape[0], k_win)])
    out["fn__brec_window_k6__V"], out["fn__brec_window_k6__off"] = pack(wins)

    # IndRefineNauty (the BREC PRiSM row): live indrefine.py on the same spectra
    indrefine_goldens(out, mats, [s["norm_lap_f32"] for s in specs])

    # ---- PE-level goldens (live DGL wrappers) ------------------------------
    pe_cases = {
        "zinc_gatedgcn": (fresh.spectral_m3_positional_encoding, {}, 8),
        "zinc_gatedgcn_heat_t2": (fresh.spectral_m3_heat_positional_encoding, {"t": 2.0}, 8),
        "zinc_pna": (fresh.spectral_qr_refined_eigval_positional_encoding, {}, 8),
        "moltox21_gatedgcn": (fresh.spectral_m3_inv_sqrt_lap_positional_encoding, {}, 3),
        "moltox21_pna": (fresh.spectral_m3_eigval_positional_encoding, {}, 16),
        "molpcba_gatedgcn": (fresh.spectral_qr_refined_eigval_positional_encoding, {}, 3),
        "molpcba_gatedgcn_plain": (fresh.spectral_qr_refined_positional_encoding, {}, 3),
        "molpcba_pna": (fresh.spectral_qr_refined_eigval_positional_encoding, {}, 16),
    }
    gl = [dgl_graph(A) for A in mats]
    for preset, (fn, kw, k0) in pe_cases.items():
        for k in sorted({k0, *PE_EXTRA_KS}):
            arrs = []
            for g in gl:
                g = fn(g, k, **kw)
                arrs.append(g.ndata["pos_enc"].numpy().astype(np.float32))
            out[f"pe__{preset}__k{k}"] = np.concatenate(arrs, axis=0)

    # baselines through the live molecular wrappers
    for name, fn in (("map", mol.map_positional_encoding),
                     ("oap", mol.oap_positional_encoding),
                     ("lap", mol.lap_positional_encoding)):
        for k in (3, 8):
            arrs = []
            for A in mats:
                g = fn(dgl_graph(A), k)
                arrs.append(g.ndata["pos_enc"].detach().numpy().astype(np.float32))
            out[f"pe__{name}__k{k}"] = np.concatenate(arrs, axis=0)

    # ---- GF(2) reference (Algorithm 1) -------------------------------------
    rng = np.random.default_rng(0)
    gf_inputs, gf_s, gf_canon, gf_part, gf_fast_idx, gf_fast = [], [], [], [], [], []
    for t in range(10):
        n = int(rng.integers(5, 12))
        M = rng.standard_normal((n, n))
        M = (M + M.T) / 2
        _, U = np.linalg.eigh(M)
        res = gf2ref.prism_canonicalize(U)
        gf_inputs.append(U)
        gf_s.append(res.sign_vector)
        gf_canon.append(res.canonical)
        gf_part.append(res.partition)
        if len(set(map(tuple, np.round(np.abs(U), 8).tolist()))) == n:
            gf_fast_idx.append(t)
            gf_fast.append(gf2ref.fastsign(U))
    # symmetric graphs with simple spectrum (non-injective signature)
    for key in ("path8", "path3", "binary_tree"):
        if key not in names:
            continue
        A = graphs[key]
        lam, U = np.linalg.eigh(np.diag(A.sum(1)) - A + np.diag(np.arange(A.shape[0]) * 1e-3))
        res = gf2ref.prism_canonicalize(U)
        gf_inputs.append(U)
        gf_s.append(res.sign_vector)
        gf_canon.append(res.canonical)
        gf_part.append(res.partition)
    out["gf2__U"], out["gf2__U_off"] = pack(gf_inputs)
    out["gf2__sizes"] = np.array([u.shape[0] for u in gf_inputs], dtype=np.int64)
    out["gf2__s"], out["gf2__s_off"] = pack(gf_s)
    out["gf2__canon"], out["gf2__canon_off"] = pack(gf_canon)
    out["gf2__part"], out["gf2__part_off"] = pack(gf_part)
    out["gf2__fast_idx"] = np.array(gf_fast_idx, dtype=np.int64)
    out["gf2__fast"], out["gf2__fast_off"] = pack(gf_fast)

    path = os.path.abspath(args.out)
    np.savez_compressed(path, **out)
    print(f"wrote {path}: {os.path.getsize(path) / 1e6:.2f} MB, {len(names)} graphs")


if __name__ == "__main__":
    main()
