"""Positional encodings come from the prism package; caching and parallel evaluation do not change them."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
dgl = pytest.importorskip("dgl")

import prism  # noqa: E402
from prism.baselines import MAP, OAP, lap_pe  # noqa: E402
from graphdata.positional_encoding import add_pos_enc, cache_path, make_pe_method  # noqa: E402
from helpers import fake_dataset  # noqa: E402

METHODS = [("prism", "zinc_gatedgcn"), ("prism", "zinc_pna"), ("prism", "moltox21_gatedgcn"),
           ("prism", "moltox21_pna"), ("prism", "molpcba_gatedgcn"), ("prism", "molpcba_pna"),
           ("map", None), ("oap", None), ("lap_pe", None), ("signnet", None)]


def adjacency(g):
    return g.adj_external(scipy_fmt="csr")


@pytest.mark.parametrize("pe_init,preset", METHODS)
def test_method_matches_prism(pe_init, preset):
    ds = fake_dataset("zinc", n_graphs=3)
    fn, tag, config = make_pe_method(pe_init, preset)
    for g in ds.train.graph_lists:
        A = adjacency(g)
        out = fn(A, 8)
        assert out.shape == (g.num_nodes(), 8) and out.dtype == np.float32
        if pe_init == "prism":
            expected = prism.PRiSM.from_preset(preset).pos_enc(A, 8)
        elif pe_init == "map":
            expected = MAP.from_profile("gnn").pos_enc(A, 8)
        elif pe_init == "oap":
            expected = OAP.from_profile("gnn").pos_enc(A, 8)
        else:
            expected = lap_pe(A, 8)
        assert np.array_equal(out, expected)


def test_preset_is_required_for_prism_only():
    with pytest.raises(ValueError):
        make_pe_method("prism", None)
    with pytest.raises(ValueError):
        make_pe_method("map", "zinc_pna")
    with pytest.raises(ValueError):
        make_pe_method("unknown", None)


def test_cache_key_depends_on_preset_and_k():
    a = make_pe_method("prism", "zinc_pna")
    b = make_pe_method("prism", "zinc_gatedgcn")
    assert cache_path("c", "d", a[1], a[2], 8) != cache_path("c", "d", b[1], b[2], 8)
    assert cache_path("c", "d", a[1], a[2], 8) != cache_path("c", "d", a[1], a[2], 16)
    assert cache_path("c", "d", a[1], a[2], 8) == cache_path("c", "d", a[1], a[2], 8)


def test_add_pos_enc_cache_and_workers(tmp_path):
    ds = fake_dataset("zinc", n_graphs=6)
    path = add_pos_enc(ds, "prism", "zinc_pna", 8, cache_dir=str(tmp_path), workers=1)
    first = [g.ndata["pos_enc"].clone() for g in ds.train.graph_lists]
    assert first[0].dtype == torch.float32 and first[0].shape[1] == 8
    assert (tmp_path / path.split(str(tmp_path) + "/")[1]).exists()
    for g in ds.train.graph_lists:  # second call reads the cache
        del g.ndata["pos_enc"]
    add_pos_enc(ds, "prism", "zinc_pna", 8, cache_dir=str(tmp_path), workers=1)
    assert all(torch.equal(a, g.ndata["pos_enc"]) for a, g in zip(first, ds.train.graph_lists))
    add_pos_enc(ds, "prism", "zinc_pna", 8, cache_dir=str(tmp_path / "other"), workers=2)
    assert all(torch.equal(a, g.ndata["pos_enc"]) for a, g in zip(first, ds.train.graph_lists))
