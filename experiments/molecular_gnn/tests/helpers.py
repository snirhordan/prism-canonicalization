"""Small synthetic molecular graphs for the tests."""

import numpy as np
import torch

import dgl


def random_molecule(rng, n=None, task="zinc"):
    """A bidirected ring with a few chords; node/edge features as the datasets provide them."""
    n = int(rng.integers(6, 13)) if n is None else n
    edges = {(i, (i + 1) % n) for i in range(n)}
    for _ in range(max(1, n // 4)):
        a, b = rng.choice(n, size=2, replace=False)
        edges.add((int(a), int(b)))
    src = [a for a, b in edges] + [b for a, b in edges]
    dst = [b for a, b in edges] + [a for a, b in edges]
    g = dgl.graph((src, dst), num_nodes=n)
    if task == "zinc":
        g.ndata["feat"] = torch.from_numpy(rng.integers(0, 28, size=n)).long()
        g.edata["feat"] = torch.from_numpy(rng.integers(1, 4, size=g.num_edges())).long()
    else:
        g.ndata["feat"] = torch.zeros(n, 9, dtype=torch.long)
        g.edata["feat"] = torch.zeros(g.num_edges(), 3, dtype=torch.long)
    return g


def fake_dataset(task="zinc", n_graphs=8, seed=0):
    from types import SimpleNamespace

    rng = np.random.default_rng(seed)

    def split(count):
        return SimpleNamespace(graph_lists=[random_molecule(rng, task=task) for _ in range(count)])

    return SimpleNamespace(name="fake_" + task, train=split(n_graphs), val=split(2), test=split(2))
