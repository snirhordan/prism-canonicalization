"""The vectorized padding mask of MaskedGINDeepSigns equals the loop of the original implementation."""

import pytest

torch = pytest.importorskip("torch")
dgl = pytest.importorskip("dgl")

from helpers import random_molecule  # noqa: E402
import numpy as np  # noqa: E402
from layers.deepsigns import MaskedGINDeepSigns  # noqa: E402


class OriginalMaskedGINDeepSigns(MaskedGINDeepSigns):
    """Forward pass of the original implementation (mask built with a Python loop)."""

    device = torch.device("cpu")

    def batched_n_nodes(self, n_nodes):
        return torch.cat([size * torch.ones(size).to(self.device) for size in n_nodes])

    def forward(self, g, x):
        if self.use_signnet:
            x = self.enc(g, x) + self.enc(g, -x)
        else:
            x = self.enc(g, x)
        orig_shape = x.shape
        n_nodes = self.batched_n_nodes(g.batch_num_nodes().unsqueeze(1))
        mask = torch.cat([torch.arange(orig_shape[1]).unsqueeze(0) for i in range(orig_shape[0])])
        mask = (mask.to(self.device) < n_nodes.unsqueeze(1)).bool()
        x[~mask] = 0
        x = x.sum(dim=1)
        x = self.rho(x)
        return x.reshape(orig_shape[0], self.k, 1)


@pytest.mark.parametrize("use_signnet", [False, True])
def test_mask_matches_original(use_signnet):
    rng = np.random.default_rng(0)
    sizes = [4, 7, 12, 3]  # includes graphs with fewer nodes than k
    g = dgl.batch([random_molecule(rng, n=n) for n in sizes])
    x = torch.randn(g.num_nodes(), 8, 1)
    kwargs = dict(in_channels=1, hidden_channels=16, out_channels=16, num_layers=3, k=8, use_bn=True,
                  dropout=0.0, activation="relu", use_signnet=use_signnet)
    new = MaskedGINDeepSigns(**kwargs).eval()
    old = OriginalMaskedGINDeepSigns(**kwargs).eval()
    old.load_state_dict(new.state_dict())
    assert torch.equal(new(g, x.clone()), old(g, x.clone()))


def test_sign_invariance_of_signnet_mode():
    rng = np.random.default_rng(1)
    g = dgl.batch([random_molecule(rng, n=n) for n in (6, 9)])
    x = torch.randn(g.num_nodes(), 5, 1)
    net = MaskedGINDeepSigns(1, 16, 16, 3, 5, use_bn=True, dropout=0.0, use_signnet=True).eval()
    assert torch.allclose(net(g, x.clone()), net(g, -x.clone()), atol=1e-5)
