"""Shared fixtures: the golden file produced by ``dev/make_golden.py``."""

import os

import numpy as np
import pytest

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "golden.npz")


class Golden:
    """Accessor for graphs, eigendecompositions and live outputs in golden.npz."""

    def __init__(self, path=FIXTURE):
        self.z = np.load(path)
        self.names = [str(s) for s in self.z["meta__names"]]
        self.sizes = self.z["meta__sizes"]
        self.node_off = np.concatenate([[0], np.cumsum(self.sizes)])

    def _slice(self, flat, off, i, shape):
        return np.asarray(flat[off[i]:off[i + 1]]).reshape(shape)

    def adj(self, i):
        n = int(self.sizes[i])
        return self._slice(self.z["meta__adj"], self.z["meta__adj_off"], i, (n, n)).astype(np.float64)

    def spec(self, op, i):
        n = int(self.sizes[i])
        c = n - 1 if op == "norm_adj" else n
        lam = self._slice(self.z[f"spec__{op}__lam"], self.z[f"spec__{op}__lam_off"], i, (c,))
        V = self._slice(self.z[f"spec__{op}__V"], self.z[f"spec__{op}__V_off"], i, (n, c))
        return lam.astype(np.float64), V.astype(np.float64)

    def fn(self, key, i, op):
        lam, V = self.spec(op, i)
        return self._slice(self.z[f"fn__{key}__V"], self.z[f"fn__{key}__off"], i, V.shape)

    def fn_shape(self, key, i, shape):
        return self._slice(self.z[f"fn__{key}__V"], self.z[f"fn__{key}__off"], i, shape)

    def fn_order(self, key, i):
        n = int(self.sizes[i])
        return self._slice(self.z[f"fn__{key}__order"], self.z[f"fn__{key}__order_off"], i, (n,))

    def pe(self, key, i):
        rows = self.z[key]
        return rows[self.node_off[i]:self.node_off[i + 1]]

    def keys(self, prefix):
        return sorted(k for k in self.z.files if k.startswith(prefix))


@pytest.fixture(scope="session")
def golden():
    if not os.path.exists(FIXTURE):
        pytest.skip("tests/fixtures/golden.npz missing; run dev/make_golden.py")
    return Golden()


def graph_ids(golden_obj=None):
    if not os.path.exists(FIXTURE):
        return []
    return list(range(len(Golden().names)))


def graph_id_names():
    if not os.path.exists(FIXTURE):
        return []
    return Golden().names
