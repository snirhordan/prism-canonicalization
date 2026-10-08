"""ZINC 12K (benchmarking-gnns split: 10000 / 1000 / 1000 graphs).

The dataset is the pickle ``ZINC.pkl`` of benchmarking-gnns (Dwivedi et al., MIT License),
downloaded by ``scripts/download_data.py`` to ``<data_root>/molecules/ZINC.pkl``.
"""

import os
import pickle

import dgl
import torch
import torch.utils.data


class MoleculeDGL(torch.utils.data.Dataset):
    """One split of ZINC: a list of DGL graphs (atom types in ``ndata['feat']``, bond types in
    ``edata['feat']``) and their regression targets.

    The pickle of benchmarking-gnns stores instances of this class under the module path
    ``data.molecules``; ``_Unpickler`` maps that path here.
    """

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """Return the idx-th sample as (DGLGraph, label)."""
        return self.graph_lists[idx], self.graph_labels[idx]


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'data.molecules' and name == 'MoleculeDGL':
            return MoleculeDGL
        if module == 'dgl.heterograph' and name == 'DGLHeteroGraph':
            return dgl.DGLGraph  # the graph class was renamed after the pickle was written
        return super().find_class(module, name)


class MoleculeDataset(torch.utils.data.Dataset):
    def __init__(self, name, data_root='data'):
        """Load ZINC from ``<data_root>/molecules/<name>.pkl``."""
        if name != 'ZINC':
            raise ValueError("only the 12K subset 'ZINC' is supported")
        self.name = name
        path = os.path.join(data_root, 'molecules', name + '.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError("%s not found; run scripts/download_data.py --data_root %s" % (path, data_root))
        with open(path, 'rb') as f:
            f = _Unpickler(f).load()
        self.train = f[0]
        self.val = f[1]
        self.test = f[2]
        self.num_atom_type = f[3]
        self.num_bond_type = f[4]

    def subset(self, n_train, n_val, n_test):
        """Keep the first graphs of every split (smoke tests)."""
        for split, n in ((self.train, n_train), (self.val, n_val), (self.test, n_test)):
            split.graph_lists = split.graph_lists[:n]
            split.graph_labels = split.graph_labels[:n]
            split.n_samples = len(split.graph_lists)

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(labels).unsqueeze(1)
        tab_sizes_n = [graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n = [torch.FloatTensor(size, 1).fill_(1. / float(size)) for size in tab_sizes_n]
        snorm_n = torch.cat(tab_snorm_n).sqrt()
        batched_graph = dgl.batch(graphs)
        return batched_graph, labels, snorm_n
