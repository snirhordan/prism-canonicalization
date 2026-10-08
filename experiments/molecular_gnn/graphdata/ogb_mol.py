"""OGB molecular property prediction datasets (MolTox21, MolPCBA) with the scaffold split.

The data are downloaded by OGB to ``<data_root>/ogb``.  As in the harness of Ma et al. (PKU-ML/
canonicalization), graphs with at most 5 nodes are dropped from every split.
"""

import os

import dgl
import torch
from ogb.graphproppred import DglGraphPropPredDataset, Evaluator
from torch.utils.data import Dataset


class OGBMOLDGL(Dataset):
    def __init__(self, data, split):
        self.split = split
        self.data = [g for g in data[self.split]]
        self.graph_lists = []
        self.graph_labels = []
        for g in self.data:
            if g[0].number_of_nodes() > 5:
                self.graph_lists.append(g[0])
                self.graph_labels.append(g[1])
        self.n_samples = len(self.graph_lists)

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """Return the idx-th sample as (DGLGraph, label)."""
        return self.graph_lists[idx], self.graph_labels[idx]


class OGBMOLDataset(Dataset):
    def __init__(self, name, data_root='data'):
        self.name = name.lower()
        self.dataset = DglGraphPropPredDataset(name=self.name, root=os.path.join(data_root, 'ogb'))
        split_idx = self.dataset.get_idx_split()
        self.train = OGBMOLDGL(self.dataset, split_idx['train'])
        self.val = OGBMOLDGL(self.dataset, split_idx['valid'])
        self.test = OGBMOLDGL(self.dataset, split_idx['test'])
        self.evaluator = Evaluator(name=self.name)

    def subset(self, n_train, n_val, n_test):
        """Keep the first graphs of every split (smoke tests)."""
        for split, n in ((self.train, n_train), (self.val, n_val), (self.test, n_test)):
            split.graph_lists = split.graph_lists[:n]
            split.graph_labels = split.graph_labels[:n]
            split.n_samples = len(split.graph_lists)

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        labels = torch.stack(labels)
        tab_sizes_n = [graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n = [torch.FloatTensor(size, 1).fill_(1. / float(size)) for size in tab_sizes_n]
        snorm_n = torch.cat(tab_snorm_n).sqrt()
        return batched_graph, labels, snorm_n
