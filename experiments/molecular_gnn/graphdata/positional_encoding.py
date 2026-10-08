"""Positional encodings of the molecular graphs.

Every canonicalizing method of the paper goes through ``prism``:

``prism``    ``prism.PRiSM.from_preset(<preset>)``, the variant used for the table cell
``map``      ``prism.baselines.MAP.from_profile('gnn')``
``oap``      ``prism.baselines.OAP.from_profile('gnn')``
``lap_pe``, ``signnet``
             ``prism.baselines.lap_pe`` (raw eigenvectors; the sign ambiguity is handled by the model)

All of them map the adjacency matrix of a graph and the number of columns k to a float32
array of shape (n, k).  The encodings of a dataset are cached under ``cache_dir`` in a file
whose name carries a hash of the full method configuration, so a cache is never reused for a
different preset, k or prism version.
"""

import hashlib
import json
import multiprocessing
import os
import pickle
import time

import numpy as np
import torch

import prism
from prism import PRiSM
from prism.baselines import MAP, OAP, lap_pe


def make_pe_method(pe_init, prism_preset=None):
    """Return ``(fn, tag, config)``: ``fn(A, k)`` is the encoding, ``tag`` names the cache, ``config``
    is a JSON-serializable description used for the cache hash."""
    if pe_init == 'prism':
        if prism_preset is None:
            raise ValueError("pe_init='prism' needs a prism_preset (see prism.presets.PRESETS)")
        canon = PRiSM.from_preset(prism_preset)
        return canon.pos_enc, 'prism_' + prism_preset, {'prism': canon.get_config()}
    if prism_preset is not None:
        raise ValueError("prism_preset is only used with pe_init='prism'")
    if pe_init == 'map':
        method = MAP.from_profile('gnn')
        return method.pos_enc, 'map', {'map': dict(vars(method))}
    if pe_init == 'oap':
        method = OAP.from_profile('gnn')
        return method.pos_enc, 'oap', {'oap': dict(vars(method))}
    if pe_init in ('lap_pe', 'signnet'):
        return lap_pe, 'lap', {'lap_pe': 'eig(I - D^-1/2 A D^-1/2), columns 1..k'}
    raise ValueError("unknown pe_init %r" % pe_init)


def cache_path(cache_dir, dataset_name, tag, config, k):
    description = json.dumps({'config': config, 'k': k, 'prism': prism.__version__, 'dataset': dataset_name},
                             sort_keys=True, default=str)
    digest = hashlib.sha1(description.encode()).hexdigest()[:10]
    return os.path.join(cache_dir, '%s_%s' % (dataset_name, tag), 'k=%d_%s.pkl' % (k, digest))


_WORKER_FN = None


def _init_worker(fn):
    global _WORKER_FN
    _WORKER_FN = fn
    torch.set_num_threads(1)


def _encode(args):
    A, k = args
    return _WORKER_FN(A, k)


def compute_pos_enc(graphs, fn, k, workers=1, log_every=50000):
    """Positional encodings of a list of DGL graphs, as a list of (n, k) float32 arrays."""
    t0 = time.time()
    adjacencies = (g.adj_external(scipy_fmt='csr') for g in graphs)
    if workers <= 1:
        out = []
        for i, A in enumerate(adjacencies):
            out.append(fn(A, k))
            if log_every and (i + 1) % log_every == 0:
                print('[PE] %d/%d graphs, %.0fs' % (i + 1, len(graphs), time.time() - t0), flush=True)
        return out
    ctx = multiprocessing.get_context('fork')
    with ctx.Pool(workers, initializer=_init_worker, initargs=(fn,)) as pool:
        out = []
        for i, pe in enumerate(pool.imap(_encode, ((A, k) for A in adjacencies), chunksize=64)):
            out.append(pe)
            if log_every and (i + 1) % log_every == 0:
                print('[PE] %d/%d graphs, %.0fs' % (i + 1, len(graphs), time.time() - t0), flush=True)
    return out


def add_pos_enc(dataset, pe_init, prism_preset, k, cache_dir='cache', workers=1, use_cache=True):
    """Store the positional encoding of every graph of ``dataset`` in ``g.ndata['pos_enc']``.

    ``dataset.train``, ``.val`` and ``.test`` hold the graphs in ``graph_lists``.  The encodings
    are read from / written to the cache when ``use_cache`` is set.
    """
    fn, tag, config = make_pe_method(pe_init, prism_preset)
    path = cache_path(cache_dir, dataset.name, tag, config, k)
    splits = (dataset.train, dataset.val, dataset.test)
    cached = None
    if use_cache and os.path.exists(path):
        with open(path, 'rb') as f:
            cached = pickle.load(f)
        if [len(c) for c in cached] != [len(s.graph_lists) for s in splits]:
            print('[PE] cache %s does not match the dataset, recomputing' % path, flush=True)
            cached = None
        else:
            print('[PE] loaded %s' % path, flush=True)
    if cached is None:
        t0 = time.time()
        cached = [compute_pos_enc(s.graph_lists, fn, k, workers=workers) for s in splits]
        print('[PE] computed %s k=%d for %s in %.1fs' % (tag, k, dataset.name, time.time() - t0), flush=True)
        if use_cache:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = '%s.tmp%d' % (path, os.getpid())
            with open(tmp, 'wb') as f:
                pickle.dump(cached, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)  # atomic: concurrent runs never see a partial file
    for split, pes in zip(splits, cached):
        for g, pe in zip(split.graph_lists, pes):
            g.ndata['pos_enc'] = torch.from_numpy(np.ascontiguousarray(pe))
    return path
