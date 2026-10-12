"""Make ``import dgl`` work on nodes without a system CUDA 11 toolkit.

dgl 2.1.0+cu118 loads ``libgraphbolt_pytorch_*.so`` when it is imported, and that library needs
``libnvrtc.so.11.2``.  The environment of this experiment installs the library through pip
(``nvidia-cuda-nvrtc-cu11``, under ``site-packages/nvidia/cuda_nvrtc/lib``), but that directory is not on the
dynamic loader's search path, so ``import dgl`` raises ``ImportError: Cannot load Graphbolt C++ library`` on a
machine whose system libraries do not provide it.  Importing this module loads the pip copy first.  It does
nothing when the library is already available, and must be imported before ``dgl``.
"""

import ctypes
import glob
import os
import sys


def preload_nvrtc():
    try:
        ctypes.CDLL('libnvrtc.so.11.2')
        return
    except OSError:
        pass
    for root in sys.path:
        for path in glob.glob(os.path.join(root, 'nvidia', 'cuda_nvrtc', 'lib', 'libnvrtc.so.11*')):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                return
            except OSError:
                continue


preload_nvrtc()
