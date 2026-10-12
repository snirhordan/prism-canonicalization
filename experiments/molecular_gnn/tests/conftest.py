import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # the experiment directory: layers, nets, graphdata, train, utils

import utils.preload  # noqa: E402,F401  (dgl needs libnvrtc.so.11.2, see utils/preload.py)
