"""Download the datasets of the molecular experiments into ``<data_root>`` (default ``./data``).

    python scripts/download_data.py [--data_root data] [zinc] [moltox21] [molpcba]

ZINC 12K is the pickle of benchmarking-gnns (Dwivedi et al., MIT License), stored as
``<data_root>/molecules/ZINC.pkl``.  MolTox21 and MolPCBA are fetched by the ``ogb`` package into
``<data_root>/ogb``.  Nothing is downloaded twice.
"""

import argparse
import builtins
import os
import sys
import urllib.request

ZINC_URL = "https://data.dgl.ai/dataset/benchmarking-gnns/ZINC.pkl"
OGB_NAMES = {"moltox21": "ogbg-moltox21", "molpcba": "ogbg-molpcba"}


def download_zinc(data_root):
    target = os.path.join(data_root, "molecules", "ZINC.pkl")
    if os.path.exists(target):
        print("ZINC: %s exists" % target)
        return
    os.makedirs(os.path.dirname(target), exist_ok=True)
    tmp = target + ".part"
    print("ZINC: downloading %s" % ZINC_URL, flush=True)
    urllib.request.urlretrieve(ZINC_URL, tmp)
    os.replace(tmp, target)
    print("ZINC: wrote %s (%.1f MB)" % (target, os.path.getsize(target) / 1e6))


def download_ogb(name, data_root):
    from ogb.graphproppred import DglGraphPropPredDataset

    if not sys.stdin.isatty():  # ogb may ask whether to update a stale copy; answer yes when unattended
        builtins.input = lambda *args, **kwargs: "y"
    root = os.path.join(data_root, "ogb")
    dataset = DglGraphPropPredDataset(name=name, root=root)
    print("%s: %d graphs in %s" % (name, len(dataset), root))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", default=os.environ.get("PRISM_DATA_ROOT", "data"))
    parser.add_argument("datasets", nargs="*", help="any of zinc, moltox21, molpcba (default: all three)")
    args = parser.parse_args()
    wanted = args.datasets or ["zinc", "moltox21", "molpcba"]
    unknown = [d for d in wanted if d not in ("zinc", "moltox21", "molpcba")]
    if unknown:
        parser.error("unknown dataset(s): %s" % ", ".join(unknown))
    for name in wanted:
        if name == "zinc":
            download_zinc(args.data_root)
        else:
            download_ogb(OGB_NAMES[name], args.data_root)


if __name__ == "__main__":
    main()
