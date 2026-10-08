# Third-party notices: experiments/molecular_gnn

Parts of this directory are adapted from the following MIT-licensed projects. Each file adapted from
them says so in its header.

| project | adapted in this directory | copyright |
|---|---|---|
| PKU-ML/canonicalization (<https://github.com/PKU-ML/canonicalization>), Ma et al., *A Canonicalization Perspective on Invariant and Equivariant Learning*, NeurIPS 2024 | training harness, data pipeline, configs, network wiring | Copyright (c) 2024 PKU-ML |
| SignNet-BasisNet (<https://github.com/cptq/SignNet-BasisNet>), Lim et al. | `layers/deepsigns.py`, `layers/gnns.py`, `layers/mlp.py` | Copyright (c) 2022 Derek Lim |
| gnn-lspe (<https://github.com/vijaydwivedi75/gnn-lspe>), Dwivedi et al. | `layers/pna_layer.py`, `layers/pna_utils.py`, network classes | Copyright (c) 2022 Vijay Prakash Dwivedi, Anh Tuan Luu, Thomas Laurent, Yoshua Bengio and Xavier Bresson |
| benchmarking-gnns (<https://github.com/graphdeeplearning/benchmarking-gnns>), Dwivedi et al. | `layers/gatedgcn_layer.py`, `layers/mlp_readout_layer.py`, ZINC loading and collation, training loops | Copyright (c) 2022 Vijay Prakash Dwivedi, Chaitanya K. Joshi, Anh Tuan Luu, Thomas Laurent, Yoshua Bengio, Xavier Bresson |
| PNA (<https://github.com/lukecavabarrett/pna>), Corso et al. | aggregators, scalers and dense layers in `layers/pna_utils.py` | Copyright (c) 2020 Gabriele Corso, Luca Cavalleri |

The data are not part of the repository: ZINC 12K is the benchmarking-gnns subset of the ZINC database, the
MolTox21 and MolPCBA datasets come from the Open Graph Benchmark (<https://ogb.stanford.edu>), whose `ogb`
package and the `dgl` library are installed as dependencies and not redistributed.

## MIT License (identical text for all five projects above)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
