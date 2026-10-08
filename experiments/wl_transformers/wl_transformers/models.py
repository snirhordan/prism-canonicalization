"""Models of the transformer / BREC harness.

``Transformer`` is the pair-token graph transformer of "Aligning Transformers
with Weisfeiler-Leman" (Mueller and Morris, ICML 2024) with a Laplacian
positional encoding (LPE).  The canonical variants replace the random sign flips
of LPE by canonical eigenvectors and can add the canonical vertex rank as
sinusoidal or rotary (RoPE) position information.  ``GNN`` is the GINE backbone
with the same eigenvector embedding, used for the BREC benchmark.

Adapted from https://github.com/luis-mueller/wl-transformers (MIT License,
Copyright (c) 2024 Luis Mueller).  See ``THIRD_PARTY_NOTICES``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool
from torch_geometric.utils import to_dense_batch
from torchvision.ops import stochastic_depth

from .pe_types import resolve_pe_type


class PaddedEigenEmbedding(torch.nn.Module):
    """phi: embeds every (eigenvector entry, eigenvalue) pair; NaN padding maps to 0."""

    def __init__(self, embed_dim, bias, phi_depth=2):
        super().__init__()
        layers = []
        in_dim = 2
        for i in range(phi_depth):
            out_dim = 2 * embed_dim if i < phi_depth - 1 else embed_dim
            layers.append(torch.nn.Linear(in_dim, out_dim, bias=bias))
            layers.append(torch.nn.ReLU())
            in_dim = out_dim
        self.ffn = torch.nn.Sequential(*layers)

    def forward(self, eigvecs, eigvals):
        x = torch.stack((eigvecs, eigvals), 2)
        empty_mask = torch.isnan(x)
        x[empty_mask] = 0
        return self.ffn(x)


class EigenEmbedding(torch.nn.Module):
    """DeepSets over the eigenvector columns: ``rho(sum_i phi(v_i, lambda_i))``.

    Non-canonical (``canonical=False``) embeddings apply an independent random
    sign flip to every eigenvector during training.  ``position_aware`` adds a
    learnable perturbation to the eigenvalues.
    """

    def __init__(self, num_vecs, eigen_dim, bias=True, position_aware=False,
                 canonical=False, minimal_rho=False, phi_depth=2):
        super().__init__()
        self.num_vecs = num_vecs
        self.position_aware = position_aware
        self.canonical = canonical
        if position_aware:
            self.eps = torch.nn.Parameter(1e-12 * torch.arange(num_vecs).unsqueeze(0))

        self.phi = PaddedEigenEmbedding(eigen_dim, bias, phi_depth=phi_depth)
        if minimal_rho:
            self.rho = torch.nn.Sequential(
                torch.nn.Linear(eigen_dim, eigen_dim, bias=bias),
                torch.nn.ReLU(),
            )
        else:
            self.rho = torch.nn.Sequential(
                torch.nn.Linear(eigen_dim, 2 * eigen_dim, bias=bias),
                torch.nn.ReLU(),
                torch.nn.Linear(2 * eigen_dim, eigen_dim, bias=bias),
                torch.nn.ReLU(),
            )

    def forward(self, eigvals, eigvecs, edge_index, batch):
        eigvecs = eigvecs[:, : self.num_vecs]
        eigvals = eigvals[:, : self.num_vecs]

        if self.training and not self.canonical:
            sign_flip = torch.rand(eigvecs.size(1), device=eigvecs.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            eigvecs = eigvecs * sign_flip.unsqueeze(0)

        if self.position_aware:
            eigvals = eigvals + self.eps[:, : self.num_vecs]

        eigen_embed = self.phi(eigvecs, eigvals)
        return self.rho(eigen_embed.sum(1))


def _eigen_embedding(pe_type, num_vecs, eigen_dim, bias, phi_depth):
    pe = resolve_pe_type(pe_type)
    return EigenEmbedding(
        num_vecs, eigen_dim, bias,
        position_aware=pe.eps,
        canonical=pe.canon is not None,
        minimal_rho=pe.minimal_rho,
        phi_depth=phi_depth,
    )


class PairEncoder(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.encoder = torch.nn.Linear(2 * in_dim, out_dim, bias=bias)

    def forward(self, x, token_index):
        return self.encoder(torch.cat([x[token_index[0]], x[token_index[1]]], -1))


class PairTokenizer(torch.nn.Module):
    """Tokens are the edges and self loops of a graph (order 2)."""

    def __init__(self, atom_encoder, bond_encoder, num_phi_layers, num_vecs,
                 max_degree, eigen_dim, degree_dim, embed_dim, bias, pe_type):
        super().__init__()
        pe = resolve_pe_type(pe_type)
        self.pe_type = pe_type
        self.eigen_embed = _eigen_embedding(pe_type, num_vecs, eigen_dim, bias, num_phi_layers)
        self.eigen_encoder = PairEncoder(eigen_dim, embed_dim, bias)

        # Sinusoidal rank embedding of the canonical vertex order.
        if pe.rank == "sinusoidal":
            self._sin_proj = torch.nn.Linear(eigen_dim, eigen_dim)

        if degree_dim > 0:
            self.degree_embed = torch.nn.Embedding(max_degree, degree_dim)
            self.degree_encoder = PairEncoder(degree_dim, embed_dim, bias)
        self.atom_encoder = atom_encoder
        self.node_encoder = PairEncoder(embed_dim, embed_dim, bias)
        self.bond_encoder = bond_encoder
        self.graph_token = torch.nn.Parameter(torch.zeros((1, 1, embed_dim)))

    @staticmethod
    def _sinusoidal_encoding(rank, dim, device):
        pos = rank.unsqueeze(1).float()
        i = torch.arange(0, dim, 2, device=device).float()
        div = torch.exp(i * -(math.log(10000.0) / dim))
        enc = torch.zeros(rank.size(0), dim, device=device)
        enc[:, 0::2] = torch.sin(pos * div)
        enc[:, 1::2] = torch.cos(pos * div)
        return enc

    def forward(self, data):
        token_index, real_edges = data.token_index, data.real_edges
        x = self.atom_encoder(data.x)

        if hasattr(self, "degree_embed"):
            d = self.degree_embed(data.degrees)
        l = self.eigen_embed(data.eigvals, data.eigvecs, data.edge_index, data.batch)

        if hasattr(self, "_sin_proj") and hasattr(data, "canon_rank"):
            sin_enc = self._sinusoidal_encoding(data.canon_rank, l.size(1), l.device)
            l = l + self._sin_proj(sin_enc)

        e = torch.zeros(
            (token_index.size(1), x.size(1)), device=x.device, dtype=x.dtype
        )
        if not hasattr(data, "edge_attr") or data.edge_attr is None:
            data.edge_attr = torch.ones(
                (data.edge_index.size(1), 1), device=data.x.device, dtype=data.x.dtype
            )
        e[real_edges] = self.bond_encoder(data.edge_attr)

        token_embed = (
            e + self.eigen_encoder(l, token_index) + self.node_encoder(x, token_index)
        )

        if hasattr(self, "degree_encoder"):
            token_embed = token_embed + self.degree_encoder(d, token_index)

        if not hasattr(data, "batch") or data.batch is None:
            batch = torch.zeros(data.x.size(0), device=data.x.device, dtype=torch.long)[
                token_index[0]
            ]
        else:
            batch = data.batch[token_index[0]]
        token_embed, mask = to_dense_batch(token_embed, batch)

        graph_token = self.graph_token.repeat(token_embed.size(0), 1, 1)
        token_embed = torch.cat([graph_token, token_embed], 1)

        padding_mask = torch.zeros(
            (mask.size(0), mask.size(1) + 1), dtype=mask.dtype, device=mask.device
        )
        padding_mask[:, 0] = 1
        padding_mask[:, 1:] = mask

        return token_embed, padding_mask


class LayerNorm(torch.nn.Module):
    """LayerNorm with an optional bias."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(ndim))
        self.bias = torch.nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class RotaryEmbedding(torch.nn.Module):
    """Rotary position embedding (RoPE) following RoFormer / LLaMA.

    Precomputes the frequency table; the positions are the canonical vertex
    ranks produced by the canonicalizer.
    """

    def __init__(self, dim, max_positions=4096, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_positions = max_positions

    def forward(self, positions):
        """``positions``: (B, T) integers.  Returns cos, sin of shape (B, T, dim)."""
        pos = positions.unsqueeze(-1).float()
        freqs = pos * self.inv_freq.unsqueeze(0).unsqueeze(0)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def _rotate_half(x):
    """[x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]"""
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_pos_emb(x, cos, sin):
    """x: (B, H, T, D); cos, sin: (B, T, D)."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return x * cos + _rotate_half(x) * sin


class FastSelfAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, bias=True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.c_attn = torch.nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.c_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_dropout = torch.nn.Dropout(dropout)
        self.n_head = num_heads
        self.n_embd = embed_dim
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

    def forward(self, x, key_padding_mask, rope_cos=None, rope_sin=None):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if rope_cos is not None and rope_sin is not None:
            # cos/sin are float32; return to the dtype of v (what CUDA autocast
            # does inside scaled_dot_product_attention)
            q = apply_rotary_pos_emb(q, rope_cos, rope_sin).to(v.dtype)
            k = apply_rotary_pos_emb(k, rope_cos, rope_sin).to(v.dtype)

        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=~key_padding_mask,
            dropout_p=self.dropout if self.training else 0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return [y]


class TransformerLayer(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, attention_dropout,
                 bias=False, drop_path=0.0):
        super().__init__()

        self.ln_1 = LayerNorm(embed_dim, bias)
        self.ln_2 = LayerNorm(embed_dim, bias)

        self.attention = FastSelfAttention(embed_dim, num_heads, attention_dropout, bias=bias)

        self.attention_dropout = torch.nn.Dropout(dropout)
        self.stochastic_depth = lambda x: (
            stochastic_depth(x, drop_path, "batch", self.training)
            if drop_path > 0
            else x
        )

        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim, bias=bias),
            torch.nn.GELU(),
            torch.nn.Linear(embed_dim, embed_dim, bias=bias),
        )

    def forward(self, x, mask=None, rope_cos=None, rope_sin=None):
        _x = x
        x = self.ln_1(x)
        x = self.attention(x, key_padding_mask=~mask,
                           rope_cos=rope_cos, rope_sin=rope_sin)[0]

        x = self.attention_dropout(x)
        x = self.stochastic_depth(x)

        x = _x + x
        return x + self.stochastic_depth(self.ffn(self.ln_2(x)))


def init_weights(module):
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class Transformer(torch.nn.Module):
    """Pair-token graph transformer with a graph token and a graph-level head."""

    def __init__(self, atom_encoder, bond_encoder, num_phi_layers, num_vecs, max_degree,
                 eigen_dim, degree_dim, num_layers, embed_dim, num_heads, output_dim,
                 dropout=0.0, attention_dropout=0.0, bias=True, stochastic_depth=False,
                 token_ln=False, pe_type="LPE", rope_base_freq=10000.0):
        super().__init__()
        pe = resolve_pe_type(pe_type)

        self.tokenizer = PairTokenizer(
            atom_encoder, bond_encoder, num_phi_layers, num_vecs, max_degree,
            eigen_dim, degree_dim, embed_dim, bias, pe_type,
        )

        if token_ln:
            self.token_ln = LayerNorm(embed_dim, bias)

        self.layers = torch.nn.Sequential(
            *[
                TransformerLayer(
                    embed_dim,
                    num_heads,
                    dropout,
                    attention_dropout,
                    bias,
                    drop_path=(0.1 * (i + 1) / num_layers) if stochastic_depth else 0.0,
                )
                for i in range(num_layers)
            ]
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim, bias=bias),
            torch.nn.GELU(),
            LayerNorm(embed_dim, bias),
            torch.nn.Linear(embed_dim, output_dim, bias=bias),
        )
        self.num_heads = num_heads
        self.pe_type = pe_type

        # RoPE positions are the canonical ranks of the vertices.
        if pe.rank == "rope":
            self.rotary_emb = RotaryEmbedding(embed_dim // num_heads, base=rope_base_freq)

    def forward(self, data):
        x, mask = self.tokenizer(data)

        rope_cos, rope_sin = None, None
        if hasattr(self, "rotary_emb") and hasattr(data, "canon_rank"):
            token_index = data.token_index
            batch = (data.batch[token_index[0]]
                     if hasattr(data, "batch") and data.batch is not None
                     else torch.zeros(token_index.size(1), dtype=torch.long, device=x.device))
            # Per-token position = canonical rank of the source vertex; the
            # graph token sits at position 0.
            token_positions = data.canon_rank[token_index[0]]
            token_pos_dense, _ = to_dense_batch(token_positions, batch)
            graph_pos = torch.zeros(token_pos_dense.size(0), 1, dtype=torch.long, device=x.device)
            positions = torch.cat([graph_pos, token_pos_dense], dim=1)
            rope_cos, rope_sin = self.rotary_emb(positions)

        mask = mask.unsqueeze(1) == mask.unsqueeze(2)
        mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        if hasattr(self, "token_ln"):
            x = self.token_ln(x)

        for layer in self.layers:
            x = layer(x, mask, rope_cos=rope_cos, rope_sin=rope_sin)

        return self.mlp(x[:, 0, :])

    def reset_parameters(self):
        self.apply(init_weights)


def create_transformer(cfg, atom_encoder, bond_encoder, output_dim, device):
    model = Transformer(
        atom_encoder=atom_encoder,
        bond_encoder=bond_encoder,
        num_phi_layers=cfg.num_phi_layers,
        num_vecs=cfg.num_vecs,
        max_degree=cfg.max_degree,
        eigen_dim=cfg.eigen_dim,
        degree_dim=cfg.degree_dim,
        num_layers=cfg.num_layers,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        output_dim=output_dim,
        dropout=cfg.dropout,
        attention_dropout=cfg.attention_dropout,
        bias=cfg.bias,
        stochastic_depth=cfg.stochastic_depth,
        token_ln=cfg.token_ln,
        pe_type=cfg.pe_type,
        rope_base_freq=getattr(cfg, "rope_base_freq", 10000.0),
    ).to(device)
    model.reset_parameters()
    return model


class GINLayer(torch.nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.conv = GINEConv(torch.nn.Dropout(dropout))
        self.ffn = torch.nn.Sequential(
            torch.nn.BatchNorm1d(embed_dim),
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.GELU(),
            torch.nn.Linear(embed_dim, embed_dim),
        )
        self.norm = torch.nn.BatchNorm1d(embed_dim)

    def forward(self, data):
        x = self.conv(data.x, data.edge_index, data.edge_attr)
        x = data.x + x
        data.x = self.norm(x + self.ffn(x))
        return data


class GNN(torch.nn.Module):
    """GINE backbone; the eigenvector embedding is added to the node features."""

    def __init__(self, atom_encoder, bond_encoder, num_phi_layers, num_layers, embed_dim,
                 pooling, output_dim, num_vecs, dropout, pe_type):
        super().__init__()
        pe = resolve_pe_type(pe_type)
        if pe.rank is not None:
            raise ValueError(f"backbone=gnn does not use rank embeddings (pe_type={pe_type!r})")
        self.pe_type = pe_type
        self.atom_encoder = atom_encoder
        self.bond_encoder = bond_encoder
        self.eigen_embed = _eigen_embedding(pe_type, num_vecs, embed_dim, True, num_phi_layers)
        self.layers = torch.nn.ModuleList(
            [GINLayer(embed_dim, dropout) for _ in range(num_layers)]
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.GELU(),
            torch.nn.Linear(embed_dim, output_dim),
        )
        self.pooling = pooling

    def forward(self, data):
        data.x = self.atom_encoder(data.x)
        pe = self.eigen_embed(data.eigvals, data.eigvecs, data.edge_index, data.batch)
        data.x = data.x + pe
        data.edge_attr = self.bond_encoder(data.edge_attr)
        for layer in self.layers:
            data = layer(data)

        if self.pooling == "sum":
            return self.mlp(global_add_pool(data.x, data.batch))
        elif self.pooling == "mean":
            return self.mlp(global_mean_pool(data.x, data.batch))
        else:
            raise ValueError(f"Pooling {self.pooling} not supported")

    def reset_parameters(self):
        self.apply(init_weights)


def create_gnn(cfg, atom_encoder, bond_encoder, output_dim, device):
    model = GNN(
        atom_encoder=atom_encoder,
        bond_encoder=bond_encoder,
        num_phi_layers=cfg.num_phi_layers,
        num_layers=cfg.num_layers,
        embed_dim=cfg.embed_dim,
        pooling=cfg.pooling,
        output_dim=output_dim,
        num_vecs=cfg.num_vecs,
        dropout=cfg.dropout,
        pe_type=cfg.pe_type,
    ).to(device)
    model.reset_parameters()
    return model
