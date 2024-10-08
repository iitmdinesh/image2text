from typing import Optional, Tuple

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from models.functions import normalize_gradients
from configs.models import (
    SelfAttentionConfig,
    TransformerConfig,
    MLPConfig,
    MoEConfig,
    SelfAttentionType,
)


class PeerLookupQueryUnit(nn.Module):
    def __init__(
        self,
        num_embed: int,
        emb_dim: int,
        topk: int
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(emb_dim, num_embed, bias=False)
        self.topk = topk

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        topk = torch.topk(self.linear(x), k=self.topk, dim=-1)
        return topk.values, topk.indices


class PeerLookup(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_units: int,
        topk: int,
        nhead: int = 1,
        query_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.query_dim: int = query_dim or (in_features // 2)
        self.residual = nn.Linear(in_features, out_features, bias=False)
        self.query_linear = nn.Linear(in_features, self.query_dim * nhead, bias=False)
        self.key_linear = nn.Linear(in_features, in_features * nhead, bias=False)
        self.nhead = nhead
        self.num_query_units = int(math.sqrt(num_units))
        self.topk = topk
        if self.num_query_units * self.num_query_units != num_units:
            raise ValueError(
                f"num_units must be a perfect square but {num_units} was not"
            )
        self.query_left = PeerLookupQueryUnit(
            self.num_query_units,
            self.query_dim,
            topk,
        )
        self.query_right = PeerLookupQueryUnit(
            self.num_query_units,
            self.query_dim,
            topk,
        )
        self.emb_in = nn.Embedding(num_units, in_features)
        self.emb_out = nn.Embedding(num_units, out_features)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        bs, seq_len, in_features = inp.shape
        x = self.query_linear(inp).view(bs, seq_len, -1, self.query_dim)
        inp_proj = self.key_linear(inp).view(bs, seq_len, -1, in_features)
        residual = self.residual(inp)

        left_v, left_i = self.query_left(x)
        right_v, right_i = self.query_right(x)

        cross = (left_v.unsqueeze(-1) + right_v.unsqueeze(-2)).view(
            bs, seq_len, -1, self.topk * self.topk
        )
        y = torch.topk(cross, k=self.topk, dim=-1)
        dot, indices = y.values, y.indices
        scores = F.softmax(dot, dim=-1)  # (b, s, h, k)

        left_i_selected = indices // self.topk
        right_i_selected = indices % self.topk
        left_i_trimmed = left_i.gather(-1, left_i_selected)
        right_i_trimmed = right_i.gather(-1, right_i_selected)

        final_indices = (
            left_i_trimmed * self.topk
            + right_i_trimmed
        )  # (b, s, h, k)

        inp_expert = self.emb_in(final_indices)
        out_expert = self.emb_out(final_indices)

        in_dot = torch.einsum("bshkd,bshd->bshk", inp_expert, inp_proj)
        in_act = self.act(in_dot)  # (b, s, h, k)

        final_weight = scores * in_act
        return (
            torch.einsum("bshk,bshkd->bsd", final_weight, out_expert)
            + residual
        )


class CosineVectorEmbedding(nn.Module):
    """
    LSH based vector embedding for highly non-linear ops
    """

    def __init__(self, inp_dim: int, emb_dim: int, n_proj: int = 16, num_bins: int = 20):
        super().__init__()
        self.register_buffer(
            'projection_mat',
            F.normalize(torch.randn((inp_dim, n_proj)), p=2.0, dim=0),
            persistent=True,
        )
        resolution = 2.0 / num_bins
        self.register_buffer(
            'grid',
            torch.linspace(-1, 1, num_bins + 1)[:-1] + 0.5 * resolution,
            persistent=True,
        )
        self.register_buffer(
            'pos_offset',
            ((num_bins + 1) * torch.arange(0, n_proj, dtype=torch.long)).long().reshape(-1, 1, 1),
            persistent=True,
        )
        self.emb = nn.EmbeddingBag((num_bins + 1) * n_proj, emb_dim)
        self.emb_dim = emb_dim
        self.n_proj = n_proj

    def forward(self, x):
        bs, seq_len, emb_dim = x.size()
        z = F.normalize(x, p=2.0, dim=-1) @ self.projection_mat
        z = torch.bucketize(z, self.grid).transpose(0, -1)
        z = (z + self.pos_offset).transpose(0, -1).contiguous()
        return self.emb(z.view(-1, self.n_proj)).reshape(bs, seq_len, self.emb_dim)


class CosineLinear(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn((out_dim, inp_dim)) / math.sqrt(inp_dim))

    def forward(self, x):
        return F.linear(F.normalize(x, p=2.0, dim=-1), F.normalize(self.weight, p=2.0, dim=-1))


class LearnableCosineVectorEmbedding(nn.Module):
    """
    Learnable LSH based vector indexer for highly non-linear ops
    """

    def __init__(self,
                 inp_dim: int,
                 emb_dim: int,
                 n_proj: int = 16,
                 num_bins: int = 20,
                 sigma_inflation_factor: float = 1.0,
                 top_k: Optional[int] = None):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_proj = n_proj
        self.num_bins = num_bins
        self.top_k = None if top_k is None else min(top_k, num_bins)
        self.sigma2 = (sigma_inflation_factor * 2.0 / num_bins) ** 2
        self.proj = CosineLinear(inp_dim, n_proj)
        self.mean = nn.Parameter(2 * torch.rand((1, 1, n_proj, num_bins)) - 1)
        self.emb = nn.Linear(self.n_proj * self.num_bins, emb_dim, bias=False)

    def forward(self, x):
        bs, seq_len, _ = x.shape
        z = self.gaussian_kernel(self.proj(x))
        return self.emb(z.view(bs, seq_len, self.n_proj * self.num_bins))

    def gaussian_kernel(self, x):
        diff = x.unsqueeze(-1) - self.mean
        act = torch.exp(-0.5 * diff * diff / self.sigma2)
        out = act.clone()
        if self.top_k is not None:
            # useful for compression
            top_k, _ = torch.topk(act, k=self.top_k, dim=-1, largest=True, sorted=True)
            out[act < top_k[..., -1].unsqueeze(-1)] = 0.0
        return F.normalize(out, p=2.0, dim=-1)


class CompositeCosineVectorEmbedding(nn.Module):
    """
    LSH based vector indexer for highly non-linear ops
    """

    def __init__(self,
                 inp_dim: int,
                 emb_dim: int,
                 num_bins: Tuple[int, ...],
                 n_proj: int,
                 learnable: bool):
        super().__init__()
        lsh_clazz = LearnableCosineVectorEmbedding if learnable else CosineVectorEmbedding
        self.emb = nn.ModuleList([
            lsh_clazz(inp_dim=inp_dim, emb_dim=emb_dim, n_proj=n_proj, num_bins=k) for k in num_bins
        ])

    def forward(self, x):
        x = x.unsqueeze(1)
        result = torch.empty((0,), device=x.device)
        for k, mod in enumerate(self.emb):
            if k == 0:
                result = mod(x)
            else:
                result = result + mod(x)
        return result.squeeze(1)


class MLP(nn.Module):
    """
    Multilayer perceptron
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            gate_sizes: Optional[Tuple[int, ...]] = None,
            bias: bool = True,
            add_residual_connection: bool = False
    ):
        super().__init__()
        gate_sizes = gate_sizes if gate_sizes is not None else []
        previous_shape = in_features
        blocks = []
        for shape in gate_sizes:
            blocks.append(nn.Linear(previous_shape, shape, bias=bias))
            blocks.append(nn.GELU(approximate="tanh"))
            previous_shape = shape
        blocks.append(nn.Linear(previous_shape, out_features, bias=bias))
        self.add_residual_connection = add_residual_connection
        self.model = nn.Sequential(*blocks)
        if add_residual_connection:
            self.residual_connector = nn.Linear(in_features, out_features) \
                if in_features != out_features else nn.Identity()
        else:
            self.residual_connector = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.add_residual_connection:
            return self.model(x) + self.residual_connector(x)
        return self.model(x)


class ConvMLP(nn.Module):
    """
    Convolutional Multilayer perceptron
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            kernel_size: Tuple[int, int],
            gate_sizes: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        gate_sizes = gate_sizes if gate_sizes is not None else []
        previous_shape = in_features
        blocks = []
        for shape in gate_sizes:
            blocks.append(nn.Conv2d(previous_shape, shape, kernel_size, padding='same'))
            blocks.append(nn.GELU(approximate="tanh"))
            previous_shape = shape
        blocks.append(nn.Conv2d(previous_shape, out_features, kernel_size, padding='same'))
        self.model = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class _MoEUnit(nn.Module):
    def __init__(self, in_features: int, out_features: int, proj_features: int):
        """
        :param in_features: input number of units
        :param out_features: output number of units
        :param proj_features: number of units in the intermediate projection matrix (used to reduce model size)
        """
        super().__init__()
        self.l1 = nn.Linear(in_features, proj_features)
        self.l2 = nn.Linear(proj_features, out_features)
        self.activation = nn.GELU(approximate="tanh")

    def forward(self, x):
        return self.l2(self.activation(self.l1(x)))


class MoELinear(nn.Module):
    """
    Uses Mixture of Experts

    A parameter efficient version of MLP module that utilizes Mixture of Experts (MoE) to combine the outputs of
    multiple low-dimensional (small values of proj_features) Cross modules
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 proj_features: int,
                 num_experts: int,
                 bias: bool = True,
                 top_k: int = 1,
                 gate_sizes: Optional[Tuple[int, ...]] = None):
        """
        :param in_features: input number of units
        :param out_features: output number of units
        :param proj_features: number of units in the intermediate projection matrix (used to reduce model size)
        :param num_experts: number of experts
        """
        super().__init__()
        self._in_features = in_features
        self._out_features = out_features
        self.expert_gates = MLP(in_features, num_experts, gate_sizes=gate_sizes, bias=bias)
        self.experts = nn.ModuleList([_MoEUnit(in_features, out_features, proj_features) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x):
        # implementation taken partially from https://github.com/dzhulgakov/llama-mistral
        in_shape = x.shape
        out_shape = in_shape[:-1] + (self._out_features,)
        x = x.view(-1, self._in_features)
        expert_gate_values = (self.expert_gates(x) / math.sqrt(self._in_features)).softmax(dim=-1)

        # Note expert weights won't add to one and this enables gradient flow even when top_k = 1
        # See https://arxiv.org/abs/2101.03961 for the special case when top_k = 1
        expert_weights, expert_indices = torch.topk(expert_gate_values, self.top_k, dim=-1)
        flat_expert_indices = expert_indices.view(-1)
        x = x.repeat_interleave(self.top_k, dim=0)
        y = torch.empty(*(x.shape[:-1] + (self._out_features,)), device=x.device)
        for i, expert in enumerate(self.experts):
            y[flat_expert_indices == i] = expert(x[flat_expert_indices == i])
        y = (y.view(*expert_weights.shape, -1) * expert_weights.unsqueeze(-1)).sum(dim=1)
        return y.view(*out_shape)


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class LayerNormND(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, shape, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(*shape))
        self.bias = nn.Parameter(torch.zeros(*shape)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class SelfAttention(nn.Module):
    def __init__(self, config: SelfAttentionConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        raise ValueError('Not implemented in base class')

    @classmethod
    def from_config(cls, config: SelfAttentionConfig):
        if config.attn_type == SelfAttentionType.MULTI_HEAD:
            return MultiHeadAttention(config)
        elif config.attn_type == SelfAttentionType.MULTI_QUERY:
            return MultiQueryAttention(config)
        raise ValueError('unknown self attn implementation!')


class MultiQueryAttention(SelfAttention):
    def __init__(self, config: SelfAttentionConfig):
        super().__init__(config)
        # query projections for all heads, but in a batch
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.kv_proj = nn.Linear(config.n_embd, 2 * config.n_embd // config.n_head, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        bs, t, _ = x.size()
        device = x.device

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.q_proj(x)
        k, v = self.kv_proj(x).split(self.n_embd // self.n_head, dim=-1)

        ones = torch.ones((bs, 1, t, 1), device=device)
        k_do = self.attn_dropout(ones)
        q_do = self.attn_dropout(ones)
        v_do = self.attn_dropout(ones)

        q = q_do * rearrange(q, 'b ... t (h e) -> b h ... t e', h=self.n_head)
        k = k_do * rearrange(k, 'b ... t (h e) -> b h ... t e', h=1)
        v = v_do * rearrange(v, 'b ... t (h e) -> b h ... t e', h=1)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # efficient attention using Flash Attention CUDA kernels
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0)
        # y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        y = rearrange(y, 'b h ... t e -> b ... t (h e)')

        # output projection
        y = self.resid_dropout(self.out_proj(y))
        return y


class MultiHeadAttention(SelfAttention):
    def __init__(self, config: SelfAttentionConfig):
        super().__init__(config)
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        device = x.device

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        ones = torch.ones((B, 1, T, 1), device=device)
        k_do = self.attn_dropout(ones)
        q_do = self.attn_dropout(ones)
        v_do = self.attn_dropout(ones)

        k = k_do * k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q_do * q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v_do * v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # efficient attention using Flash Attention CUDA kernels
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class _MLP(nn.Module):
    def __init__(self, n_embd: int, bias: bool, dropout: float, config: MLPConfig):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, int(config.ff_mult * n_embd), bias=bias)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(int(config.ff_mult * n_embd), n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class _MoEMLP(nn.Module):
    def __init__(self, n_embd: int, bias: bool, dropout: float, config: MoEConfig):
        super().__init__()
        self.c_fc = MoELinear(
            n_embd,
            int(config.ff_mult_factor * n_embd),
            proj_features=config.proj_features,
            num_experts=config.num_experts,
            bias=bias,
            top_k=config.top_k,
            gate_sizes=config.gate_sizes,
        )
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = MoELinear(
            int(config.ff_mult_factor * n_embd),
            n_embd,
            proj_features=config.proj_features,
            num_experts=config.num_experts,
            bias=bias,
            top_k=config.top_k,
            gate_sizes=config.gate_sizes,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig, seed: Optional[int] = None, n_cls: int = 0):
        super().__init__()
        self.is_causal = config.is_causal
        self.ln_1 = LayerNorm(config.attn_config.n_embd, bias=config.attn_config.bias)
        self.attn = SelfAttention.from_config(config.attn_config)
        self.ln_2 = LayerNorm(config.attn_config.n_embd, bias=config.attn_config.bias)
        if isinstance(config.rotator_config, MLPConfig):
            self.mlp = _MLP(config.attn_config.n_embd, config.attn_config.bias, config.attn_config.dropout,
                            config.rotator_config)
        elif isinstance(config.rotator_config, MoEConfig):
            self.mlp = _MoEMLP(config.attn_config.n_embd, config.attn_config.bias, config.attn_config.dropout,
                               config.rotator_config)
        else:
            raise ValueError('Unknown rotator config')
        self.is_cross_attn = config.is_cross_attn
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.attn_config.n_embd,
            num_heads=config.attn_config.n_head,
            dropout=config.attn_config.dropout,
            batch_first=True,
        ) if config.is_cross_attn else nn.Identity()
        self.ln_3 = LayerNorm(config.attn_config.n_embd, bias=config.attn_config.bias) \
            if config.is_cross_attn else nn.Identity()
        self.is_sparse = config.is_sparse_attn
        if self.is_sparse:
            # why does this help: https://medium.com/@iitmdinesh/sparse-transformers-922e010bbd27
            assert config.max_block_size is not None, 'need to specify max_block_size for sparse attention'
            n_non_zeros = int(config.sparsity_factor * config.max_block_size)
            # full_mask = torch.randperm(config.max_block_size, dtype=torch.long)
            gen = np.random.Generator(np.random.PCG64(seed=seed)) if seed is not None else np.random.default_rng()
            full_mask = torch.cat((torch.arange(0, n_cls),
                                   torch.tensor(
                                       gen.permutation(config.max_block_size - n_cls) + n_cls, dtype=torch.long)
                                   ), dim=0)
            # sort is very important for maintaining causality in attention!!!
            self.register_buffer('input_mask_idx', full_mask[:n_non_zeros].sort().values, persistent=True)
            self.register_buffer('input_mask_not_idx', full_mask[n_non_zeros:].sort().values, persistent=True)
            self.null_connector = nn.Linear(config.attn_config.n_embd,
                                            config.attn_config.n_embd,
                                            bias=config.attn_config.bias)
        else:
            self.null_connector = nn.Identity()

    def forward(self, x_orig: torch.Tensor,
                cross_attn_inputs: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None):
        idx = None
        not_idx = None
        if self.is_sparse:
            T = x_orig.size(1)
            idx = self.input_mask_idx[self.input_mask_idx < T]
            if idx.size(0) <= 1:
                return x_orig + self.null_connector(x_orig)
            not_idx = self.input_mask_not_idx[self.input_mask_not_idx < T]
            x = x_orig[:, idx]
            attn_mask = attn_mask[..., idx, :][..., idx] if attn_mask is not None else None
        else:
            x = x_orig

        if self.is_causal:
            L = x.size(-2)
            device = x.device
            attn_mask_causal = torch.ones((L, L), device=device, dtype=torch.bool).tril(diagonal=0)
            attn_mask_causal = repeat(
                attn_mask_causal.float().masked_fill(~attn_mask_causal, -float('inf')),
                's l -> b h s l', b=1, h=1
            )
        else:
            attn_mask_causal = None
        if attn_mask_causal is not None:
            if attn_mask is None:
                attn_mask = attn_mask_causal
            else:
                attn_mask = attn_mask + attn_mask_causal
        x = x + self.attn(self.ln_1(x), mask=attn_mask)
        if cross_attn_inputs is not None:
            if not self.is_cross_attn:
                raise ValueError('Model not configured for cross attn inputs!!!')
            x = x + self.cross_attn(
                query=self.ln_3(x),
                key=cross_attn_inputs,
                value=cross_attn_inputs,
                need_weights=False,
            )[0]
        x = x + self.mlp(self.ln_2(x))
        if not (torch.jit.is_scripting() or torch.jit.is_tracing()):
            x = normalize_gradients(x)
        if not self.is_sparse:
            return x
        x_final = torch.zeros_like(x_orig)
        x_final[:, idx] = x
        x_final[:, not_idx] = x_orig[:, not_idx] + self.null_connector(x_orig[:, not_idx])
        return x_final


class AdvancedPositionalBiasMLP(nn.Module):
    """Advanced positional bias using MLP per position"""
    def __init__(self,
                 context_width: int,
                 in_features: int,
                 out_features: int,
                 gate_sizes: Optional[Tuple[int, ...]] = None,
                 add_residual_connection: bool = True,
                 ):
        super().__init__()
        self.models = nn.ModuleList([
            MLP(
                in_features,
                out_features,
                gate_sizes,
                bias=True,
                add_residual_connection=add_residual_connection
            ) for _ in range(context_width)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([mod(y) for mod, y in zip(self.models, x.unbind(dim=-2))], dim=-2)
