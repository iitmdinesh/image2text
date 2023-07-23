from typing import Optional, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from models.functions import normalize_gradients
from configs.models import (
    SelfAttentionConfig,
    TransformerConfig,
    MLPConfig,
    MoEConfig,
    SelfAttentionType,
)


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
            self.residual_connector = nn.Linear(in_features, out_features)
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
                 top_k: Optional[int] = None,
                 gate_sizes: Optional[Tuple[int, ...]] = None):
        """
        :param in_features: input number of units
        :param out_features: output number of units
        :param proj_features: number of units in the intermediate projection matrix (used to reduce model size)
        :param num_experts: number of experts (typical values in the range 2 - 4)
        """
        super().__init__()
        self._in_features = in_features
        self._out_features = out_features
        self.expert_gates = MLP(in_features, num_experts, gate_sizes=gate_sizes, bias=bias)
        self.experts = nn.ModuleList([_MoEUnit(in_features, out_features, proj_features) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x):
        expert_gate_values = (self.expert_gates(x) / math.sqrt(self._in_features))
        if self.top_k is not None:
            v, _ = torch.topk(expert_gate_values, min(self.top_k, expert_gate_values.size(-1)), dim=-1, sorted=True)
            # use the smallest of the top_k to trim
            expert_gate_values[expert_gate_values < v[..., [-1]]] = -float('inf')
        expert_gate_values = expert_gate_values.softmax(dim=-1)
        expert_outputs = []
        for mod in self.experts:
            expert_outputs.append(mod(x))
        expert_outputs = torch.stack(expert_outputs, dim=-2)
        orig_shape = expert_gate_values.size()
        gates = expert_gate_values.unsqueeze(-1).expand(*orig_shape, self._out_features)
        return (expert_outputs * gates).sum(dim=-2)


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
                attn_mask_causal.masked_fill(~attn_mask_causal, -float('inf')).float(),
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


class AdvancedPositionalBias(nn.Module):
    """
    Advanced postional bias using linear layer per position
    """
    def __init__(self, context_width: int, emb_dim: int, emb_dim_out: Optional[int] = None):
        super().__init__()
        emb_dim_out = emb_dim_out if emb_dim_out is not None else emb_dim
        self.models = nn.ModuleList([nn.Linear(emb_dim, emb_dim_out) for _ in range(context_width)])

    def forward(self, x: torch.Tensor):
        return torch.cat([mod(y).unsqueeze(-2) for mod, y in zip(self.models, x.unbind(dim=-2))], dim=-2)


class AdvancedPositionalBiasMLP(nn.Module):
    """Further extension of AdvancedPositionalBias using an MLP instead of linear layer"""
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
        return torch.cat([mod(y).unsqueeze(-2) for mod, y in zip(self.models, x.unbind(dim=-2))], dim=-2)
