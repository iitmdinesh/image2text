import abc
from typing import Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from configs.models import VisionTransformerEncoderConfig, PretrainedViTConfig
from models.layers import (
    ConvMLP,
    TransformerBlock,
    LayerNormND,
    LayerNorm,
    AdvancedPositionalBiasMLP,
    CompositeCosineVectorEmbedding,
    PeerLookup,
)
from models.utils import get_lora_model
from peft import TaskType
from torchvision.models import vit_b_16, ViT_B_16_Weights


class Encoder(nn.Module, abc.ABC):
    """Base class for encoder models"""
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        raise ValueError('Not implemented in base class!')

    @classmethod
    def from_config(cls, config: Union[VisionTransformerEncoderConfig, PretrainedViTConfig]):
        if isinstance(config, PretrainedViTConfig):
            model, pretrained = PretrainedViT(config), True
        elif isinstance(config, VisionTransformerEncoderConfig):
            model, pretrained = VisionTransformerEncoder(config), False
        else:
            raise ValueError('Unknown config')
        if pretrained:
            lora_spec = config.lora_spec
            return get_lora_model(model, TaskType.FEATURE_EXTRACTION, lora_spec)
        return model

    @property
    def num_outputs(self):
        raise ValueError('Not implemented in base class')

    @property
    def output_embed_dim(self):
        raise ValueError('Not implemented in base class')


class PretrainedViT(Encoder):
    def __init__(self, config: PretrainedViTConfig):
        super().__init__(config)
        weights = ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
        model = vit_b_16(weights=weights)
        model.heads = nn.Identity()
        self.out_dim = config.n_embd_out_vit
        self.n_cls = config.n_cls
        self.use_peer = config.peer_config is not None
        self.use_lsh = not self.use_peer and config.lsh_config is not None
        self.proj = AdvancedPositionalBiasMLP(context_width=config.n_cls,
                                              in_features=768,
                                              out_features=config.n_embd_out_vit,
                                              gate_sizes=config.gate_sizes,
                                              add_residual_connection=True) \
            if not (self.use_lsh or self.use_peer) else nn.Identity()
        self.model = model
        self.refine = config.refine_base_model if not self.use_lsh else False
        if self.use_peer:
            assert config.peer_config is not None
            self.peer = PeerLookup(
                768,
                config.n_embd_out_vit,
                config.peer_config.num_units_sqrt ** 2,
                config.peer_config.topk,
                config.peer_config.nhead,
                config.peer_config.query_dim,
            )
            self.peer_proj_wt = nn.Parameter(
                torch.randn((768, 768, self.n_cls)) / math.sqrt(768),
                requires_grad=True,
            )
        else:
            self.peer = nn.Identity()
            self.peer_proj_wt = nn.Parameter(
                torch.zeros((1,)),
                requires_grad=False,
            )
        if self.use_lsh:
            assert config.lsh_config is not None
            self.lsh_emb = nn.ModuleList([
                CompositeCosineVectorEmbedding(
                    768,
                    config.n_embd_out_vit,
                    config.lsh_config.num_bins,
                    config.lsh_config.num_proj,
                    config.lsh_config.learnable,
                ) for _ in range(self.n_cls)
            ])
        else:
            self.lsh_emb = nn.ModuleList([nn.Identity()])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.model(images)
        if not self.refine:
            x = x.detach()
        if self.use_peer:
            return self.peer(torch.einsum("bd,des->bse", x, self.peer_proj_wt))
        elif self.use_lsh:
            return torch.stack([mod(x) for mod in self.lsh_emb], dim=1)
        x = F.normalize(x, p=2.0, dim=-1)
        return F.normalize(self.proj(x.unsqueeze(-2).expand(-1, self.n_cls, -1).contiguous()), p=2.0, dim=-1)

    @property
    def num_outputs(self):
        return self.n_cls

    @property
    def output_embed_dim(self):
        return self.out_dim


class VisionTransformerEncoder(Encoder):
    def __init__(self, config: VisionTransformerEncoderConfig):
        super().__init__(config)
        self.n_patches = n_patches = config.num_patches
        assert config.input.width % n_patches == 0
        assert config.input.height % n_patches == 0
        self.patch_size = (config.input.width // n_patches, config.input.height // n_patches)
        in_features = config.input.n_channels
        out_features = config.n_channels
        self.feature_extractor = ConvMLP(
            in_features,
            out_features,
            config.feature_extractor_kernel_size,
            config.feature_extractor_gate_sizes,
        )
        self.input_d = out_features * self.patch_size[0] * self.patch_size[1]
        self.out_dim = config.transformer_config.attn_config.n_embd
        self.projector = nn.Linear(self.input_d,
                                   self.out_dim,
                                   bias=config.transformer_config.attn_config.bias)
        self.ln_input = LayerNormND((n_patches ** 2, self.out_dim),
                                    config.transformer_config.attn_config.bias)
        self.transformer = nn.ModuleDict(dict(
            wpe=nn.Embedding(n_patches ** 2, self.out_dim),
            drop=nn.Dropout(config.transformer_config.attn_config.dropout),
            h=nn.ModuleList([TransformerBlock(config.transformer_config, seed=depth)
                             for depth in range(config.n_layer)]),
            ln_f=LayerNorm(self.out_dim, bias=config.transformer_config.attn_config.bias),
        ))
        self.cls_token = nn.Parameter(torch.randn(1, config.n_cls, self.out_dim) / math.sqrt(self.out_dim))
        self.n_cls = config.n_cls
        self.enable_gradient_checkpointing = config.enable_gradient_checkpointing

    def forward(self, images: torch.Tensor):
        images = self.feature_extractor(images)
        n, c, w, h = images.shape
        x = self.ln_input(self.projector(images.reshape(n, self.n_patches ** 2, self.input_d)))
        pos_emb = self.transformer.wpe(
            torch.arange(0, self.n_patches ** 2, device=x.device).unsqueeze(0)).expand(x.size(0), -1, -1)
        y = x + pos_emb
        x = torch.cat((self.cls_token.expand(x.size(0), self.n_cls, self.out_dim), self.ln_input(y)), dim=1)
        x = self.transformer.drop(x)
        jit_op = torch.jit.is_scripting() or torch.jit.is_tracing()
        for block in self.transformer.h:
            if self.enable_gradient_checkpointing and self.training and not jit_op:
                x = self.gradient_checkpointed_transformer_block(block, x)
            else:
                x = block(x)
        return self.transformer.ln_f(x[:, :self.n_cls].contiguous())

    # See https://pytorch.org/docs/stable/generated/torch.jit.ignore.html
    @torch.jit.unused
    def gradient_checkpointed_transformer_block(
            self,
            mod: nn.Module,
            x: torch.Tensor,
    ) -> torch.Tensor:
        return torch.utils.checkpoint.checkpoint(mod, x)

    @property
    def num_outputs(self):
        return self.n_cls

    @property
    def output_embed_dim(self):
        return self.out_dim
