from typing import List, Optional, Union, Tuple
from pydantic import BaseModel
from enum import Enum


class MLPConfig(BaseModel):
    ff_mult: float


class MoEConfig(BaseModel):
    num_experts: int
    proj_features: int
    ff_mult_factor: float
    gate_sizes: Optional[Tuple[int, ...]] = None
    top_k: Optional[int] = None


class SelfAttentionType(Enum):
    MULTI_HEAD: str = 'multi_head'
    MULTI_QUERY: str = 'multi_query'


class SelfAttentionConfig(BaseModel):
    attn_dropout: float = 0.1
    bias: bool = True
    dropout: float = 0.1
    n_head: int = 12
    n_embd: int = 768
    attn_type: SelfAttentionType


class TransformerConfig(BaseModel):
    rotator_config: Union[MoEConfig, MLPConfig]
    is_causal: bool = False
    is_cross_attn: bool = False
    max_block_size: Optional[int] = None
    is_sparse_attn: bool = False
    sparsity_factor: float = 0.5
    attn_config: SelfAttentionConfig


class ImageInputSpec(BaseModel):
    n_channels: int = 3
    width: int
    height: int


class VisionTransformerEncoderConfig(BaseModel):
    transformer_config: TransformerConfig
    enable_gradient_checkpointing: bool = False
    input: ImageInputSpec
    n_layer: int = 12
    n_cls: int = 32
    num_patches: int
    n_channels: int
    feature_extractor_gate_sizes: Optional[Tuple[int, ...]] = None
    feature_extractor_kernel_size: Tuple[int, int] = (4, 4)


class ModelType(Enum):
    GPT2 = "gpt2"
    GPT2_MEDIUM = "gpt2-medium"
    GPT2_LARGE = "gpt2-large"
    GPT2_XL = "gpt2-xl"


class TransformerDecoderConfig(BaseModel):
    pretrained_model: Optional[ModelType] = None
    enable_gradient_checkpointing: bool = False
    n_layer: int = 12
    skip_alternate_cross_attn: bool = True
    block_size: int = 128
    vocab_size: int = 50258  # GPT-2 + one extra token: <MSK>
    transformer_config: TransformerConfig


class LoraSpec(BaseModel):
    enable_lora: bool = True
    r: int = 16
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    target_modules: Optional[List[str]] = None
    force_enable_update_modules: Optional[List[str]] = None


class HuggingfaceDecoderConfig(BaseModel):
    use_cross_attn: bool = True
    vocab_size: int = 50257  # pass this in from the tokenizer
    model_str: str = 'gpt2-medium'
    extra_tokens: int = 1  # one extra token: <MSK>
    load_in_4bit: bool = True
    prepare_for_kbit_training: bool = True
    enable_gradient_checkpointing: bool = True
    lora_spec: LoraSpec


class ViTConfig(BaseModel):
    n_embd_out_vit: int


class VisionEncoderDecoderConfig(BaseModel):
    vision_encoder_config: Union[VisionTransformerEncoderConfig, ViTConfig]
    decoder_config: Union[TransformerDecoderConfig, HuggingfaceDecoderConfig]
    loose_match_decoder_state_dict: bool = False
    chkpt_path: Optional[str] = None
    use_cross_attn: bool = False
    use_soft_prompting: bool = True
    no_repeat_n_grams: Tuple[int, ...] = (2, 3, 4, 5)

