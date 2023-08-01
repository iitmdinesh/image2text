from typing import List, Optional, Union, Tuple
from pydantic import BaseModel
from enum import Enum


# LORA makes sense when we have a pretrained model that
# we want to fine-tune in a parameter efficient manner.
# the above is enforced *silently* in code
class LoraSpec(BaseModel):
    r: int = 16
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    target_modules: Optional[List[str]] = None
    force_enable_update_modules: Optional[List[str]] = None


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


class EncoderConfig(BaseModel):
    n_cls: int
    lora_spec: Optional[LoraSpec] = None


class VisionTransformerEncoderConfig(EncoderConfig):
    transformer_config: TransformerConfig
    enable_gradient_checkpointing: bool = False
    input: ImageInputSpec
    n_layer: int = 12
    num_patches: int
    n_channels: int
    feature_extractor_gate_sizes: Optional[Tuple[int, ...]] = None
    feature_extractor_kernel_size: Tuple[int, int] = (4, 4)


class PretrainedViTConfig(EncoderConfig):
    refine_base_model: bool = True
    n_embd_out_vit: int
    gate_sizes: Optional[Tuple[int, ...]] = None


class ModelType(Enum):
    GPT2 = "gpt2"
    GPT2_MEDIUM = "gpt2-medium"
    GPT2_LARGE = "gpt2-large"
    GPT2_XL = "gpt2-xl"


class DecoderConfig(BaseModel):
    lora_spec: Optional[LoraSpec] = None
    enable_gradient_checkpointing: bool = False
    vocab_size: int


class TransformerDecoderConfig(DecoderConfig):
    transformer_config: TransformerConfig
    use_advanced_pos_emb: bool = False
    advanced_pos_emb_gate_sizes: Optional[Tuple[int, ...]] = None
    pretrained_model: Optional[ModelType] = None
    n_layer: int
    skip_alternate_cross_attn: bool = True
    block_size: int


class HuggingfaceDecoderConfig(DecoderConfig):
    use_cross_attn: bool
    model_str: str
    extra_tokens: int
    load_in_4bit: bool
    prepare_for_kbit_training: bool


class VisionEncoderDecoderConfig(BaseModel):
    vision_encoder_config: Union[VisionTransformerEncoderConfig, PretrainedViTConfig]
    decoder_config: Union[TransformerDecoderConfig, HuggingfaceDecoderConfig]
    loose_match_decoder_state_dict: bool = False
    chkpt_path: Optional[str] = None
    use_cross_attn: bool = False
    use_soft_prompting: bool = True
    no_repeat_n_grams: Tuple[int, ...] = (2, 3, 4, 5)
