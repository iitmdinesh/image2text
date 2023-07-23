from typing import Optional, List
from pydantic import BaseModel
from configs.models import VisionEncoderDecoderConfig


class TrainerWrapperConfig(BaseModel):
    moco_momentum: Optional[float] = None  # 0.995
    moco_alpha: Optional[float] = None  # 0.4
    training_temperature: float = 1.0
    weight_fn: str = 'constant'
    actual_vocab_size: int
    mask_fraction: float = 0.15
    random_mask_fraction: float = 0.2
    eos_token_weight: Optional[float] = None


class OptimizerConfig(BaseModel):
    lr: float = 6e-4
    weight_decay: float = 0.0


class TrainingConfig(BaseModel):
    model: VisionEncoderDecoderConfig
    disable_flash: bool = False
    ignore_index: int = -100
    batch_size: int = 8
    dataloader_buffer_size: int = 5
    shuffle: bool = True
    gradient_accumulation_steps: int = 4
    epochs: int = 6
    num_steps: Optional[int] = None
    num_val_steps: Optional[int] = None
    precision: str = 'fp16'
    add_eos: bool = True
    tokenizer_str: str
    reset_moco_after_k_epochs: Optional[List[int]] = None
    trainer: TrainerWrapperConfig
    optimizer: OptimizerConfig
