from typing import Optional, List, Tuple
from pydantic import BaseModel
from configs.models import VisionEncoderDecoderConfig


class TrainerWrapperConfig(BaseModel):
    moco_momentum: Optional[float] = None  # 0.995
    moco_alpha: Optional[float] = None  # 0.4
    training_temperature: float = 1.0
    weight_fn: str = 'constant'
    mask_fraction: float = 0.0  # 0.15
    random_mask_fraction: float = 0.0  # 0.2
    eos_token_weight: Optional[float] = None
    add_contrastive_loss: bool = False  # only makes sense when input and output weights are "tied"
    training_contrastive_temperature: float = 1.0


class OptimizerConfig(BaseModel):
    lr: float
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.999)
    target_modules: Optional[List[str]] = None


class TrainingConfig(BaseModel):
    model: VisionEncoderDecoderConfig
    disable_flash: bool = False
    ignore_index: int = -100
    batch_size: int
    dataloader_buffer_size: int = 5
    shuffle: bool = True
    gradient_accumulation_steps: int = 1
    epochs: int = 1
    num_steps: Optional[int] = None
    num_val_steps: Optional[int] = None
    precision: str = 'no'
    tokenizer_str: str
    reset_moco_after_k_epochs: Optional[List[int]] = None
    trainer: TrainerWrapperConfig
    optimizers: List[OptimizerConfig]
    use_snr_optim: bool = False
