import torch
import torch.nn as nn
from smart_open import open as smart_open
from copy import deepcopy

from configs.models import TransformerConfig


def update_state_dict_from_partial_checkpoint(model: nn.Module, chkpt_path: str, map_location=None) -> nn.Module:
    full_state_dict = model.state_dict()
    possibly_partial_state_dict = torch.load(smart_open(chkpt_path, mode='rb'), map_location=map_location)
    full_state_dict.update(possibly_partial_state_dict)
    model.load_state_dict(full_state_dict)
    return model


def mutate_transformer_config(config: TransformerConfig, depth: int, skip_alternate_cross_attn: bool):
    if config.is_cross_attn and skip_alternate_cross_attn and depth % 2:
        config = deepcopy(config)
        config.is_cross_attn = False
        return config
    return config
