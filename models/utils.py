from typing import Optional, List

import torch
import torch.nn as nn
from peft import (
    LoraConfig,
    TaskType,
)
from peft.tuners import LoraModel

from smart_open import open as smart_open
from copy import deepcopy

from configs.models import TransformerConfig, LoraSpec
import fnmatch


class PatternMatcher:
    def __init__(self, patterns: Optional[List[str]]):
        self.patterns = patterns

    def match(self, candidate):
        if self.patterns is None or len(self.patterns) == 0:
            return True
        for pattern in self.patterns:
            if fnmatch.fnmatch(candidate, pattern):
                return True
        return False


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


def get_lora_model(model: nn.Module, task_type: TaskType, lora_spec: Optional[LoraSpec]) -> nn.Module:
    if lora_spec is None:
        return model
    lora_config = LoraConfig(
        r=lora_spec.r,
        lora_alpha=lora_spec.lora_alpha,
        lora_dropout=lora_spec.lora_dropout,
        task_type=task_type,
        inference_mode=False,
        target_modules=lora_spec.target_modules,
    )
    # hack to use peft-lora for any forward signature (not just huggingface/transformers models)
    model = LoraModel(model, {'default': lora_config}, adapter_name='default')

    if lora_spec.force_enable_update_modules is not None:
        matcher = PatternMatcher(patterns=lora_spec.force_enable_update_modules)
        for n, p in model.named_parameters():
            if matcher.match(n):
                p.requires_grad = True
    return model
