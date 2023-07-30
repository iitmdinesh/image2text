from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.models import VisionEncoderDecoderConfig
from configs.trainer import TrainerWrapperConfig
from models.vision_encoder_decoder import VisionEncoderDecoder
from transformers import PreTrainedTokenizer


class ModelTrainerWrapper(nn.Module):
    """
    Trains the model on three tasks
        (i) Causal LM on summary of embeddings conditioned on input embedding sequence
        (ii) Causal Masked LM on summary of embeddings conditioned on input embedding sequence
    """

    def __init__(
            self,
            model_config: VisionEncoderDecoderConfig,
            tokenizer: PreTrainedTokenizer,
            trainer_config: TrainerWrapperConfig,
            ignore_index: int = -100,
    ):
        super().__init__()
        self.model = VisionEncoderDecoder(config=model_config)
        self.model_m = VisionEncoderDecoder(config=model_config) \
            if (trainer_config.moco_momentum is not None and trainer_config.moco_alpha is not None) else None
        self.is_momentum = trainer_config.moco_momentum is not None and trainer_config.moco_alpha is not None
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.temperature = trainer_config.training_temperature
        self.contrastive_temperature = trainer_config.training_contrastive_temperature
        self.weight_fn = trainer_config.weight_fn
        self.mask_fraction = trainer_config.mask_fraction
        self.random_mask_fraction = trainer_config.random_mask_fraction
        self.eos_token_weight = trainer_config.eos_token_weight
        self.momentum = trainer_config.moco_momentum
        self.alpha = trainer_config.moco_alpha
        self.add_contrastive_loss = trainer_config.add_contrastive_loss
        self._model_pairs = [[self.model, self.model_m]]
        self.copy_momentum_params()

    @torch.no_grad()
    def copy_momentum_params(self):
        if not self.is_momentum:
            return
        # need to correctly copy buffers as well
        self.model_m.load_state_dict(self.model.state_dict())

    @torch.no_grad()
    def _momentum_update(self):
        if not self.is_momentum:
            return
        for model_pair in self._model_pairs:
            for (n1, param), (n2, param_m) in zip(model_pair[0].named_parameters(), model_pair[1].named_parameters()):
                assert n1 == n2
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    def target_emb(self, input_ids):
        return self.model.decoder.get_inputs_embeds(input_ids)

    def forward(self, images, input_ids, attn_msk=None) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.model(images=images, ids=input_ids, attn_msk=attn_msk)
        return output.logits, output.hidden_state

    @torch.no_grad()
    def forward_m(self, images, input_ids, attn_msk=None) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.model_m(images=images, ids=input_ids, attn_msk=attn_msk)
        return output.logits, output.hidden_state

    def train_step(self, images, labels):
        return self._train_or_val_step_helper(images, labels, True)

    def val_step(self, images, labels):
        return self._train_or_val_step_helper(images, labels, False)

    def get_weights(self, labels):
        device = labels.device
        if self.weight_fn == 'constant':
            weights = torch.ones_like(labels, dtype=torch.float)
        elif self.weight_fn == 'inverse_sqrt_position':
            batch_size = labels.size(0)
            context_length = labels.size(1)
            weights = 1.0 / torch.sqrt(torch.arange(1, context_length + 1, dtype=torch.float, device=device).unsqueeze(
                0).expand(batch_size, -1))
        else:
            raise ValueError(f'unknown weight_fn: {self.weight_fn}')

        if self.eos_token_weight is not None:
            # EOS token prediction is important
            weights[labels == self.tokenizer.eos_token_id] = self.eos_token_weight
        weights[labels == self.ignore_index] = 0.0
        return (weights / (1e-3 + weights.sum(dim=-1, keepdim=True))) / weights.size(0)

    def compute_contrastive_loss(self, hidden_state, labels):
        device = hidden_state.device
        labels = labels[..., :hidden_state.size(-2)].contiguous()
        if hidden_state.size(-2) > labels.size(-1):
            hidden_state = hidden_state[..., :labels.size(-1), :].contiguous()
        else:
            hidden_state = hidden_state.contiguous()
        weights = self.get_weights(labels)
        attn_mask = labels != self.ignore_index
        hidden_target = self.target_emb(torch.where(attn_mask, labels, torch.zeros_like(labels)))

        predictions = \
            hidden_state.view(-1, hidden_state.size(-1)) @ hidden_target.view(-1, hidden_target.size(-1)).T
        predictions = torch.where(attn_mask.view(1, -1),
                                  predictions,
                                  -float('inf') * torch.ones_like(predictions))
        targets = torch.arange(0, predictions.size(0), device=device, dtype=torch.long)
        losses = F.cross_entropy(predictions / self.contrastive_temperature, targets, reduction='none')
        # infs are due to empty rows. these have zero weights anyway. so filter them out
        losses = torch.where(losses.isinf(), torch.zeros_like(losses), losses)
        return (losses.view(-1) * weights.view(-1)).sum()

    def compute_lm_loss(self, lm_logits, labels, lm_logits_moco=None):
        # no need to shift labels because input has already been padded
        labels = labels[..., :lm_logits.size(-2)].contiguous()
        if lm_logits.size(-2) > labels.size(-1):
            lm_logits = lm_logits[..., :labels.size(-1), :].contiguous()
            if lm_logits_moco is not None:
                lm_logits_moco = lm_logits_moco[..., :labels.size(-1), :].contiguous()
        else:
            lm_logits = lm_logits.contiguous()
            if lm_logits_moco is not None:
                lm_logits_moco = lm_logits_moco.contiguous()

        weights = self.get_weights(labels)

        if lm_logits_moco is not None:
            num_classes = lm_logits.size(-1)
            targets = F.one_hot(
                torch.where(labels == self.ignore_index, num_classes * torch.ones_like(labels),
                            labels),
                num_classes + 1
            )[..., :-1]
            targets_smoothed = self.alpha * F.softmax(lm_logits_moco / self.temperature, dim=-1) + \
                (1 - self.alpha) * targets
            return -(torch.sum(F.log_softmax(lm_logits / self.temperature, dim=-1) * targets_smoothed, dim=-1) *
                     weights).sum()

        # Flatten the tokens
        losses = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)) / self.temperature,
                                 labels.view(-1),
                                 ignore_index=self.ignore_index,
                                 reduction='none')
        return (losses.view(-1) * weights.view(-1)).sum()

    def _train_or_val_step_helper(self, images, labels, is_train: bool):
        input_ids = torch.where(
            labels != self.ignore_index,
            labels,
            self.tokenizer.eos_token_id * torch.ones_like(labels)
        )
        attn_msk = labels != self.ignore_index

        if is_train and self.mask_fraction > 0:
            mask = self.tokenizer.mask_token_id * torch.ones_like(input_ids)
            # typically 20% random corruption of mask
            corrupted_mask = torch.where(
                torch.rand_like(input_ids, dtype=torch.float) <= self.random_mask_fraction,
                torch.randint_like(input_ids, low=0, high=self.tokenizer.vocab_size),
                mask
            )

            # typically 15% random masking
            corrupted_inputs = torch.where(
                torch.rand_like(input_ids, dtype=torch.float) <= self.mask_fraction,
                corrupted_mask,
                input_ids
            )
            # important to reestablish norms (even though they don't matter due to
            # attn_mask and labels)
            corrupted_inputs = torch.where(
                labels != self.ignore_index,
                corrupted_inputs,
                self.tokenizer.eos_token_id * torch.ones_like(labels)
            )
        else:
            # don't mask validation data
            corrupted_inputs = input_ids

        bs = corrupted_inputs.size(0)
        sl = corrupted_inputs.size(1)
        corrupted_inputs = torch.cat((
            self.tokenizer.bos_token_id * torch.ones((bs, 1), device=corrupted_inputs.device, dtype=torch.long),
            corrupted_inputs,
        ), dim=1)[:, :sl]
        attn_msk = torch.cat((
            torch.ones((bs, 1), device=attn_msk.device, dtype=torch.bool),
            attn_msk,
        ), dim=1)[:, :sl]

        step = 'train' if is_train else 'val'
        lm_logits, hidden_state = self(images, corrupted_inputs, attn_msk)
        if self.is_momentum and is_train:
            lm_logits_moco, _ = self.forward_m(images, corrupted_inputs, attn_msk)
        else:
            lm_logits_moco = None
        loss = self.compute_lm_loss(lm_logits, labels, lm_logits_moco=lm_logits_moco)
        metrics = {f'{step}_loss_lm': loss.detach()}
        if self.add_contrastive_loss:
            loss_contrastive = self.compute_contrastive_loss(hidden_state, labels)
            metrics[f'{step}_loss_contrastive'] = loss_contrastive.detach()
            loss = loss + loss_contrastive

        if is_train:
            self._momentum_update()

        return loss, metrics
