from typing import Optional, Union, List, Iterator, Tuple

import torch
import torch.nn as nn
import torch.utils.data

from tqdm.auto import tqdm
from accelerate import Accelerator
from smart_open import open as smart_open
from contextlib import nullcontext

from training.wrapper import ModelTrainerWrapper
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


def normalize_label(input_ids, attn_mask, ignore_index):
    to_attd = attn_mask.sum(dim=-1).clamp(0, attn_mask.size(-1) - 1).unsqueeze(-1)
    linear = torch.arange(0, attn_mask.size(-1), device=to_attd.device).unsqueeze(0)
    attn_mask = linear <= to_attd
    return torch.where(attn_mask.bool(), input_ids, ignore_index * torch.ones_like(input_ids))


def unpack_batch(batch,
                 ignore_index: int = -100):
    images, input_ids_0, input_ids_1, input_ids_2, input_ids_3, input_ids_4 = \
        batch['image'], batch['input_ids_0'], batch['input_ids_1'], batch['input_ids_2'], batch['input_ids_3'], \
        batch['input_ids_4']
    attn_mask_0, attn_mask_1, attn_mask_2, attn_mask_3, attn_mask_4 = \
        batch['attn_mask_0'], batch['attn_mask_1'], batch['attn_mask_2'], batch['attn_mask_3'], \
        batch['attn_mask_4']
    labels_0 = normalize_label(input_ids_0, attn_mask_0, ignore_index)
    labels_1 = normalize_label(input_ids_1, attn_mask_1, ignore_index)
    labels_2 = normalize_label(input_ids_2, attn_mask_2, ignore_index)
    labels_3 = normalize_label(input_ids_3, attn_mask_3, ignore_index)
    labels_4 = normalize_label(input_ids_4, attn_mask_4, ignore_index)
    return images, labels_0, labels_1, labels_2, labels_3, labels_4


class WrapperDataLoader:
    def __init__(self, dataloader: torch.utils.data.DataLoader, batch_size: int, ignore_idx: int, epochs: int):
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.ignore_idx = ignore_idx
        self.epochs = epochs

    def __len__(self):
        return 5 * len(self.dataloader)

    def __iter__(self):
        for _ in range(self.epochs):
            for batch in self.dataloader:
                images, labels_0, labels_1, labels_2, labels_3, labels_4 = \
                    unpack_batch(batch, ignore_index=self.ignore_idx)
                images = torch.cat([images] * 5, dim=0)
                labels = torch.cat([labels_0, labels_1, labels_2, labels_3, labels_4], dim=0)
                perm = torch.randperm(images.size(0))
                images = images[perm]
                labels = labels[perm]
                yield from zip(torch.split(images, self.batch_size, dim=0),
                               torch.split(labels, self.batch_size, dim=0))


def train_loop(model_wrapper: Union[nn.parallel.DistributedDataParallel,
                                    ModelTrainerWrapper],
               optimizer: torch.optim.Optimizer,
               train_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]],
               epoch: int,
               num_steps: Optional[int],
               accelerator: Accelerator,
               disable_flash: bool = False,
               reset_moco_after_k_epochs: Optional[List[int]] = None,
               logging_callback=None,
               chckpt_fname=None):
    model_wrapper.train()
    if isinstance(model_wrapper, nn.parallel.DistributedDataParallel):
        train_step = model_wrapper.module.train_step
        reset_method = model_wrapper.module.copy_momentum_params
    else:
        train_step = model_wrapper.train_step
        reset_method = model_wrapper.copy_momentum_params
    device = accelerator.device
    stop = False
    num_steps = 100 if num_steps is None else num_steps
    with tqdm(range(num_steps), unit="batch", disable=not accelerator.is_local_main_process) as tepoch:
        for step in tepoch:
            tepoch.set_description(f'Epoch: {epoch}')
            try:
                images, labels = next(train_iter)
            except StopIteration:
                stop = True
                break
            images, labels = images.to(device), labels.to(device)
            ctx = torch.backends.cuda.sdp_kernel(enable_flash=False) if disable_flash else nullcontext()
            with ctx:
                with accelerator.autocast():
                    with accelerator.accumulate(model_wrapper):
                        loss, metrics = train_step(images, labels)
                        accelerator.backward(loss)
                        optimizer.step()
                        optimizer.zero_grad()

            tepoch.set_postfix(**{k: v.cpu().item() for k, v in metrics.items()})
            if accelerator.is_local_main_process:
                if logging_callback is not None:
                    logging_callback({k: v.cpu().item() for k, v in metrics.items()}, batch=step, epoch=epoch)

    if reset_moco_after_k_epochs is not None and (epoch + 1) in reset_moco_after_k_epochs:
        reset_method()
    
    if chckpt_fname is not None:
        accelerator.wait_for_everyone()
        unwrapped_model: nn.Module = accelerator.unwrap_model(model_wrapper).model
        # FIXME: We want to some thing like the following but that wont work in all places (think buffers like pca proj
        #  matrix
        # since we fine tune large models, we can save some space by doing this
        # sd = unwrapped_model.state_dict()
        # to_save = {}
        # for k, p in unwrapped_model.named_parameters():
        #     if p.requires_grad:
        #         to_save[k] = sd[k]
        accelerator.save(unwrapped_model.state_dict(), smart_open(chckpt_fname, mode='wb'))
    return stop


def val_loop(model_wrapper: Union[nn.parallel.DistributedDataParallel,
                                  ModelTrainerWrapper],
             val_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]],
             epoch: int,
             num_val_steps: Optional[int],
             accelerator: Accelerator,
             disable_flash: bool = False,
             ):
    model_wrapper.eval()
    device = accelerator.device
    loss_all = []
    metrics_all = {}
    num_steps = 100 if num_val_steps is None else num_val_steps
    if isinstance(model_wrapper, nn.parallel.DistributedDataParallel):
        val_step = model_wrapper.module.val_step
    else:
        val_step = model_wrapper.val_step
    with tqdm(range(num_steps), unit="batch", disable=not accelerator.is_local_main_process) as tepoch:
        for _ in tepoch:
            tepoch.set_description(f'Epoch: {epoch}')
            images, labels = next(val_iter)
            images, labels = images.to(device), labels.to(device)
            ctx = torch.backends.cuda.sdp_kernel(enable_flash=False) if disable_flash else nullcontext()
            with ctx:
                with torch.no_grad():
                    with accelerator.no_sync(model_wrapper):
                        with accelerator.autocast():

                            loss, metrics = val_step(images, labels)
                            loss_all.append(accelerator.gather(loss))
                            metrics = accelerator.gather(metrics)
                            for k in metrics:
                                metrics_all[k] = metrics_all.setdefault(k, 0.0) + \
                                                 metrics[k].mean().cpu().item() / num_steps
            tepoch.set_postfix(**{k: v.cpu().item() for k, v in metrics.items()})
    loss_all = torch.stack(loss_all, dim=0)
    loss = loss_all.mean().item()
    return loss, metrics_all
