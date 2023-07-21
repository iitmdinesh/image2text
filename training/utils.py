from typing import Optional, Union, List

import torch
import torch.nn as nn
import torch.utils.data

from tqdm.auto import tqdm
from accelerate import Accelerator
from smart_open import open as smart_open
from contextlib import nullcontext

from training.wrapper import ModelTrainerWrapper


def normalize_label(input_ids, attn_mask, ignore_index):
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


def train_loop(model_wrapper: Union[nn.parallel.DistributedDataParallel,
                                    ModelTrainerWrapper],
               optimizer: torch.optim.Optimizer,
               train_dl: torch.utils.data.DataLoader,
               epoch: int,
               num_steps: Optional[int],
               accelerator: Accelerator,
               disable_flash: bool = False,
               ignore_index: int = -100,
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
    with tqdm(train_dl, unit="batch", disable=not accelerator.is_local_main_process) as tepoch:
        for step, batch in enumerate(tepoch):
            if num_steps is not None and num_steps == step:
                break
            tepoch.set_description(f'Epoch: {epoch}')
            images, labels_0, labels_1, labels_2, labels_3, labels_4 = unpack_batch(batch,
                                                                                    ignore_index=ignore_index)
            images, labels_0, labels_1, labels_2, labels_3, labels_4 = \
                images.to(device), labels_0.to(device), labels_1.to(device), labels_2.to(device), labels_3.to(device), \
                labels_4.to(device)
            ctx = torch.backends.cuda.sdp_kernel(enable_flash=False) if disable_flash else nullcontext()
            with ctx:
                with accelerator.autocast():
                    with accelerator.accumulate(model_wrapper):
                        for labels in [labels_0, labels_1, labels_2, labels_3, labels_4]:
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


def val_loop(model_wrapper: Union[nn.parallel.DistributedDataParallel,
                                  ModelTrainerWrapper],
             val_dl: torch.utils.data.DataLoader,
             epoch: int,
             num_val_steps: Optional[int],
             accelerator: Accelerator,
             disable_flash: bool = False,
             ignore_index: int = -100,
             ):
    model_wrapper.eval()
    device = accelerator.device
    loss_all = []
    metrics_all = {}
    num_steps = len(val_dl) if num_val_steps is None else num_val_steps
    if isinstance(model_wrapper, nn.parallel.DistributedDataParallel):
        val_step = model_wrapper.module.val_step
    else:
        val_step = model_wrapper.val_step
    with tqdm(val_dl, unit="batch", disable=not accelerator.is_local_main_process) as tepoch:
        for step, batch in enumerate(tepoch):
            if num_val_steps is not None and num_val_steps == step:
                break
            tepoch.set_description(f'Epoch: {epoch}')
            images, labels_0, labels_1, labels_2, labels_3, labels_4 = unpack_batch(batch,
                                                                                    ignore_index=ignore_index)
            images, labels_0, labels_1, labels_2, labels_3, labels_4 = \
                images.to(device), labels_0.to(device), labels_1.to(device), labels_2.to(device), labels_3.to(device), \
                labels_4.to(device)
            ctx = torch.backends.cuda.sdp_kernel(enable_flash=False) if disable_flash else nullcontext()
            with ctx:
                with torch.no_grad():
                    with accelerator.no_sync(model_wrapper):
                        with accelerator.autocast():
                            for labels in [labels_0, labels_1, labels_2, labels_3, labels_4]:
                                loss, metrics = val_step(images, labels)
                                loss_all.append(accelerator.gather(loss))
                                metrics = accelerator.gather(metrics)
                                for k in metrics:
                                    metrics_all[k] = metrics_all.setdefault(k, 0.0) + \
                                                     metrics[k].mean().cpu().item() / (5 * num_steps)
            tepoch.set_postfix(**{k: v.cpu().item() for k, v in metrics.items()})
    loss_all = torch.stack(loss_all, dim=0)
    loss = loss_all.mean().item()
    return loss, metrics_all
