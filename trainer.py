from typing import Union

import torch
import torch.nn as nn
import torch.utils.data


import yaml

from configs.trainer import TrainingConfig
from configs.models import ViTConfig
from training.wrapper import ModelTrainerWrapper
from training.utils import train_loop, val_loop, WrapperDataLoader, PatternMatcher

from accelerate import Accelerator
from transformers import AutoTokenizer, PreTrainedTokenizer
from deeplake import load, Dataset
from torchvision import transforms
from torchvision.models import ViT_B_16_Weights


from argparse import ArgumentParser


def eval_model(model_wrapper: Union[nn.parallel.DistributedDataParallel, ModelTrainerWrapper],
               accelerator,
               tokenizer,
               val_iter,
               epoch,
               ignore_index,
               prompt='A',
               num_candidates=2,
               ):
    accelerator.print(f"Model perf at the end of the {epoch}-th epoch")
    accelerator.print("Val:")
    batch = next(val_iter)
    images, labels = batch
    device = accelerator.device
    x = images.to(device)[:1].expand(num_candidates, -1, -1, -1)
    label_ = labels[0].to(device)
    model_wrapper = accelerator.unwrap_model(model_wrapper)
    with accelerator.autocast():
        if prompt is None:
            decoded_ids = torch.empty((x.size(0), 0), device=x.device, dtype=torch.long)
        else:
            decoded_ids = torch.tensor(
                tokenizer(text=prompt).input_ids,
                dtype=torch.long).to(device).unsqueeze(0).expand(x.size(0), -1).contiguous()

        result = model_wrapper.model.generate(images=x,
                                              prompt_ids=decoded_ids,
                                              temperature=1.0,
                                              max_new_tokens=128,
                                              top_k=16)
        result = tokenizer.batch_decode(result)
        reference = tokenizer.batch_decode([label_[label_ != ignore_index]])[0]

    accelerator.print('truth', reference, '\n')
    for gen in result:
        i = gen.find(tokenizer.eos_token)
        gen = gen[:i]
        accelerator.print(gen)


def get_dataloader(tokenizer, batch_size, shuffle, is_vit):
    txforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
        transforms.Normalize(mean=(0.4274, 0.4218, 0.3878), std=(0.2754, 0.2705, 0.2874)),
    ]) if not is_vit else ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1.transforms()
    ds: Dataset = load('hub://activeloop/flickr30k')

    tokenizer.pad_token = tokenizer.eos_token

    def _tok(x):
        return tokenizer(text=x[0], return_tensors='pt',
                         max_length=256,
                         truncation='longest_first',
                         padding='max_length')

    def _transform(x):
        image = x['image'] if not is_vit else torch.tensor(x['image']).permute(2, 0, 1)
        result = {
            'image': txforms(image),
        }
        for k in range(5):
            data = x[f'caption_{k}']
            tokenized = _tok(data)
            result[f'input_ids_{k}'] = tokenized.input_ids.squeeze(0)
            result[f'attn_mask_{k}'] = tokenized.attention_mask.squeeze(0)
        return result

    train_dl = ds.query("SELECT * WHERE ROW_NUMBER() < 27000"). \
        pytorch(batch_size=batch_size, shuffle=shuffle, num_workers=0, transform=_transform,
                buffer_size=256, use_local_cache=True)
    val_dl = ds.query("SELECT * WHERE ROW_NUMBER() >= 27000 "). \
        pytorch(batch_size=batch_size, shuffle=shuffle, num_workers=0, transform=_transform,
                buffer_size=32, use_local_cache=True)
    return train_dl, val_dl


def main(args):
    obj = yaml.safe_load(open(args.config_file, 'r'))
    config: TrainingConfig = TrainingConfig.parse_obj(obj)
    accelerator = Accelerator(
        device_placement=True,
        split_batches=True,
        mixed_precision=config.precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        even_batches=True,
    )

    accelerator.print(config)

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(config.tokenizer_str)
    kwargs = {}
    if tokenizer.eos_token_id is None:
        kwargs['eos_token'] = '<EOS>'
    if tokenizer.mask_token_id is None and config.trainer.mask_fraction > 0:
        kwargs['mask_token'] = '<MSK>'
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_str, **kwargs)

    train_dl, val_dl = get_dataloader(tokenizer,
                                      config.dataloader_buffer_size * config.batch_size,
                                      config.shuffle,
                                      isinstance(config.model.vision_encoder_config, ViTConfig))
    train_dl, val_dl = WrapperDataLoader(train_dl, batch_size=config.batch_size, ignore_idx=config.ignore_index,
                                         epochs=config.epochs), \
                       WrapperDataLoader(val_dl, batch_size=config.batch_size, ignore_idx=config.ignore_index,
                                         epochs=100000)

    model_wrapper = ModelTrainerWrapper(
        model_config=config.model,
        tokenizer=tokenizer,
        trainer_config=config.trainer,
        ignore_index=config.ignore_index,
    ).to(accelerator.device)
    accelerator.print(model_wrapper.model)
    param_groups = []
    for optim_config in config.optimizers:
        if optim_config.target_modules is not None:
            matcher = PatternMatcher(optim_config.target_modules)
            names = ",".join([n for n, p in model_wrapper.named_parameters() if matcher.match(n)])
            params = nn.ParameterList([p for n, p in model_wrapper.named_parameters() if matcher.match(n)])
            accelerator.print(f'Optimizing the following params:\n{names} with lr={optim_config.lr} and '
                              f'weight_decay={optim_config.weight_decay}')
        else:
            # don't add momentum model params to optimizer state as it will bloat memory footprint
            params = nn.ParameterList([p for n, p in model_wrapper.named_parameters() if n.startswith('model.')])
        param_group = {
            'lr': optim_config.lr,
            'weight_decay': optim_config.weight_decay,
            'params': params
        }
        param_groups.append(param_group)

    optimizer = torch.optim.AdamW(
        param_groups,
    )
    model_wrapper, optimizer, train_dl, val_dl = \
        accelerator.prepare(model_wrapper, optimizer, train_dl, val_dl, device_placement=[False, True, True, True])
    train_iter = iter(train_dl)
    val_iter = iter(val_dl)
    for epoch in range(10000):
        stop = train_loop(
            model_wrapper,
            optimizer,
            train_iter,
            epoch,
            config.num_steps,
            accelerator,
            disable_flash=config.disable_flash,
            reset_moco_after_k_epochs=config.reset_moco_after_k_epochs,
            chckpt_fname=args.chkpt_file)
        if stop:
            break
        eval_model(model_wrapper, accelerator, tokenizer, val_iter, epoch, config.ignore_index)
        loss, metrics = val_loop(
            model_wrapper,
            val_iter,
            epoch,
            config.num_val_steps,
            accelerator,
            disable_flash=config.disable_flash,
        )
        accelerator.print(f'Epoch: {epoch}, loss: {loss}, metrics: {metrics}')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config_file', required=True, type=str)
    parser.add_argument('--chkpt_file', required=False, type=str, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
