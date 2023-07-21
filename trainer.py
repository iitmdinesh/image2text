from typing import Union

import torch.nn as nn
import torch.utils.data


import yaml

from configs.trainer import TrainingConfig
from training.wrapper import ModelTrainerWrapper
from training.utils import train_loop, unpack_batch

from accelerate import Accelerator
from transformers import AutoTokenizer, PreTrainedTokenizer
import deeplake
from torchvision import transforms
import torch


from argparse import ArgumentParser


def eval_model(model_wrapper: Union[nn.parallel.DistributedDataParallel, ModelTrainerWrapper],
               accelerator,
               tokenizer,
               train_dl,
               epoch,
               ignore_index,
               prompt='The',
               num_candidates=2,
               ):
    accelerator.print(f"Model perf at the end of the {epoch}-th epoch")
    accelerator.print("Train:")
    batch = next(iter(train_dl))
    images, labels_0, labels_1, labels_2, labels_3, labels_4 = unpack_batch(batch,
                                                                            ignore_index=ignore_index)
    device = accelerator.device
    x = images.to(device)[:1].expand(num_candidates, -1, -1)
    label_ = labels_0[0]
    model_wrapper = accelerator.unwrap_model(model_wrapper)
    with accelerator.autocast():
        if prompt is None:
            decoded_ids = torch.empty((x.size(0), 0), device=x.device, dtype=torch.long)
        else:
            decoded_ids = torch.tensor(
                tokenizer(text=prompt).input_ids,
                dtype=torch.long).to(device).unsqueeze(0).expand(x.size(0), -1).contiguous()

        result = model_wrapper.model.generate(x=x,
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


def get_dataloader(tokenizer):
    txform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
    ])
    ds = deeplake.load('hub://activeloop/flickr30k')

    tokenizer.pad_token = tokenizer.eos_token

    def _tok(x):
        return tokenizer(text=x[0], return_tensors='pt',
                         max_length=256,
                         truncation='longest_first',
                         padding='max_length')

    def _transform(x):
        result = {
            'image': txform(x['image']),
        }
        for k in range(5):
            data = x[f'caption_{k}']
            tokenized = _tok(data)
            result[f'input_ids_{k}'] = tokenized.input_ids.squeeze(0)
            result[f'attn_mask_{k}'] = tokenized.attention_mask.squeeze(0)
        return result

    dl = ds.pytorch(batch_size=4, shuffle=False, num_workers=8, transform=_transform)
    return dl


def main(args):
    obj = yaml.safe_load(open(args.config_file, 'r'))
    config: TrainingConfig = TrainingConfig.model_validate(obj)
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
    if tokenizer.mask_token_id is None:
        kwargs['mask_token'] = '<MSK>'
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_str, **kwargs)

    train_dl = get_dataloader(tokenizer)

    model_wrapper = ModelTrainerWrapper(
        model_config=config.model,
        tokenizer=tokenizer,
        trainer_config=config.trainer,
        ignore_index=config.ignore_index,
    ).to(accelerator.device)
    accelerator.print(model_wrapper.model)
    optimizer = torch.optim.AdamW(
        model_wrapper.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
    )
    model_wrapper, optimizer, train_dl = \
        accelerator.prepare(model_wrapper, optimizer, train_dl, device_placement=[False, True, True])
    for epoch in range(config.epochs):
        train_loop(model_wrapper,
                   optimizer,
                   train_dl,
                   epoch,
                   config.num_steps,
                   accelerator,
                   disable_flash=config.disable_flash,
                   eos_token_id=None,
                   ignore_index=config.ignore_index,
                   reset_moco_after_k_epochs=config.reset_moco_after_k_epochs,
                   chckpt_fname=args.chkpt_file)
        eval_model(model_wrapper, accelerator, tokenizer, train_dl, epoch, config.ignore_index)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config_file', required=True, type=str)
    parser.add_argument('--chkpt_file', required=False, type=str, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
