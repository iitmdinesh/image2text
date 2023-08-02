# README

## Dataset
Flickr30K (https://datasets.activeloop.ai/docs/ml/datasets/flickr30k-dataset/) 

## Getting started
```shell
pip install -r requirements.txt
```

## Launch jobs
```shell
mkdir checkpoints
```
On Mac OSX
```shell
export PYTORCH_ENABLE_MPS_FALLBACK=1; accelerate launch trainer.py --config_file training_configs/local/nano.yaml --chkpt_file checkpoints/nano.pt
```
To disable MPS / CUDA, do this
```shell
accelerate launch --cpu trainer.py --config_file training_configs/local/nano.yaml --chkpt_file checkpoints/nano.pt
```
Choose configs from `training_configs/local` for local runs or `training_configs/gpu` for gpu runs or 
write your own (In which case it must a yaml file that maps to the `TrainingConfig` class in 
`configs/trainer.py`). Try using `training_configs/local/nano.yaml` for fast prototyping locally.

## Add a new Huggingface model as a decoder?
* Subclass `HuggingfaceDecoder` in `models/decoder.py` like how `GPT2HuggingfaceDecoder` and `FalconHuggingfaceDecoder`
  do so.
* Update the `Decoder.from_config` method in the same file appropriately 

## References
 * Some transformer encoder/decoder code was taken from nanoGPT repository (https://github.com/karpathy/nanoGPT)
 * Huggingface ecosystem (transformers, accelerate and peft libraries and of course the model hub)
 * Momentum distillation (https://arxiv.org/abs/2107.07651). Why does this help? Intuition is this helps with choosing 
   the right scale for learning rate (self-similarity that momentum distillation enforces essentially promotes slower 
   updates)

## Validation results
Using config `training_configs/local/nano.yaml`

<img src="assets/skateboarders.png" alt="Goal" width="550"/>

Labels:

1. The teenage boys in black shirts skateboarding in a skate park , surrounded by people 
2. Two teen guys are doing skateboard tricks at a skate park 
3. Two kids in jeans perform skateboard tricks 
4. Two guys are in the air while skateboarding 
5. Two teenagers doing tricks on skateboards

Predictions:

1. A young man is riding his skateboard while wearing a helmet.
2. Three skateboarders are doing tricks on a sidewalk.
3. A man skateboarded down some stairs.
4. Two skateboarders skateboarding on the sidewalk of a park.
5. A man skateboards down a brick railing on an empty lot.
6. Three people playing tricks in a park.
7. A boy skateboarding down stairs.
8. A man in a black shirt is doing a skate park trick.

<img src="assets/dog-snow.png" alt="Goal" width="550"/>

Labels:

1. Two white dogs walk through a huge bank of mountain snow 
2. Two white dogs are walking through deep white snow 
3. Two fluffy white dogs are in the snow
4. Two white dogs walk in a snowy setting 
5. Two white dogs walking in the snow 


Predictions:

1. A young dog running through snow.
2. A dog running through snow
3. A dog on a sled in the snow.
4. A dog in the snow.
5. A black dog stands on a snowy side of a trail.
6. Two small dogs running down the road.
7. A dog is carrying a small snow pack in a snowy field.
8. A dog is running in the snow in a field of trees.

