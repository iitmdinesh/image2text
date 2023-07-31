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

1. A man is skateboarding.
2. Man playing skateboard.
3. A woman with blue shirt is skating down a wall.
4. A rollerblader doing a trick with his wheels.
5. A man in a black jacket on a skateboard does a jump off a railing.
6. A man on a skateboard is playing a black skateboarding game at a park.
7. A skateboarder is performing stunt tricks in front of a graffiti backdrop.
8. A man with a skateboard sits in front of an orange building.

<img src="assets/dog-snow.png" alt="Goal" width="550"/>

Labels:

1. Two white dogs walk through a huge bank of mountain snow 
2. Two white dogs are walking through deep white snow 
3. Two fluffy white dogs are in the snow
4. Two white dogs walk in a snowy setting 
5. Two white dogs walking in the snow 


Predictions:

1. A big dog standing in the snow.
2. A brown dog is in the snow, on his back.
3. A brown dog is running across snowy terrain.
4. A dog playing in the icy snow with a ball.
5. A dog runs on a snowy road.
6. A dog jumps in a snowy field.
7. A dog running in the snow.
8. A small dog running through snow.

