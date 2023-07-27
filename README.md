# README

## Dataset
Flickr30K (https://datasets.activeloop.ai/docs/ml/datasets/flickr30k-dataset/) 

## Getting started
```shell
pip install -r requirements.txt
```

## Launch jobs
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
