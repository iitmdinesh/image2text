from typing import Union, Optional, Tuple

import abc
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint

from configs.models import (
    TransformerDecoderConfig,
    HuggingfaceDecoderConfig,
    MLPConfig,
    ModelType,
)
from models.layers import (
    TransformerBlock,
    LayerNorm,
    AdvancedPositionalBiasMLP,
)
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedModel,
    GPT2LMHeadModel,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from models.utils import mutate_transformer_config


class Decoder(nn.Module, abc.ABC):
    """
    Base class for decoder models
    """
    def __init__(self):
        super().__init__()

    @classmethod
    def from_config(cls, config: Union[TransformerDecoderConfig, HuggingfaceDecoderConfig],
                    loose=False,
                    space_for_prompt=0):
        if isinstance(config, TransformerDecoderConfig):
            if config.pretrained_model is None:
                return TransformerDecoder(config, space_for_prompt)
            model_type = config.pretrained_model
            config_args = {
                ModelType.GPT2: dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                ModelType.GPT2_MEDIUM: dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
                ModelType.GPT2_LARGE: dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
                ModelType.GPT2_XL: dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
            }[model_type]
            if not loose:
                assert config.n_layer == config_args['n_layer'], 'provided configs do not match the pretrained model'
                assert config.transformer_config.attn_config.n_embd == config_args['n_embd'], \
                    'provided configs do not match the pretrained model'
                assert config.transformer_config.attn_config.n_head == config_args['n_head'], \
                    'provided configs do not match the pretrained model'
                assert config.transformer_config.attn_config.bias is True, 'provided configs do not match the pretrained model'
                assert config.block_size == 1024, 'provided configs do not match the pretrained model'
                assert not config.transformer_config.is_sparse_attn, 'provided configs do not match the pretrained model'
                assert config.transformer_config.is_causal is True, 'provided configs do not match the pretrained model'
                assert isinstance(config.transformer_config.rotator_config, MLPConfig) and \
                       config.transformer_config.rotator_config.ff_mult == 4, 'provided configs do not match the pretrained ' \
                                                                       'model'
            assert config.vocab_size >= 50257, 'vocab should not shrink'
            model = TransformerDecoder(config, space_for_prompt)
            sd = model.state_dict()

            model_hf: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(model_type.value)
            if config.vocab_size > 50257:
                model_hf.resize_token_embeddings(config.vocab_size)

            sd_hf = model_hf.state_dict()

            # copy while ensuring all of the parameters are aligned and match in names and shapes
            sd_keys_hf = sd_hf.keys()
            sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
            sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # same, just the mask (buffer)
            transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
            # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
            # this means that we have to transpose these weights when we import them
            # assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
            for k in sd_keys_hf:
                if any(k.endswith(w) for w in transposed):
                    # special treatment for the Conv1D weights we need to transpose
                    if k in sd:
                        assert sd_hf[k].shape[::-1] == sd[k].shape, k
                        with torch.no_grad():
                            sd[k].copy_(sd_hf[k].t())
                    else:
                        if not loose:
                            raise ValueError(f'{k} is not present in state dict!!!')
                else:
                    if k in sd:
                        # vanilla copy over the other parameters
                        assert sd_hf[k].shape == sd[k].shape, k
                        with torch.no_grad():
                            sd[k].copy_(sd_hf[k])
                    else:
                        if not loose:
                            raise ValueError(f'{k} is not present in state dict!!!')

            # finally tie weights of the model
            model.tie_weights()
            return model

        elif isinstance(config, HuggingfaceDecoderConfig):
            if config.model_str.startswith('gpt2'):
                return GPT2HuggingfaceDecoder(config)
            elif config.model_str.startswith('tiiuae/falcon'):
                return FalconHuggingfaceDecoder(config)
            else:
                print("Warning! Can use this constructor only if you don't want to do soft prompting in an "
                      "encoder-decoder setup")
                return HuggingfaceDecoder(config)
        else:
            raise ValueError('Unknown config type!!!')

    def forward(self,
                idx: Optional[torch.LongTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                cross_attn_embeds: Optional[torch.FloatTensor] = None,
                attn_msk: Optional[torch.BoolTensor] = None) -> \
            Tuple[torch.FloatTensor, torch.FloatTensor]:
        raise ValueError('not implemented in the base class')

    def tie_weights(self):
        pass

    def get_inputs_embeds(self, idx: torch.LongTensor):
        raise ValueError('not implemented in the base class')

    @property
    def block_size(self):
        raise ValueError('not implemented in the base class')

    @property
    def n_embd(self):
        raise ValueError('not implemented in the base class')


class TransformerDecoder(Decoder):
    def __init__(self, config: TransformerDecoderConfig, space_for_prompt: int):
        super().__init__()
        self.config = config
        self.use_advanced_pos_emb = config.use_advanced_pos_emb
        self.enable_gradient_checkpointing = config.enable_gradient_checkpointing
        self.skip_alternate_cross_attn = config.skip_alternate_cross_attn

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.transformer_config.attn_config.n_embd),
            wpe=AdvancedPositionalBiasMLP(
                context_width=config.block_size,
                in_features=self.n_embd,
                out_features=self.n_embd,
                gate_sizes=config.advanced_pos_emb_gate_sizes,
                add_residual_connection=True,
            ) if self.use_advanced_pos_emb else nn.Embedding(config.block_size, self.n_embd),
            drop=nn.Dropout(config.transformer_config.attn_config.dropout),
            h=nn.ModuleList([TransformerBlock(
                mutate_transformer_config(config.transformer_config, depth, config.skip_alternate_cross_attn),
                depth,
                space_for_prompt,
            )
                for depth in range(config.n_layer)
            ]),
            ln_f=LayerNorm(self.n_embd,
                           bias=config.transformer_config.attn_config.bias),
        ))
        self.lm_head = nn.Linear(self.n_embd, config.vocab_size, bias=False)
        self.tie_weights()

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def tie_weights(self):
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self,
                idx: Optional[torch.LongTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                cross_attn_embeds: Optional[torch.FloatTensor] = None,
                attn_msk: Optional[torch.BoolTensor] = None) -> \
            Tuple[torch.FloatTensor, torch.FloatTensor]:
        assert not (idx is None and inputs_embeds is None)
        assert idx is None or inputs_embeds is None
        if idx is not None:
            device = idx.device
            b, t = idx.size()
        else:
            device = inputs_embeds.device
            b, t, _ = inputs_embeds.size()

        assert t <= self.block_size, f"Cannot forward sequence of length {t}, " \
                                     f"block size is only {self.block_size}"
        if inputs_embeds is None:
            # forward the GPT model itself
            # token embeddings of shape (b, t, n_embd)
            inputs_embeds = self.transformer.wte(idx)

        if self.use_advanced_pos_emb:
            x = self.transformer.drop(self.transformer.wpe(inputs_embeds))
        else:
            # shape (t)
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            # position embeddings of shape (t, n_embd)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(inputs_embeds + pos_emb)

        jit_op = torch.jit.is_scripting() or torch.jit.is_tracing()
        for depth, block in enumerate(self.transformer.h):
            if self.skip_alternate_cross_attn:
                cross_attn_inputs = cross_attn_embeds if depth % 2 == 0 else None
            else:
                cross_attn_inputs = cross_attn_embeds
            if self.enable_gradient_checkpointing and self.training and not jit_op:
                x = self.gradient_checkpointed_transformer_block(block, x, cross_attn_inputs, attn_msk)
            else:
                x = block(x, cross_attn_inputs=cross_attn_inputs, attn_mask=attn_msk)
        hidden_state = x
        x = self.transformer.ln_f(x)
        return self.lm_head(x), hidden_state

    def get_inputs_embeds(self, idx: torch.LongTensor):
        return self.transformer.wte(idx)

    @property
    def block_size(self):
        return self.config.block_size

    # See https://pytorch.org/docs/stable/generated/torch.jit.ignore.html
    @torch.jit.unused
    def gradient_checkpointed_transformer_block(
            self,
            mod: nn.Module,
            x: torch.Tensor,
            cross_attn_inputs: Optional[torch.Tensor],
            attn_msk: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return torch.utils.checkpoint.checkpoint(
            lambda *args: mod(args[0], cross_attn_inputs=args[1], attn_mask=args[2]),
            x,
            cross_attn_inputs,
            attn_msk,
        )
    @property
    def n_embd(self):
        return self.config.transformer_config.attn_config.n_embd


class HuggingfaceDecoder(Decoder, abc.ABC):
    """Need to implement the get_inputs_embeds method based on the arch of the Huggingface model"""
    def __init__(self, config: HuggingfaceDecoderConfig):
        super().__init__()
        self.config = config
        self.use_cross_attn = config.use_cross_attn

        if config.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        else:
            bnb_config = None

        kwargs = {}
        if config.use_cross_attn:
            hf_config = AutoModelForCausalLM.from_pretrained(
                config.model_str,
                quantization_config=bnb_config,
                trust_remote_code=True,
            ).config
            if hasattr(hf_config, 'add_cross_attention'):
                hf_config.add_cross_attention = True
            else:
                raise ValueError("Don't know how to use cross attention with this model. "
                                 "Suggest you try a different config!!!")
            kwargs['config'] = hf_config

        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(config.model_str,
                                                                      quantization_config=bnb_config,
                                                                      trust_remote_code=True,
                                                                      **kwargs)
        self.hf_config = model.config
        model.resize_token_embeddings(config.vocab_size + config.extra_tokens)

        if config.enable_gradient_checkpointing:
            model.gradient_checkpointing_enable()

        if config.prepare_for_kbit_training:
            model = prepare_model_for_kbit_training(model,
                                                    use_gradient_checkpointing=config.enable_gradient_checkpointing)

        lora_spec = config.lora_spec
        if lora_spec.enable_lora:
            lora_config = LoraConfig(
                r=lora_spec.r,
                lora_alpha=lora_spec.lora_alpha,
                lora_dropout=lora_spec.lora_dropout,
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                target_modules=lora_spec.target_modules,
            )
            model = get_peft_model(model, lora_config)

            if lora_spec.force_enable_update_modules is not None:
                for n, p in model.named_parameters():
                    for pattern in lora_spec.force_enable_update_modules:
                        if pattern in n:
                            p.requires_grad = True

        self.backbone = model

    def tie_weights(self):
        self.backbone.tie_weights()

    def forward(self,
                idx: Optional[torch.LongTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                cross_attn_embeds: Optional[torch.FloatTensor] = None,
                attn_msk: Optional[torch.BoolTensor] = None) -> \
            Tuple[torch.FloatTensor, torch.FloatTensor]:
        assert not (idx is None and inputs_embeds is None)
        assert idx is None or inputs_embeds is None
        if self.use_cross_attn:
            outputs = self.backbone(
                encoder_hidden_states=cross_attn_embeds,
                input_ids=idx,
                inputs_embeds=inputs_embeds,
                attention_mask=attn_msk,
                output_hidden_states=True,
            )
        else:
            # some models error out if we add encoder_hidden_states as input. so only add if needed
            outputs = self.backbone(
                input_ids=idx,
                inputs_embeds=inputs_embeds,
                attention_mask=attn_msk,
                output_hidden_states=True,
            )
        return outputs.logits, outputs.hidden_states[-1]


class GPT2HuggingfaceDecoder(HuggingfaceDecoder):
    def __init__(self, config: HuggingfaceDecoderConfig):
        assert config.model_str.startswith('gpt2')
        super().__init__(config)

    def get_inputs_embeds(self, idx: torch.LongTensor):
        return self.backbone.transformer.wte(idx)

    @property
    def block_size(self):
        return 1024

    @property
    def n_embd(self):
        return self.hf_config.n_embd


class FalconHuggingfaceDecoder(HuggingfaceDecoder):
    def __init__(self, config: HuggingfaceDecoderConfig):
        assert config.model_str.startswith('tiiuae/falcon')
        assert config.vocab_size >= 65024
        super().__init__(config)

    def get_inputs_embeds(self, idx: torch.LongTensor):
        return self.backbone.transformer.word_embeddings(idx)

    @property
    def block_size(self):
        return 2048

    @property
    def n_embd(self):
        return self.hf_config.hidden_size
