from typing import Optional
import torch
import torch.nn as nn
from einops import repeat

from transformers import (
    LogitsProcessorList,
    NoRepeatNGramLogitsProcessor,
)
from configs.models import VisionEncoderDecoderConfig
from object_models import VisionEncoderDecoderModelOutput
from models.encoder import Encoder
from models.decoder import Decoder
from models.utils import update_state_dict_from_partial_checkpoint


class VisionEncoderDecoder(nn.Module):
    """Encoder Decoder model for conditional generation"""
    def __init__(self,
                 config: VisionEncoderDecoderConfig,
                 encoder: Optional[Encoder] = None,
                 decoder: Optional[Decoder] = None,
                 ):
        super().__init__()
        self.config = config
        encoder = encoder if encoder is not None else Encoder.from_config(config.vision_encoder_config)
        self.space_for_prompt = encoder.num_outputs if config.use_soft_prompting else 0
        self.decoder = decoder if decoder is not None else \
            Decoder.from_config(config=config.decoder_config,
                                loose=config.loose_match_decoder_state_dict,
                                space_for_prompt=self.space_for_prompt)
        decoder_n_embd = self.decoder.n_embd
        if encoder.output_embed_dim != decoder_n_embd:
            self.encoder = nn.Sequential(
                encoder,
                nn.Linear(encoder.output_embed_dim, decoder_n_embd, bias=False),
            )
        else:
            self.encoder = encoder
        self.processor = LogitsProcessorList(
            [NoRepeatNGramLogitsProcessor(ngram_size=no_repeat_n_gram)
             for no_repeat_n_gram in config.no_repeat_n_grams]
        )
        self.use_cross_attn = config.use_cross_attn
        self.use_soft_prompting = config.use_soft_prompting
        if not (self.use_cross_attn or self.use_soft_prompting):
            raise ValueError('Misconfigured!!! Need to either use cross attn or soft prompting or both')
        if config.chkpt_path is not None:
            update_state_dict_from_partial_checkpoint(self, config.chkpt_path, map_location=None)

    def forward(
            self,
            images: Optional[torch.FloatTensor],
            ids: torch.LongTensor,
            attn_msk: Optional[torch.BoolTensor] = None,
            encoder_output: Optional[torch.Tensor] = None,
    ) -> VisionEncoderDecoderModelOutput:
        if encoder_output is None:
            encoder_output = self.encoder(images)
        bs = encoder_output.size(0)
        if attn_msk is not None:
            if len(attn_msk.size()) == 2:
                s = attn_msk.size(1)
                if attn_msk.size(0) == bs:
                    attn_msk = repeat(attn_msk, 'bs s -> bs h s l', h=1, l=s)
                else:
                    attn_msk = repeat(attn_msk, 's l -> bs h s l', bs=bs, h=1)
            elif len(attn_msk.size()) == 3:
                if attn_msk.size(0) == bs:
                    attn_msk = repeat(attn_msk, 'bs s l -> bs h s l', h=1)
                else:
                    attn_msk = repeat(attn_msk, 'h s l -> bs h s l', bs=bs)

        # decoder is causal, so add this
        L = ids.size(-1)
        device = ids.device
        attn_mask_causal = torch.ones((L, L), device=device, dtype=torch.bool).tril(diagonal=0)
        attn_mask_causal = repeat(
            attn_mask_causal,
            's l -> b h s l', b=1, h=1
        )
        attn_msk = attn_mask_causal if attn_msk is None else torch.logical_and(attn_msk, attn_mask_causal)

        if self.use_soft_prompting:
            inputs_embeds = torch.cat(
                (encoder_output, self.decoder.get_inputs_embeds(ids)),
                dim=-2
            )[..., :self.decoder.block_size, :].contiguous()
            if attn_msk is not None:
                bs, ncls, _ = encoder_output.size()
                _, h, s, _ = attn_msk.size()
                device = encoder_output.device
                attn_msk_new = -float('inf') * torch.ones((bs, h, ncls + s, ncls + s), device=device)
                # everyone can attend to cls tokens
                attn_msk_new[..., :ncls, :] = 0
                # non-cls can be attended by non-cls attn_msk permitting
                attn_msk = attn_msk.masked_fill(~attn_msk, -float('inf')).float()
                attn_msk[attn_msk == 1] = 0
                attn_msk_new[..., ncls:, ncls:] = attn_msk
                attn_msk = attn_msk_new[..., :self.decoder.block_size, :self.decoder.block_size].contiguous()
            else:
                bs, ncls, _ = encoder_output.size()
                h = 1
                s = ids.size(-1)
                device = encoder_output.device
                attn_msk = -float('inf') * torch.ones((bs, h, ncls + s, ncls + s), device=device)
                # everyone can attend to cls tokens
                attn_msk[..., :ncls, :] = 0
                # non-cls can be attended by non-cls
                attn_msk[..., ncls:, ncls:] = 0
                attn_msk = attn_msk[..., :self.decoder.block_size, :self.decoder.block_size].contiguous()
            ids = None
            offset = encoder_output.size(-2)
        else:
            inputs_embeds = None
            offset = 0
            if attn_msk is not None:
                attn_msk = attn_msk.masked_fill(~attn_msk, -float('inf')).float()
                attn_msk[attn_msk == 1] = 0
        if self.use_cross_attn:
            cross_attn_values = encoder_output
        else:
            cross_attn_values = None
        logits, hidden_state = self.decoder(
            idx=ids,
            inputs_embeds=inputs_embeds,
            cross_attn_embeds=cross_attn_values,
            attn_msk=attn_msk,
        )
        return VisionEncoderDecoderModelOutput(
            encoder_output=encoder_output,
            logits=logits[..., offset:, :].contiguous(),
            hidden_state=hidden_state,
        )

    @torch.no_grad()
    def generate(self, images, prompt_ids, max_new_tokens=128, temperature=1.0, top_k=None, nucleus_p=None) -> \
            torch.LongTensor:
        blk_size = self.decoder.block_size - self.space_for_prompt
        assert max_new_tokens <= blk_size - prompt_ids.size(-1)
        encoder_output = None
        decoder_ids = prompt_ids
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            decoder_ids_cond = decoder_ids if decoder_ids.size(-1) <= blk_size else \
                decoder_ids[..., -blk_size:].contiguous()
            # forward the model to get the logits for the index in the sequence
            output = self(images=images, ids=decoder_ids_cond, encoder_output=encoder_output)
            encoder_output = output.encoder_output
            logits = output.logits
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[..., -1, :] / temperature
            logits = self.processor(decoder_ids, logits)
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)), sorted=True, dim=-1)
                logits[logits < v[..., [-1]]] = -float('inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = logits.softmax(dim=-1)
            if nucleus_p is not None:
                # Apply nucleus (top-p) sampling
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # Find the indices to truncate based on nucleus_p while ensuring at least one nnz
                threshold_p = torch.maximum(nucleus_p * torch.ones_like(sorted_probs[:, 0]), sorted_probs[:, 0]). \
                    unsqueeze(1)
                batch_idx, vocab_idx = (cumulative_probs > threshold_p).nonzero(as_tuple=True)

                sorted_probs[batch_idx, vocab_idx] = 0
                sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

                # Sample from the truncated distribution
                sampled_index = torch.multinomial(sorted_probs, num_samples=1)
                idx_next = sorted_indices.gather(dim=-1, index=sampled_index)
            else:
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            decoder_ids = torch.cat((decoder_ids, idx_next), dim=-1)

        return decoder_ids
