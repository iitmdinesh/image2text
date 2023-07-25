import math
from typing import Optional

import torch

from models.vision_encoder_decoder import VisionEncoderDecoder
from transformers import LogitsProcessorList, NoRepeatNGramLogitsProcessor


class BeamSearchTokenGenerator:
    def __init__(self,
                 model: VisionEncoderDecoder,
                 beam_width: int = 3,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None,
                 max_new_tokens=64,
                 no_repeat_n_grams=(2, 3, 4),
                 beam_expansion_factor: int = 4,
                 eos_token_id: Optional[int] = None,
                 consolidation_temperature: float = 1.0,
                 length_boost: float = 1.0
                 ):
        self.model = model
        self.beam_width = beam_width
        self.beam_expansion_factor = beam_expansion_factor
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.consolidation_temperature = consolidation_temperature
        self.top_k = top_k
        self.eos_token_id = eos_token_id
        self.length_boost = math.log(length_boost)
        self.processor = LogitsProcessorList(
            [NoRepeatNGramLogitsProcessor(ngram_size=no_repeat_n_gram) for no_repeat_n_gram in no_repeat_n_grams])

    def __call__(self, inputs, decoded_ids):
        self.model.eval()
        x = self.model.encoder(inputs).repeat([self.beam_width, 1, 1, 1])
        n_cls = x.size(2)
        n_embed = x.size(-1)
        x = x.reshape(-1, n_cls, n_embed)
        num_tokens_provided = decoded_ids.size(-1) - 1
        decoded_ids = decoded_ids.unsqueeze(0).expand(self.beam_width, -1, -1)

        cumulative_log_scores = torch.zeros((self.beam_width, inputs.size(0)), device=x.device)

        while not (decoded_ids.size(-1) >= (self.max_new_tokens + num_tokens_provided) or
                   bool(((decoded_ids == self.eos_token_id).sum(dim=-1) > 0).all())):
            next_ids, next_log_scores = self.decode_next(x, decoded_ids)
            decoded_ids, cumulative_log_scores = self.consolidate_candidates(
                decoded_ids, cumulative_log_scores, next_ids, next_log_scores
            )
        # beams are sorted by score already, pick the first one
        # result = decoded_ids[0, ...]
        # score = cumulative_log_scores[0, ...]
        return decoded_ids.permute([1, 0, 2]), cumulative_log_scores.permute([1, 0])

    def decode_next(self, x, decoded_ids):
        batch_size = decoded_ids.size(1)
        current = decoded_ids.size(2)
        decoded_ids = decoded_ids.reshape(-1, current)
        if self.eos_token_id is not None:
            where_eos = (decoded_ids[:, -1] == self.eos_token_id).unsqueeze(-1)
        else:
            where_eos = torch.zeros_like(decoded_ids[:, [-1]], dtype=torch.bool)
        outputs = self.model(images=None, ids=decoded_ids, encoder_output=x)

        scores = outputs.logits[..., -1, :]
        scores = self.processor(decoded_ids, scores)
        # optionally crop the logits to only the top k options
        if self.top_k is not None:
            v, _ = torch.topk(scores, min(self.top_k, scores.size(-1)), sorted=True, dim=-1)
            scores[scores < v[..., [-1]]] = -float('inf')
        if self.temperature <= 0:
            prob = scores.log_softmax(dim=-1)
            _, next_id = scores.topk(k=self.beam_expansion_factor, dim=-1, sorted=False)
        else:
            prob = (scores / self.temperature).log_softmax(dim=-1)
            next_id = torch.multinomial(prob.exp(), num_samples=self.beam_expansion_factor)
        log_scores = torch.gather(prob, -1, next_id)
        if self.eos_token_id is not None:
            next_id = torch.where(
                torch.logical_and(where_eos, log_scores + self.length_boost < 0),
                self.eos_token_id * torch.ones_like(next_id),
                next_id
            )
            log_scores = torch.where(
                torch.logical_and(where_eos, log_scores + self.length_boost < 0),
                torch.zeros_like(log_scores),
                log_scores + self.length_boost
            )
        next_id = next_id.reshape(self.beam_width, batch_size, self.beam_expansion_factor)
        log_scores = log_scores.reshape(self.beam_width, batch_size, self.beam_expansion_factor)
        return next_id, log_scores

    def consolidate_candidates(self, decoded_ids, cumulative_log_scores, next_ids, next_log_scores):
        beams_idx, candidates_idx = self.identify(cumulative_log_scores, next_log_scores)
        decoded_ids, cumulative_log_scores, next_ids, next_log_scores = \
            self.gather_results(decoded_ids, cumulative_log_scores, next_ids, next_log_scores,
                                beams_idx, candidates_idx)
        return torch.cat((decoded_ids, next_ids), dim=-1), cumulative_log_scores + next_log_scores

    def gather_results(self, decoded_ids, cumulative_log_scores, next_ids, next_log_scores, beams_idx, candidates_idx):
        # beams_idx (bs, bw)
        # candidates_idx (bs, bw)
        decoded_ids = torch.gather(
            decoded_ids.transpose(0, 1), 1, beams_idx.unsqueeze(-1).expand(-1, -1, decoded_ids.size(-1))
        ).transpose(0, 1)
        cumulative_log_scores = torch.gather(
            cumulative_log_scores.transpose(0, 1), 1, beams_idx
        ).transpose(0, 1)

        next_ids = torch.gather(
            next_ids.transpose(0, 1),
            1,
            beams_idx.unsqueeze(-1).expand(-1, -1, next_ids.size(-1))
        ).transpose(0, 1)
        next_ids = torch.gather(next_ids.transpose(0, 1), -1, candidates_idx.unsqueeze(-1)).transpose(0, 1)

        next_log_scores = torch.gather(
            next_log_scores.transpose(0, 1),
            1,
            beams_idx.unsqueeze(-1).expand(-1, -1, next_log_scores.size(-1))
        ).transpose(0, 1)
        next_log_scores = torch.gather(next_log_scores.transpose(0, 1), -1, candidates_idx.unsqueeze(-1)). \
            transpose(0, 1). \
            squeeze(-1)

        return decoded_ids, cumulative_log_scores, next_ids, next_log_scores

    def identify(self, cumulative_log_scores, next_log_scores):
        device = cumulative_log_scores.device
        bs = cumulative_log_scores.size(1)
        k = self.beam_width * self.beam_expansion_factor
        cumulative_log_scores_expanded = (cumulative_log_scores.unsqueeze(2) + next_log_scores). \
            permute(1, 0, 2).reshape(-1, k)
        beams_expanded = torch.arange(0, self.beam_width, device=device).unsqueeze(0).unsqueeze(2).repeat(
            [bs, 1, self.beam_expansion_factor]).reshape(-1, k)
        candidates_expanded = \
            torch.arange(0, self.beam_expansion_factor, device=device).unsqueeze(0).unsqueeze(1).repeat(
                [bs, self.beam_width, 1]).reshape(-1, k)
        if self.consolidation_temperature <= 0:
            _, best_pos = cumulative_log_scores_expanded.topk(k=self.beam_width, dim=-1, sorted=True)
        else:
            prob = (cumulative_log_scores_expanded / self.consolidation_temperature).softmax(dim=-1)
            best_pos = torch.multinomial(prob, num_samples=self.beam_width)
        beams_to_keep = torch.gather(beams_expanded, -1, best_pos)
        candidates_to_keep = torch.gather(candidates_expanded, -1, best_pos)
        return beams_to_keep, candidates_to_keep
