import unittest
import torch

from models.vision_encoder_decoder import VisionEncoderDecoder
from configs.models import (
    MoEConfig,
    SelfAttentionConfig,
    SelfAttentionType,
    TransformerConfig,
    VisionTransformerEncoderConfig,
    TransformerDecoderConfig,
    VisionEncoderDecoderConfig,
    ImageInputSpec,
)


class VisionEncoderDecoderTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_vision_encoder_decoder(self):
        decoder_transformer_config = TransformerConfig(
            rotator_config=MoEConfig(
                num_experts=4,
                proj_features=8,
                gate_sizes=None,
                ff_mult_factor=2.5,
                top_k=2,
            ),
            attn_config=SelfAttentionConfig(
                attn_type=SelfAttentionType.MULTI_QUERY,
                n_embd=64,
                n_head=4,
            ),
            is_causal=True,
            is_cross_attn=True,
        )
        decoder_config = TransformerDecoderConfig(
            transformer_config=decoder_transformer_config,
            n_layer=2,
            block_size=256,
            vocab_size=1024,
        )
        encoder_transformer_config = TransformerConfig(
            rotator_config=MoEConfig(
                num_experts=4,
                proj_features=8,
                gate_sizes=None,
                ff_mult_factor=2.5,
                top_k=2,
            ),
            attn_config=SelfAttentionConfig(
                attn_type=SelfAttentionType.MULTI_QUERY,
                n_embd=64,
                n_head=4,
            ),
            is_causal=False,
            is_cross_attn=False,
        )
        image_input_spec = ImageInputSpec(
            n_channels=3,
            width=128,
            height=128,
        )
        vision_encoder_config = VisionTransformerEncoderConfig(
            transformer_config=encoder_transformer_config,
            enable_gradient_checkpointing=True,
            input=image_input_spec,
            n_layer=2,
            n_cls=24,
            num_patches=32,
            n_channels=32,
            feature_extractor_gate_sizes=(8, 16),
            feature_extractor_kernel_size=(4, 4),
        )
        config = VisionEncoderDecoderConfig(
            vision_encoder_config=vision_encoder_config,
            decoder_config=decoder_config,
            use_cross_attn=True,
            use_soft_prompting=True,
        )
        model = VisionEncoderDecoder(config)
        inp = torch.randint(0, 256, (96, 3, 128, 128)).float()
        ids = torch.randint(0, 1024, (96, 192,))
        attn_mask = torch.randint(0, 2, (192, 192), dtype=torch.bool)
        outs = model(images=inp, ids=ids, attn_msk=attn_mask)
        self.assertEqual((96, 24, 64), outs.encoder_output.shape)
        self.assertEqual((96, 192, 1024), outs.logits.shape)

        generated_ids = model.generate(inp[:2], ids[:2], max_new_tokens=16, temperature=1.0, top_k=16)

        self.assertEqual((2, 192 + 16), generated_ids.shape)
