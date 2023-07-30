from collections import namedtuple


VisionEncoderDecoderModelOutput = namedtuple('VisionEncoderDecoderModelOutput', ['encoder_output', 'logits',
                                                                                 'hidden_state'])
