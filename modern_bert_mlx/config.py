from typing import Tuple

from pydantic import BaseModel, Field


class AttentionConfig(BaseModel):
    num_attention_heads: int = 12
    attention_bias: bool = False
    attention_dropout: float = 0.0
    global_attn_every_n_layers: int = 3
    global_rope_theta: float = 160000.0
    local_attention: int = 128
    local_rope_theta: int = 10000.0
    rope_max_pos_embeddings: int = 2048
    output_attn: bool = 12


class ModernBertConfig(BaseModel):
    # Embedding args
    vocab_size: int = 50368
    hidden_dim: int = 768
    pad_token_idx: int = 50283  # Not supported?
    embed_drop_p: float = 0.0
    # NN args
    intermediate_dim: int = 1152
    layer_norm_eps: float = 1e-5
    norm_bias: bool = False
    # Encoder Args
    encoder_num_layers: int = 22
    encoder_drop_p: float = 0.0
    # Attention Args
    attention: AttentionConfig = Field(default_factory=AttentionConfig)
    # MLP Args
    mlp_bias: bool = False
    mlp_dropout: float = 0.0
    # Decoder Args
    decoder_bias: bool = True
