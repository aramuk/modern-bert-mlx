from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


# parity with HuggingFace version
class ModernBERTConfig(BaseModel):
    vocab_size: int = 50368
    hidden_size: int = 768
    intermediate_size: int = 1152
    num_hidden_lauers: int = 22
    hidden_activation: str = "gelu"
    max_position_embeddings: int = 8192
    initializer_range: float = 0.02
    initializer_cutoff_factor: float = 2.0
    norm_eps: float = 1e-5
    norm_bias: bool = False
    pad_token_id: int = 50283
    eos_token_id: int = 50282
    bos_token_id: int = 50281
    cls_token_id: int = 50281
    sep_token_id: int = 50282
    global_rope_theta: float = 160000.0
    attention_bias: int = False
    attention_dropout: float = 0.0
    global_attn_every_n_layers: int = 3
    local_attention: int = 128
    local_rope_theta: float = 10000.0
    embedding_dropout: float = 0.0
    mlp_bias: bool = False
    mlp_dropout: float = 0.0
    decoder_bias: bool = True
    classifier_pooling: Literal["cls", "mean"] = "cls"
    classifier_dropout: float = 0.0
    classifier_bias: bool = False
    classifier_activation: str = "gelu"
    deterministic_flash_attn: bool = False
    sparse_prediction: bool = False
    sparse_pred_ignore_index: int = -100
    reference_compile: Optional[bool] = None
    repad_logits_with_grad: bool = False


class AttentionImpl(str, Enum):
    naive = "naive"
    sdpa = "sdpa"
    flash = "flash"


class AttentionConfig(BaseModel):
    implementation: AttentionImpl = AttentionImpl.naive
    num_attention_heads: int = 12
    attention_bias: bool = False
    attention_dropout: float = 0.0
    global_attn_every_n_layers: int = 3
    global_rope_theta: float = 160000.0
    local_attention: int = 128
    local_rope_theta: int = 10000.0
    rope_max_pos_embeddings: int = 2048
    output_attn: bool = True


class ModernBertConfig(BaseModel):
    # Embedding args
    context_length: int = 8192
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
