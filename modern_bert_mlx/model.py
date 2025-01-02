from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from pydantic import BaseModel

from modern_bert_mlx.nn import ModuleList


class ModernBertConfig(BaseModel):
    # Embedding args
    vocab_size: int = 50368
    embed_dim: int = 768
    pad_token_idx: int = 50283  # Not supported?
    embed_drop_p: float = 0.0
    # NN args
    hidden_dim: int = 2304  # 768 * 3
    intermediate_dim: int = 1152  # 2304 / 2
    layer_norm_eps: float = 1e-5
    # Encoder Args
    encoder_num_layers: int = 22
    encoder_drop_p: float = 0.0
    # Attention Args
    num_attention_heads: int = 12



class ModernBertEmbedder(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.embed_dim)
        self.norm = nn.LayerNorm(
            config.embed_dim, eps=config.layer_norm_eps, affine=True
        )
        self.drop = nn.Dropout(p=config.embed_drop_p)

    def __call__(
        self, input_ids: mx.array, token_type_ids: Optional[mx.array] = None
    ) -> mx.array:
        # TODO: Conditionally run JIT implementation based on self.config.
        h = self.drop(self.norm(self.tok_embedding(input_ids)))
        return h


class ModernBertMLP(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.Wi = nn.Linear(config.embed_dim, config.hidden_dim, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(config.encoder_drop_p)
        self.Wo = nn.Linear(config.intermediate_dim, config.embed_dim, bias=False)

    def __call__(self, h: mx.array) -> mx.array:
        inp, gate = self.Wi(h).split(2, dim=-1)
        out = self.Wo(self.drop(self.act(inp) * gate))
        return out


class ModernBertAttention(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.nheads = self.config.num_attention_heads
        self.head_dim = self.config.hidden_dim // self.config.num_attention_heads
        self.Wqkv = nn.Linear(config.embed_dim, config.hidden_dim, bias=False)
        self.rotary_emb = nn.RoPE(self.head_dim)
        self.Wo = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.drop = nn.Identity()

    def __call__(
        self,
        h: mx.array,
        mask: mx.array,
        output_attn: Optional[bool] = False,
    ) -> Union[Tuple[mx.array, mx.array], mx.array]:
        qkv = self.Wqkv(h)
        print(qkv.shape)
        batch_size, seq_len, all_dims = qkv.shape
        # [batch_size, seq_len, 3, nheads,  head_dim]
        qkv = qkv.view(batch_size, -1, 3, self.nheads, self.head_dim)
        print(qkv.shape)

        Q, K, V = map(mx.squeeze, qkv.transpose(3, 1).split(3, axis=2))

        # TODO: use same rotation for Q, K
        # TODO: verify the dimension is correct
        # [batch_size, nheads, seqlen, head_dim]
        Q_rot = self.rotary_emb(Q)
        K_rot = self.rotary_emb(K)
        # Q_rot = mx.core.fast.rope(Q, dims=self.head_dim, traditional=False)
        # K_rot = mx.core.fast.rope(K, dims=self.head_dim, traditional=False)

        # TODO: select configured attention implementation dynamically
        attn_out: mx.array = mx.core.fast.scaled_dot_product_attention(
            Q_rot, K_rot, V, mask=mask
        )
        attn_out = attn_out.transpose(1, 2).view(batch_size, -1, self.config.hidden_dim)
        h = attn_out[0]
        h = self.drop(self.Wo(h))

        return (h, attn_out[1:]) if output_attn else h


class ModernBertEncoderLayer(nn.Module):
    def __init__(self, config: ModernBertConfig, has_attn_norm: bool = True):
        super().__init__()
        if has_attn_norm:
            self.attn_norm = nn.LayerNorm(
                config.embed_dim, config.layer_norm_eps, affine=True
            )
        else:
            self.attn_norm = nn.Identity()
        self.attn = ModernBertAttention(config)
        self.mlp_norm = nn.LayerNorm(
            config.embed_dim, config.layer_norm_eps, affine=True
        )
        self.mlp = ModernBertMLP(config)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        attn_out = self.attn(x, mask)
        attn_out = self.attn_norm(x + attn_out)

        mlp_out = self.mlp(x)
        mlp_out = self.mlp_norm(attn_out)
        return mlp_out, mask


class ModernBertBackbone(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.embedder = ModernBertEmbedder(config)
        self.encoder = ModuleList(
            [ModernBertEncoderLayer(config, has_attn_norm=False)]
            + [ModernBertEncoderLayer(config) for _ in range(1, config.encoder_num_layers)]
        )
        self.final_norm = nn.LayerNorm(
            config.intermediate_dim, eps=config.layer_norm_eps, affine=True
        )

    def __call__(
        self,
        input_ids: mx.array,
        token_type_ids: mx.array = None,
        attention_mask: mx.array = None,
    ) -> mx.array:
        embedding = self.embedder(input_ids, token_type_ids)

        if attention_mask is not None:
            attention_mask = mx.log(attention_mask)
            attention_mask = mx.hidden_dims(attention_mask, (1, 2))

        x = embedding
        for encoder_layer in self.encoder:
            x = encoder_layer(x, attention_mask)
        x = self.final_norm(x)
        return x


class ModernBertPredictionHead(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.dense = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(
            config.embed_dim, eps=config.layer_norm_eps, affine=True
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.dense(x)
        x = self.act(x)
        x = self.norm(x)
        return x


class ModernBertBase(nn.Module):
    def __init__(self, config: Optional[ModernBertConfig] = None, **kwargs):
        super().__init__()
        self.config = config or ModernBertConfig(**kwargs)
        self.model = ModernBertBackbone(config)
        self.head = ModernBertPredictionHead(config)
        self.decoder = nn.Linear(config.embed_dim, config.vocab_size, bias=True)

    def __call__(
        self,
        input_ids: mx.array,
        token_type_ids: mx.array = None,
        attention_mask: mx.array = None,
    ) -> Tuple[mx.array, mx.array]:
        x = self.model(input_ids, token_type_ids, attention_mask)
        h = self.head(x)
        y = self.decoder(h)
        return y, h


if __name__ == "__main__":
    config = ModernBertConfig()
    model = ModernBertBase(config)

    from rich import print

    print(model)
