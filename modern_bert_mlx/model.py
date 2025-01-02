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
    norm_bias: bool = False
    # Encoder Args
    encoder_num_layers: int = 22
    encoder_drop_p: float = 0.0
    # Attention Args
    num_attention_heads: int = 12
    output_attn: bool = False
    # Decoder Args
    decoder_bias: bool = True


class ModernBertEmbedder(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.embed_dim)
        self.norm = nn.LayerNorm(
            config.embed_dim,
            eps=config.layer_norm_eps,
            affine=True,
            bias=config.norm_bias,
        )
        self.drop = nn.Dropout(p=config.embed_drop_p)

    def __call__(self, input_ids: mx.array) -> mx.array:
        # TODO: Conditionally run JIT implementation based on self.config.
        h = self.drop(self.norm(self.tok_embeddings(input_ids)))
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
        inp, gate = self.Wi(h).split(2, axis=-1)
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
    ) -> Tuple[mx.array, mx.array]:
        qkv = self.Wqkv(h)
        # print(qkv.shape)
        batch_size, seq_len, all_dims = qkv.shape
        # [batch_size, seq_len, 3, nheads,  head_dim]
        # See: https://github.com/ml-explore/mlx/discussions/1527
        qkv = qkv.reshape(batch_size, seq_len, 3, self.config.num_attention_heads, -1)
        # print(qkv.shape)

        Q, K, V = map(
            lambda a: mx.squeeze(a, axis=2),
            qkv.transpose(0, 3, 2, 1, 4).split(3, axis=2),
        )

        # TODO: use same rotation for Q, K
        # TODO: verify the dimension is correct
        # [batch_size, nheads, seqlen, clea]
        Q_rot = self.rotary_emb(Q)
        K_rot = self.rotary_emb(K)

        # TODO: select configured attention implementation dynamically
        # print(Q_rot.shape, K_rot.shape, V.shape, mask.shape)
        # print(Q_rot.shape, K_rot.shape, V.shape)
        # TODO: Use SDPA instead of eager attention
        head_dim = Q.shape[-1]
        scale = head_dim**-0.5
        attn_out = Q_rot @ K_rot.transpose(0, 1, 3, 2) * scale
        attn_out = mx.softmax(attn_out, axis=-1) @ V
        # attn_out = mx.fast.scaled_dot_product_attention(
        #     Q_rot, K_rot, V, mask=mask, stream=mx.gpu
        # )

        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        h = attn_out[0]
        h = self.drop(self.Wo(h))

        return (h, attn_out[1:]) if self.config.output_attn else (h,)


class ModernBertEncoderLayer(nn.Module):
    def __init__(self, config: ModernBertConfig, has_attn_norm: bool = True):
        super().__init__()
        self.config = config
        if has_attn_norm:
            self.attn_norm = nn.LayerNorm(
                config.embed_dim,
                config.layer_norm_eps,
                affine=True,
                bias=config.norm_bias,
            )
        else:
            self.attn_norm = nn.Identity()
        self.attn = ModernBertAttention(config)
        self.mlp_norm = nn.LayerNorm(
            config.embed_dim,
            config.layer_norm_eps,
            affine=True,
            bias=config.norm_bias,
        )
        self.mlp = ModernBertMLP(config)

    def __call__(self, h: mx.array, mask: mx.array) -> mx.array:
        attn_out = self.attn(h, mask, output_attn=self.config.output_attn)
        attn_out = self.attn_norm(h + attn_out[0])

        mlp_out = self.mlp(attn_out)
        mlp_out = self.mlp_norm(attn_out)
        return mlp_out, mask


class ModernBertBackbone(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.embedder = ModernBertEmbedder(config)
        self.encoder = ModuleList(
            [ModernBertEncoderLayer(config, has_attn_norm=False)]
            + [
                ModernBertEncoderLayer(config)
                for _ in range(1, config.encoder_num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(
            config.embed_dim,
            eps=config.layer_norm_eps,
            affine=True,
            bias=config.norm_bias,
        )

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array = None,
    ) -> mx.array:
        embedding = self.embedder(input_ids)

        h = embedding
        for encoder_layer in self.encoder:
            h, _ = encoder_layer(h, attention_mask)
        x = self.final_norm(h)
        return x


class ModernBertPredictionHead(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.fc = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(
            config.embed_dim,
            eps=config.layer_norm_eps,
            affine=True,
            bias=config.norm_bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc(x)
        x = self.act(x)
        x = self.norm(x)
        return x


class ModernBertBase(nn.Module):
    def __init__(self, config: Optional[ModernBertConfig] = None, **kwargs):
        super().__init__()
        self.config = config or ModernBertConfig(**kwargs)
        self.backbone = ModernBertBackbone(config)
        self.head = ModernBertPredictionHead(config)
        self.decoder = nn.Linear(
            config.embed_dim, config.vocab_size, bias=config.decoder_bias
        )

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array = None,
    ) -> Tuple[mx.array, mx.array]:
        # print(f"{input_ids.shape=}")
        # print(f"{input_ids.shape=}, {attention_mask.shape=}")
        x = self.backbone(input_ids, attention_mask)
        h = self.head(x)
        y = self.decoder(h)
        return y, h


if __name__ == "__main__":
    config = ModernBertConfig()
    model = ModernBertBase(config)

    model.load_weights("modernbert-mlx.safetensors")

    from rich import print

    print(model)
