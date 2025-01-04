from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from pydantic import BaseModel

from modern_bert_mlx.nn import ModuleList
from modern_bert_mlx.config import ModernBertConfig


class ModernBertEmbedder(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.norm = nn.LayerNorm(
            config.hidden_dim,
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
        self.Wi = nn.Linear(
            config.hidden_dim, 2 * config.intermediate_dim, bias=config.mlp_bias
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout(config.mlp_dropout)
        self.Wo = nn.Linear(
            config.intermediate_dim, config.hidden_dim, bias=config.mlp_bias
        )

    def __call__(self, h: mx.array) -> mx.array:
        inp, gate = self.Wi(h).split(2, axis=-1)
        out = self.Wo(self.drop(self.act(inp) * gate))
        return out


class RoPE(nn.Module):
    def __init__(self, config: ModernBertConfig, is_global: bool = False):
        super().__init__()
        self.config = config
        self.dim = config.hidden_dim // config.attention.num_attention_heads
        self.base = (
            config.attention.global_rope_theta
            if is_global
            else config.attention.local_rope_theta
        )
        self._inv_freq = 1.0 / (self.base ** (mx.arange(0, self.dim, 2) / self.dim))
        self._cached_cos: Optional[mx.array] = None
        self._cached_sin: Optional[mx.array] = None

    def _compute_rot(self, position_ids: mx.array, dtype: mx.Dtype):
        print(self._inv_freq.shape, position_ids.shape)
        inv_freq = mx.broadcast_to(
            self._inv_freq[:, None, None],
            shape=(self._inv_freq.shape[0], position_ids.shape[1], 1),
        ).astype(dtype)
        position_ids = position_ids[:, :].astype(mx.float32)
        print(f"{inv_freq.shape=}, {position_ids.shape=}")
        rot = inv_freq @ position_ids
        print(f"{rot.shape=}")
        assert rot.shape == (
            position_ids.shape,
            self.dim // 2,
            self.dim // 2,
        ), f"rotation of shape [seq_len, head_dim, head_dim] has shape {rot.shape}"
        print(f"{rot.shape=}")
        rot = mx.concatenate([rot, rot], axis=-1)

        self._cached_cos = rot.cos().astype(dtype)
        self._cached_sin = rot.sin().astype(dtype)

    def __call__(self, x: mx.array, position_ids: mx.array, dtype: mx.Dtype):
        if self._cached_cos is None or position_ids.shape[1] > cos.shape[1]:
            self._compute_rot(position_ids, dtype)
        cos, sin = self._cached_cos, self._cached_sin

        print(f"{cos.shape=}, {sin.shape=}")
        # cos = mx.expand_dims(cos, axis=-1)
        # sin = mx.expand_dims(sin, axis=-1)
        # print(f"{cos.shape=}, {sin.shape=}")
        print(f"{x.shape=}")
        mid = x.shape[-1] // 2
        x1 = x[..., :mid]
        x2 = x[..., mid:]
        embed = mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin])
        return embed


class ModernBertAttention(nn.Module):
    def __init__(self, config: ModernBertConfig, is_global: bool = False):
        super().__init__()
        self.config = config
        self.is_global = is_global
        self.nheads = self.config.attention.num_attention_heads
        self.head_dim = self.config.hidden_dim // self.nheads
        self.Wqkv = nn.Linear(config.hidden_dim, 3 * config.hidden_dim, bias=False)
        self.rope = RoPE(config, self.is_global)
        self.Wo = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.drop = nn.Identity()

    def __call__(
        self,
        h: mx.array,
        mask: mx.array,
        position_ids: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        qkv = self.Wqkv(h)
        print(qkv.shape)
        batch_size, seq_len, all_dims = qkv.shape
        print([batch_size, seq_len, 3, self.nheads, self.head_dim])
        # RoPE
        # cos, sin = get_rope(
        #     position_ids=position_ids,
        #     dim=self.head_dim,
        #     base=(
        #         self.config.attention.global_rope_theta
        #         if self.is_global
        #         else self.config.attention.local_rope_theta
        #     ),
        #     dtype=qkv.dtype,
        # )
        # [batch_size, seq_len, 3, nheads,  head_dim]
        # See: https://github.com/ml-explore/mlx/discussions/1527
        qkv = qkv.reshape(batch_size, seq_len, 3, self.nheads, -1)
        # print(qkv.shape)
        Q, K, V = map(
            lambda a: mx.squeeze(a, axis=2),
            qkv.transpose(0, 3, 2, 1, 4).split(3, axis=2),
        )

        # TODO: use same rotation for Q, K
        # TODO: verify the dimension is correct
        # [batch_size, nheads, seqlen]
        Q_embed = self.rope(Q, position_ids=position_ids, dtype=Q.dtype)
        K_embed = self.rope(K, position_ids=position_ids, dtype=Q.dtype)

        # TODO: select configured attention implementation dynamically
        # print(Q_rot.shape, K_rot.shape, V.shape, mask.shape)
        # print(Q_rot.shape, K_rot.shape, V.shape)
        # TODO: Use SDPA instead of eager attention
        # TODO: sliding window attention
        # TODO: mask attn_out
        head_dim = Q.shape[-1]
        scale = head_dim**-0.5
        attn_out = Q_embed @ K_embed.transpose(0, 1, 3, 2) * scale
        print(attn_out.shape)
        attn_out = mx.softmax(attn_out, axis=-1) @ V
        print(attn_out.shape)
        # attn_out = mx.fast.scaled_dot_product_attention(
        #     Q_rot, K_rot, V, mask=mask, stream=mx.gpu
        # )

        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        h = attn_out[0]
        h = self.drop(self.Wo(h))

        return (h, attn_out[1:]) if self.config.attention.output_attn else (h,)


class ModernBertEncoderLayer(nn.Module):
    def __init__(
        self, config: ModernBertConfig, layer_id: int = 0, has_attn_norm: bool = True
    ):
        super().__init__()
        self.config = config
        if has_attn_norm:
            self.attn_norm = nn.LayerNorm(
                config.hidden_dim,
                config.layer_norm_eps,
                affine=True,
                bias=config.norm_bias,
            )
        else:
            self.attn_norm = nn.Identity()
        self.attn = ModernBertAttention(config)
        self.mlp_norm = nn.LayerNorm(
            config.hidden_dim,
            config.layer_norm_eps,
            affine=True,
            bias=config.norm_bias,
        )
        self.mlp = ModernBertMLP(config)

    def __call__(
        self, h: mx.array, mask: mx.array, position_ids: Optional[mx.array] = None
    ) -> mx.array:
        seq_len = h.shape[1]
        if position_ids is None:
            position_ids = mx.arange(seq_len).reshape(1, seq_len)
        attn_out = self.attn(h, mask, position_ids)
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
            config.hidden_dim,
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
        self.fc = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(
            config.hidden_dim,
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
            config.hidden_dim, config.vocab_size, bias=config.decoder_bias
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
