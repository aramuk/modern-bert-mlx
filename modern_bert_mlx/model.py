from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from pydantic import BaseModel

from modern_bert_mlx.data import create_4d_attention_mask_for_sdpa
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
    def __init__(self, config: ModernBertConfig, use_local_attention: bool):
        super().__init__()
        self.config = config
        self.dim = config.hidden_dim // config.attention.num_attention_heads
        self.base = (
            config.attention.local_rope_theta
            if use_local_attention
            else config.attention.global_rope_theta
        )
        self._inv_freq = 1.0 / (self.base ** (mx.arange(0, self.dim, 2) / self.dim))
        self._cached_cos: Optional[mx.array] = None
        self._cached_sin: Optional[mx.array] = None

    def _compute_rot(self, position_ids: mx.array, dtype: mx.Dtype):
        # inv_freq ~ [32], position_ids ~ [1, 9]
        seq_len = position_ids.shape[1]
        assert self._inv_freq.shape == (32,)
        assert position_ids.shape == (1, seq_len)

        # inv_freq ~ [1, 32, 1]
        inv_freq = self._inv_freq[None, :, None].astype(dtype)
        assert inv_freq.shape == (1, 32, 1)

        # position_ids ~ [1, 1, 9]
        position_ids = position_ids[:, None].astype(mx.float32)
        assert position_ids.shape == (1, 1, seq_len)

        # rot ~ [1, 9, 32]
        rot = (inv_freq @ position_ids).transpose(0, 2, 1)
        assert rot.shape == (1, seq_len, 32)

        # rot ~ [1, 9, 64]
        rot = mx.concatenate([rot, rot], axis=-1)
        assert rot.shape == (1, 9, 64)

        self._cached_cos = rot.cos().astype(dtype)
        self._cached_sin = rot.sin().astype(dtype)

    def _rotate_half(self, x: mx.array) -> mx.array:
        mid = x.shape[-1] // 2
        x1 = x[..., :mid]
        x2 = x[..., mid:]
        return mx.concatenate([-x2, x1], axis=-1)

    def __call__(self, x: mx.array, position_ids: mx.array, dtype: mx.Dtype):
        if (
            self._cached_cos is None
            or position_ids.shape[1] > self._cached_cos.shape[1]
        ):
            self._compute_rot(position_ids, dtype)

        cos, sin = self._cached_cos[:, None, :, :], self._cached_sin[:, None, :, :]
        assert cos.shape == sin.shape == (1, 1, position_ids.shape[1], 64)

        x_embed = x * cos + self._rotate_half(x) * sin
        assert x_embed.shape == x.shape
        return x_embed


class ModernBertAttention(nn.Module):
    def __init__(self, config: ModernBertConfig, layer_id: int):
        super().__init__()
        self.config = config
        self.use_local_attention = (
            layer_id % config.attention.global_attn_every_n_layers != 0
        )
        if self.use_local_attention:
            self.local_attention = (
                config.attention.local_attention // 2,
                config.attention.local_attention // 2,
            )
        else:
            self.local_attention = (-1, -1)
        self.nheads = self.config.attention.num_attention_heads
        self.proj_dim = 3 * config.hidden_dim
        self.head_dim = self.config.hidden_dim // self.nheads
        self.Wqkv = nn.Linear(config.hidden_dim, self.proj_dim, bias=False)
        self.rope = RoPE(config, use_local_attention=self.use_local_attention)
        self.Wo = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.drop = nn.Identity()

    def __call__(
        self,
        h: mx.array,
        attention_mask: mx.array,
        sliding_window_mask: mx.array,
        position_ids: mx.array,
        stream: mx.Stream,
    ) -> Tuple[mx.array, mx.array]:
        # TODO: use local/sliding window attention where appropriate
        if self.use_local_attention:
            attention_mask = sliding_window_mask

        batch_size, seq_len, _ = h.shape
        assert h.shape == (batch_size, seq_len, self.config.hidden_dim)
        qkv = self.Wqkv(h)
        assert qkv.shape == (batch_size, seq_len, self.proj_dim)
        # Using reshape instead of view in MLX because of unified memory:
        #   https://github.com/ml-explore/mlx/discussions/1527
        # qkv: [batch_size, seq_len, 3, nheads,  head_dim]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.nheads, -1)
        assert qkv.shape == (batch_size, seq_len, 3, self.nheads, self.head_dim)
        # Q, K, V: [batch_size, seq_len, nheads,  head_dim]
        Q, K, V = map(
            lambda a: mx.squeeze(a, axis=2),
            qkv.transpose(0, 3, 2, 1, 4).split(3, axis=2),
        )
        assert all(
            a.shape == (batch_size, self.nheads, seq_len, self.head_dim)
            for a in (Q, K, V)
        )
        # [batch_size, nheads, seq_len, head_dim]
        Q_embed = self.rope(Q, position_ids=position_ids, dtype=Q.dtype)
        K_embed = self.rope(K, position_ids=position_ids, dtype=Q.dtype)
        assert all(
            a.shape == (batch_size, self.nheads, seq_len, self.head_dim)
            for a in (Q_embed, K_embed, V)
        )

        # TODO: select configured attention implementation dynamically
        # softmax((Q_embed @ K_embed).transpose(2, 3) * scale + mask) @ V + mask
        attn_out = mx.fast.scaled_dot_product_attention(
            Q_embed,
            K_embed,
            V,
            scale=self.head_dim**-0.5,
            mask=attention_mask,
            stream=stream,
        )
        assert attn_out.shape == (batch_size, self.nheads, seq_len, self.head_dim)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        assert attn_out.shape == (batch_size, seq_len, self.config.hidden_dim)

        h = attn_out[0]
        assert h.shape == (seq_len, self.config.hidden_dim)
        h = self.drop(self.Wo(h))
        assert h.shape == (seq_len, self.config.hidden_dim)

        return (h, attn_out[1:]) if self.config.attention.output_attn else (h,)


class ModernBertEncoderLayer(nn.Module):
    def __init__(
        self, config: ModernBertConfig, layer_id: int = 0, has_attn_norm: bool = True
    ):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        if has_attn_norm:
            self.attn_norm = nn.LayerNorm(
                config.hidden_dim,
                config.layer_norm_eps,
                affine=True,
                bias=config.norm_bias,
            )
        else:
            self.attn_norm = nn.Identity()
        self.attn = ModernBertAttention(config, layer_id=self.layer_id)
        self.mlp_norm = nn.LayerNorm(
            config.hidden_dim,
            config.layer_norm_eps,
            affine=True,
            bias=config.norm_bias,
        )
        self.mlp = ModernBertMLP(config)

    def __call__(
        self,
        h: mx.array,
        attention_mask: mx.array,
        sliding_window_mask: mx.array,
        position_ids: mx.array,
        stream: mx.Stream,
    ) -> mx.array:
        attn_out = self.attn(
            h,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            stream=stream,
        )
        attn_out = self.attn_norm(h + attn_out[0])

        mlp_out = self.mlp(attn_out)
        mlp_out = self.mlp_norm(attn_out)
        return mlp_out, attention_mask


class ModernBertBackbone(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.embedder = ModernBertEmbedder(config)
        self.encoder = ModuleList(
            [ModernBertEncoderLayer(config, has_attn_norm=False, layer_id=0)]
            + [
                ModernBertEncoderLayer(config, layer_id=i)
                for i in range(1, config.encoder_num_layers)
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
        attention_mask: mx.array,
        sliding_window_mask: mx.array,
        position_ids: mx.array,
        stream: mx.Stream,
    ) -> mx.array:
        embedding = self.embedder(input_ids)

        h = embedding
        all_hidden_states = tuple()
        all_self_attentions = tuple()
        for encoder_layer in self.encoder:
            h, attn = encoder_layer(
                h,
                attention_mask=attention_mask,
                sliding_window_mask=sliding_window_mask,
                position_ids=position_ids,
                stream=stream,
            )
            all_hidden_states += (h,)
            all_self_attentions += (attn,)
        x = self.final_norm(h)

        # TODO: conditionally pad output

        return {
            "last_hidden_state": x,
            "hidden_states": mx.array(all_hidden_states),
            "attentions": mx.array(all_self_attentions),
        }


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

    def _update_attention_mask(
        self, attention_mask: mx.array, stream: mx.Stream
    ) -> mx.array:
        global_attention_mask = create_4d_attention_mask_for_sdpa(
            attention_mask, attention_mask.dtype
        )

        # Create position indices
        rows = mx.arange(
            global_attention_mask.shape[2], dtype=mx.uint16, stream=stream
        )[None, :]
        # Calculate distance between positions
        distance = mx.abs(rows - rows.T)

        # Create sliding window mask (1 for positions within window, 0 outside)
        window_mask = (distance <= self.config.attention.local_attention // 2)[
            None, None, :
        ]
        # Combine with existing mask
        # TODO: switch to mx.finfo once available.
        # print(window_mask.shape, window_mask.dtype)
        # print(global_attention_mask.shape, global_attention_mask.dtype)
        sliding_window_mask = mx.where(
            mx.logical_not(window_mask),
            global_attention_mask,
            float(np.finfo(np.float32).min),
            stream=stream,
        )

        return global_attention_mask, sliding_window_mask

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        sliding_window_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        stream: mx.Stream = mx.gpu,
    ) -> Tuple[mx.array, mx.array]:
        batch_size, seq_len = input_ids.shape[:2]
        if position_ids is None:
            position_ids = mx.arange(seq_len, stream=stream).reshape(1, -1)

        if attention_mask is None:
            attention_mask = mx.ones(
                (batch_size, seq_len), dtype=mx.uint16, stream=stream
            )

        attention_mask, sliding_window_mask = self._update_attention_mask(
            attention_mask, stream
        )

        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            stream=stream,
        )
        mx.save_safetensors("logs/modernbert_mlx/backbone-out.safetensors", outputs)
        
        x = outputs["last_hidden_state"]
        h = self.head(x)
        y = self.decoder(h)
        return y, h


if __name__ == "__main__":
    config = ModernBertConfig()
    model = ModernBertBase(config)

    model.load_weights("modernbert-mlx.safetensors")

    from rich import print

    print(model)
