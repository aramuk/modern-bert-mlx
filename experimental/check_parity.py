from pathlib import Path
from typing import List, Tuple

import mlx.core as mx
from rich import print
import matplotlib.pyplot as plt


def abs_diff(A: mx.array, B: mx.array) -> mx.array:
    return (A - B).abs().mean()


TORCH_SAVEDIR = Path("logs/modernbert_torch")
MLX_SAVEDIR = Path("logs/modernbert_mlx")


def compare_attn():
    Q, K, V = mx.random.normal((3, 1, 12, 9, 64))
    mask = mx.zeros((1, 9, 9))

    scale = 64**-0.5
    attn_weights = (Q @ K.transpose(0, 1, 3, 2)) * scale
    attn_weights = mx.softmax(attn_weights + mask, axis=-1)
    eager_out = attn_weights @ V

    sdpa_out = mx.fast.scaled_dot_product_attention(
        q=Q, k=K, v=V, scale=scale, mask=mask
    )

    print(f"diff in attention implementations: {abs_diff(eager_out, sdpa_out)}")


def compare(filename: str, keys: List[str]) -> Tuple[mx.array, mx.array]:
    torch_outs = mx.load(str(TORCH_SAVEDIR / f"{filename}.safetensors"))
    mlx_outs = mx.load(str(MLX_SAVEDIR / f"{filename}.safetensors"))

    for key in keys:
        tensor_torch = torch_outs[key]
        tensor_mlx = mlx_outs[key]
        assert (
            tensor_torch.shape == tensor_mlx.shape
        ), f"tensor_torch['{key}'] {tensor_torch.shape} does not match shape of  tensor_mlx['{key}'] {tensor_mlx.shape}"

        print(f"Difference in `{key}`: {abs_diff(tensor_torch, tensor_mlx).item():.9f}")
    return torch_outs, mlx_outs


def compare_embeddings():
    torch_outs, mlx_outs = compare("embeddings", ["embeddings"])
    torch_emb = torch_outs["embeddings"]
    mlx_emb = mlx_outs["embeddings"]
    print(torch_emb.shape, mlx_emb.shape)
    print(torch_emb.min(), torch_emb.mean(), torch_emb.max())
    print(mlx_emb.min(), mlx_emb.mean(), mlx_emb.max())
    batch_size, seq_len, hidden_dim = torch_emb.shape
    fig, ax = plt.subplots(3, 9, figsize=(10, 5))
    ax = ax.T
    for i in range(seq_len):
        im = ax[i, 0].imshow(torch_emb[0, i].reshape(32, 24))
        ax[i, 0].set_xticks([])
        ax[i, 0].set_yticks([])
        im = ax[i, 1].imshow(mlx_emb[0, i].reshape(32, 24))
        ax[i, 1].set_xticks([])
        ax[i, 1].set_yticks([])
        im = ax[i, 2].imshow((torch_emb[0, i] - mlx_emb[0, i]).reshape(32, 24))
        ax[i, 2].set_xticks([])
        ax[i, 2].set_yticks([])
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.9, 0.2, 0.025, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()
    # for i, (mlx_i, torch_i) in enumerate(zip(mlx_emb.flatten(), torch_emb.flatten())):
    #     assert abs(mlx_i - torch_i) < 1e-7, f"{i}: {mlx_i} ({mlx_i.item()}) - {torch_i} ({torch_i.item()}) = {mlx_i - torch_i}"


def display_hidden_states(out_torch: mx.array, out_mlx: mx.array):
    batch_size, seq_len, hidden_dim = out_torch.shape
    fig, ax = plt.subplots(2, 1, figsize=(15, 2))
    im = ax[0].imshow(out_torch.reshape(seq_len, hidden_dim))
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    im = ax[1].imshow(out_mlx.reshape(seq_len, hidden_dim))
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.95, 0.2, 0.025, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()


def display_attentions(out_torch: mx.array, out_mlx: mx.array):
    nheads = out_torch.shape[1]
    seq_len = out_torch.shape[2]
    fig, ax = plt.subplots(2, nheads, figsize=(10, 5))
    for i in range(nheads):
        im = ax[0, i].imshow(out_torch[:, i, :, :].reshape(seq_len, seq_len))
        ax[0, i].set_xticks([])
        ax[0, i].set_yticks([])
        im = ax[1, i].imshow(out_mlx[:, i, :, :].reshape(seq_len, seq_len))
        ax[1, i].set_xticks([])
        ax[1, i].set_yticks([])
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.95, 0.2, 0.025, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()


def main():
    compare_attn()
    compare_embeddings()

    compare("masks", ["attention_mask", "sliding_window_mask"])

    torch_outs = mx.load(str(TORCH_SAVEDIR / "backbone-out.safetensors"))
    mlx_outs = mx.load(str(MLX_SAVEDIR / "backbone-out.safetensors"))

    for key in reversed(("last_hidden_state", "hidden_states", "attentions")):
        print(f"Comparing `{key}`...")
        if key.endswith("s"):
            if mx.isnan(mlx_outs[key]).all():
                continue
            for i, (out_torch, out_mlx) in enumerate(
                zip(torch_outs[key], mlx_outs[key])
            ):
                assert (
                    diff := abs_diff(out_torch, out_mlx)
                ) <= 0.1, f"Layer {i}: `{key}` not equal {diff:.4f}"
                print(f"Layer {i}, `{key}`: diff = {diff:.9f}")
        else:
            assert (
                diff := abs_diff(torch_outs[key], mlx_outs[key])
            ) <= 0.1, f"{key}: tensors not equal {diff:.4f}"
            print(f"`{key}`: diff = {diff:.9f}")


if __name__ == "__main__":
    main()
