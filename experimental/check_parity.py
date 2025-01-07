from pathlib import Path
from typing import List

import mlx.core as mx
from rich import print
import matplotlib.pyplot as plt


def abs_diff(A: mx.array, B: mx.array) -> mx.array:
    return (A - B).abs().sum()


TORCH_SAVEDIR = Path("logs/modernbert_torch")
MLX_SAVEDIR = Path("logs/modernbert_mlx")


def compare(filename: str, keys: List[str]):
    torch_outs = mx.load(str(TORCH_SAVEDIR / f"{filename}.safetensors"))
    mlx_outs = mx.load(str(MLX_SAVEDIR / f"{filename}.safetensors"))

    for key in keys:
        tensor_torch = torch_outs[key]
        tensor_mlx = mlx_outs[key]
        assert (
            tensor_torch.shape == tensor_mlx.shape
        ), f"tensor_torch['{key}'] {tensor_torch.shape} does not match shape of  tensor_mlx['{key}'] {tensor_mlx.shape}"

        print(f"Difference in `{key}`: {abs_diff(tensor_torch, tensor_mlx).item():.9f}")

        # if key == "sliding_window_mask":
        #     print(tensor_torch.min(), tensor_torch.mean(), tensor_torch.max())
        #     # print(tensor_torch)

        #     print(tensor_mlx.min(), tensor_mlx.mean(), tensor_mlx.max())
        #     # print(tensor_mlx)

        #     global_attention_mask = mlx_outs["attention_mask"]
        #     # print(global_attention_mask.dtype)
        #     # Create position indices
        #     rows = mx.arange(
        #         global_attention_mask.shape[2],
        #         dtype=mx.int32,
        #     )[None, :]
        #     # Calculate distance between positions
        #     distance = mx.abs(rows - rows.T)
        #     # print(distance)
        #     # Create sliding window mask (1 for positions within window, 0 outside)
        #     window_mask = (distance <= 128 // 2)[None, None, :]
        #     # print(window_mask)
        #     # Combine with existing mask
        #     # TODO: switch to mx.finfo once available.
        #     # print(window_mask.shape, window_mask.dtype)
        #     # print(global_attention_mask.shape, global_attention_mask.dtype)
        #     sliding_window_mask = mx.where(
        #         mx.logical_not(window_mask),
        #         global_attention_mask,
        #         0,
        #         # float(np.finfo(np.float32).min),
        #     )
        #     # print(sliding_window_mask)

        #     print(abs_diff(tensor_torch, sliding_window_mask))


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
    compare("embeddings", ["embeddings"])
    compare("masks", ["attention_mask", "sliding_window_mask"])

    torch_outs = mx.load(str(TORCH_SAVEDIR / "backbone-out.safetensors"))
    mlx_outs = mx.load(str(MLX_SAVEDIR / "backbone-out.safetensors"))

    for key in reversed(("last_hidden_state", "hidden_states", "attentions")):
        print(f"Comparing `{key}`...")
        if key.endswith("s"):
            for i, (out_torch, out_mlx) in enumerate(
                zip(torch_outs[key], mlx_outs[key])
            ):
                # print(
                #     out_torch.shape, out_torch.min(), out_torch.mean(), out_torch.max()
                # )
                # print(out_mlx.shape, out_mlx.min(), out_mlx.mean(), out_mlx.max())
                # if key == "attentions":
                #     display_attentions(out_torch, out_mlx)
                # elif key == "hidden_states":
                #     display_hidden_states(out_torch, out_mlx)

                diff = abs_diff(out_torch, out_mlx)
                print(f"Layer {i}, `{key}`: diff = {diff:.9f}")
                # assert (
                #     diff := abs_diff(out_torch, out_mlx)
                # ) <= 0.1, f"Layer {i}: `{key}` not equal {diff:.4f}"
        else:
            # print(torch_outs[key].shape, torch_outs[key].mean())
            # print(mlx_outs[key].shape, mlx_outs[key].mean())
            print(f"`{key}`: diff = {diff:.9f}")
            # assert (
            #     diff := abs_diff(torch_outs[key], mlx_outs[key])
            # ) <= 0.1, f"{key}: tensors not equal {diff:.4f}"


if __name__ == "__main__":
    main()
