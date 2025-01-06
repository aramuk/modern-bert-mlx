import mlx.core as mx
from rich import print
import matplotlib.pyplot as plt

def abs_diff(A: mx.array, B: mx.array) -> mx.array:
    return (A - B).abs().sum()

def compare_embeddings():
    torch_outs = mx.load("logs/modernbert_torch/embeddings.safetensors")
    mlx_outs = mx.load("logs/modernbert_mlx/embeddings.safetensors")

    embeddings_torch = torch_outs["embeddings"]
    print(embeddings_torch.shape)
    embeddings_mlx = mlx_outs["embeddings"]
    print(embeddings_mlx.shape)
    print(f"Difference in embeddings: {abs_diff(embeddings_torch, embeddings_mlx).item():.6f}")


def main():
    compare_embeddings()

    torch_outs = mx.load("logs/modernbert_torch/backbone-out.safetensors")
    mlx_outs = mx.load("logs/modernbert_mlx/backbone-out.safetensors")

    for key in reversed(("last_hidden_state", "hidden_states", "attentions")):
        print(f"Comparing `{key}`...")
        if key.endswith("s"):
            for i, (out_torch, out_mlx) in enumerate(
                zip(torch_outs[key], mlx_outs[key])
            ):
                print(
                    out_torch.shape, out_torch.min(), out_torch.mean(), out_torch.max()
                )
                print(out_mlx.shape, out_mlx.min(), out_mlx.mean(), out_mlx.max())
                if key == "attentions":
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
                elif key == "hidden_states":
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

                assert (
                    diff := abs_diff(out_torch, out_mlx)
                ) <= 0.1, f"Layer {i}: `{key}` not equal {diff:.4f}"
        else:
            print(torch_outs[key].shape, torch_outs[key].mean())
            print(mlx_outs[key].shape, mlx_outs[key].mean())
            assert (
                diff := abs_diff(torch_outs[key], mlx_outs[key])
            ) <= 0.1, f"{key}: tensors not equal {diff:.4f}"

    print(torch_outs["last_hidden_state"].shape)
    print(mlx_outs["last_hidden_state"].shape)


if __name__ == "__main__":
    main()
