import mlx.core as mx
from rich import print


def main():
    torch_outs = mx.load("logs/modernbert_torch/backbone-out.safetensors")
    mlx_outs = mx.load("logs/modernbert_mlx/backbone-out.safetensors")

    def abs_diff(A: mx.array, B: mx.array) -> mx.array:
        return (A - B).abs().sum()

    for key in ("last_hidden_state", "hidden_states", "attentions"):
        print(f"Comparing `{key}`...")
        if key.endswith("s"):
            for i, (out_torch, out_mlx) in enumerate(
                zip(torch_outs[key], mlx_outs[key])
            ):
                assert (
                    diff := abs_diff(out_torch, out_mlx)
                ) <= 0.1, f"Layer {i}: tensors not equal {diff:.4f}"
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
