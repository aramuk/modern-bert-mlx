import argparse
import sys

import mlx.core as mx
import numpy as np
import torch
from rich import print
from safetensors.numpy import save_file
from transformers import AutoTokenizer, AutoModelForMaskedLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display weights of a HuggingFace transformers model"
    )
    parser.add_argument("-m", "--model-id", help="ID of model on HuggingFace hub.")
    parser.add_argument(
        "--operation",
        help="How to inspect the model",
        choices=["display", "dump_weights"],
    )
    return parser.parse_args(sys.argv[1:])


def dump_weights(model: AutoModelForMaskedLM):
    weight_path = "model.safetensors"
    data = mx.load(weight_path)
    migrated = {}
    for key in data:
        migrated_key = key
        migrated_key = migrated_key.replace('.layers', '.encoder')
        migrated_key = migrated_key.replace('.embeddings', '.embedder')
        migrated_key = migrated_key.replace('.dense.', '.fc.')
        migrated_key = migrated_key.replace('model.', 'backbone.')
        migrated[migrated_key] = np.array(data[key], copy=False)

    migrated['decoder.weight'] = model.decoder.weight.detach().numpy()
    # print(model.decoder.weight.shape)

    # print(*sorted((k, v.shape) for k,v in data.items()), sep='\n')
    save_file(migrated, 'modernbert-mlx.safetensors')

def main(args: argparse.Namespace):
    model = AutoModelForMaskedLM.from_pretrained(args.model_id)

    match args.operation:
        case "dump_weights":
            dump_weights(model)
        case "display" | _:
            print(model)


if __name__ == "__main__":
    args = parse_args()
    main(args)
