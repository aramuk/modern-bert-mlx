import argparse
import sys

from rich import print
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display weights of a HuggingFace transformers model"
    )
    parser.add_argument("-m", "--model-id", help="ID of model on HuggingFace hub.")
    return parser.parse_args(sys.argv[1:])


def main(args: argparse.Namespace):
    # tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForMaskedLM.from_pretrained(args.model_id)
    
    print(model)

if __name__ == "__main__":
    args = parse_args()
    main(args)
