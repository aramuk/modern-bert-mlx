import mlx.core as mx
import numpy as np
from rich import print
from transformers import AutoTokenizer, AutoModelForMaskedLM

from modern_bert_mlx import ModernBertBase, ModernBertConfig
from modern_bert_mlx.data import create_4d_attention_mask_for_sdpa


def main():
    model_id = "answerdotai/modernbert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    text = "The capital of France is [MASK]."
    inputs = tokenizer(text, return_tensors="np")
    inputs["input_ids"] = mx.array(inputs["input_ids"])
    # TODO: get masks working
    inputs["attention_mask"] = (
        None  # create_4d_attention_mask_for_sdpa(inputs["attention_mask"], dtype=np.float64)
    )
    # print(inputs["attention_mask"].dtype)
    print(inputs)

    def decode(logits):
        masked_index = inputs["input_ids"][0].tolist().index(tokenizer.mask_token_id)
        predicted_token_id = logits[0, masked_index].argmax(axis=-1)
        predicted_token = tokenizer.decode(predicted_token_id)
        return predicted_token

    print("ModernBertMLX says...")
    config = ModernBertConfig()
    model = ModernBertBase(config)
    model.load_weights("modernbert-mlx.safetensors")
    model.eval()
    with mx.stream(mx.gpu) as s:
        y, h = model(**inputs, stream=s)
    print(y.abs().sum())

    predicted_token = decode(np.array(y))
    print(text.replace("[MASK]", f"[red]{predicted_token}[/red]"))

    # model = AutoModelForMaskedLM.from_pretrained(model_id)
    # model.config.output_hidden_states = True
    # model.config.output_attentions = True

    # inputs = tokenizer(text, return_tensors="pt")
    # print("ModernBert says...")
    # outputs = model(**inputs)
    # print(mx.mean((outputs.logits - y) ** 2))
    # predicted_token = decode(outputs.logits)
    # print(text.replace("[MASK]", f"[red]{predicted_token}[/red]"))


if __name__ == "__main__":
    main()
