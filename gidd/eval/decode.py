import sys
import json

import torch
from transformers import AutoTokenizer


def main():
    path = sys.argv[1]

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    zs = torch.load(path, weights_only=True)

    xs = tokenizer.batch_decode(zs, skip_special_tokens=True)

    print(json.dumps(xs, indent=2))


if __name__ == "__main__":
    main()
