"""Tokenize TinyStories into uint16 GPT-2 token files: <out>/train.bin and <out>/val.bin."""
import argparse
import glob
import os

import numpy as np
import pyarrow.parquet as pq
import tiktoken
from huggingface_hub import snapshot_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/tinystories")
    args = ap.parse_args()
    root = snapshot_download("roneneldan/TinyStories", repo_type="dataset", allow_patterns=["data/*.parquet"])
    enc = tiktoken.get_encoding("gpt2")
    os.makedirs(args.out, exist_ok=True)
    for split, name in (("validation", "val.bin"), ("train", "train.bin")):
        path, n = os.path.join(args.out, name), 0
        with open(path, "wb") as f:
            for fp in sorted(glob.glob(os.path.join(root, "data", f"{split}-*.parquet"))):
                texts = pq.read_table(fp, columns=["text"]).column("text").to_pylist()
                for i in range(0, len(texts), 20000):
                    docs = enc.encode_ordinary_batch(texts[i:i + 20000], num_threads=32)
                    arr = np.concatenate([np.array(d + [enc.eot_token], dtype=np.uint16) for d in docs])
                    f.write(arr.tobytes())
                    n += arr.size
        print(f"{path}: {n:,} tokens", flush=True)


if __name__ == "__main__":
    main()
