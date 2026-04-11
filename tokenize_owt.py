"""
Load and tokenize OpenWebText using GPT2 tokenizer.

Tokenized shards get saved as numpy arrays. Training shards will have shape (shard_size,) except for potentially the very
last training shard which will be bounded between `shard_size - val_tokens` and `2*shard_size - val_tokens`. We save a 
validation shard of the last 100k tokens (by default) as other people do this (see e.g. https://arxiv.org/abs/2603.21342).
Training shards will be saved to a "train" directory and the val shard will be saved in a "val" directory, these 
directories are subdirectories of --output_dir.

For testing purposes, run on a small subset (e.g. the first 1000 documents):
    python tokenize_owt.py --shard_size 131072 --subset 1000
"""
import os
import argparse
import numpy as np
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm
from multiprocessing import Pool
import time

def tokenize(doc):
    # Tokenizer must be initialized per-process to be safe
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error() # suppresses huggingface warnings about gpt2 model's maximum sequence length
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    ids = tokenizer.encode(doc["text"], max_length=None, truncation=False)
    return [tokenizer.eos_token_id] + ids # preprend eos token to each doc (prepend vs. append doesn't really matter)

def save_shard(tokens, shard_idx, split, save_dir):
    arr = np.array(tokens, dtype=np.uint16)
    path = os.path.join(save_dir, f"{split}_{shard_idx:05d}.npy")
    np.save(path, arr)
    print(f"saved {path}  ({len(arr):,} tokens)")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_size", type=int, default=1024 ** 3, help="tokens per train shard")
    parser.add_argument("--val_tokens", type=int, default=100_000,   help="tokens reserved for validation")
    parser.add_argument("--output_dir", type=str, default="./OWT",  help="output directory")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="number of worker processes")
    parser.add_argument("--chunk_size", type=int, default=16, help="chunk size for multiprocessing")
    parser.add_argument("--subset", type=int, default=None, help="number of docs to use (for testing)")
    args = parser.parse_args()
    if args.val_tokens > args.shard_size:
        raise ValueError(f"Expect val shard size to be less than or equal to train shard size, {args.shard_size=}, {args.val_tokens=}")
    print("ARGUMENTS:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    os.makedirs(args.output_dir, exist_ok=True)
    train_dir = os.path.join(args.output_dir, "train")
    val_dir = os.path.join(args.output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    print(f"\nCREATED DIRECTORIES:\n  TRAIN_DIR: {train_dir}\n  VAL_DIR: {val_dir}\n")

    t0 = time.time()
    dataset = load_dataset("openwebtext", split="train", trust_remote_code=True)
    t1 = time.time()
    print(f"DATASET LOADED IN {t1-t0:.2f} seconds\n")
    if args.subset is not None:
        dataset = dataset.select(range(args.subset))
        print(f"\n\n --- WARNING: ONLY USING THE FIRST {args.subset=} DOCUMENTS\n\n")

    print(f"BEGINNING TOKENIZATION WITH {args.num_workers} WORKERS...")
    t0 = time.time()
    shard = []
    shard_count = 0
    with Pool(args.num_workers) as pool:
        for ids in tqdm(pool.imap(tokenize, dataset, chunksize=args.chunk_size), total=len(dataset), desc="tokenizing"):
            shard.extend(ids)

            while len(shard) >= args.shard_size:
                save_shard(shard[:args.shard_size], shard_count, "train", train_dir)
                shard = shard[args.shard_size:]
                shard_count += 1

    # Handle the remainder and create the val shard
    remainder = np.array(shard)
    if shard_count > 0: # at least one shard has been saved
        previous_shard_path = os.path.join(train_dir, f"train_{shard_count-1:05d}.npy")
        previous_shard = np.load(previous_shard_path)
        remainder = np.concatenate([previous_shard, remainder])

    new_final_train = remainder[:-args.val_tokens] # could be an extra big shard, but that is fine
    val_shard = remainder[-args.val_tokens:]

    final_train_shard_idx = max(shard_count - 1, 0)
    save_shard(new_final_train, final_train_shard_idx, "train", train_dir) # overwrites old shard if it existed
    save_shard(val_shard, 0, "val", val_dir)
    t1 = time.time()
    print(f"CREATED {final_train_shard_idx+1} TRAINING SHARDS AND 1 VAL SHARD IN {t1-t0:.2f} seconds")