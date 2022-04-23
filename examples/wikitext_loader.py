# Copyright 2021 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import transformers
import os
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
import torch


class lm_mmap_dataset(Dataset):
    def __init__(self, mmap_array, seq_length):
        self.mmap_array = mmap_array
        assert self.mmap_array.ndim == 1, "expect mmap array to be one-dimensional"
        self.seq_length = seq_length
        self.dataset_length = self.mmap_array.shape[0] // (self.seq_length + 1)
        assert self.dataset_length > 0, "try a smaller sequence length"

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        return torch.from_numpy(
            self.mmap_array[
                idx * (self.seq_length + 1) : (idx + 1) * (self.seq_length + 1)
            ]
        )


def wikitext_dataset(data_path, seq_length, split="train"):
    raw_data_path = os.path.join(data_path, f"wiki.{split}.tokens")
    processed_data_path = os.path.join(data_path, f"wiki.{split}.mmap")
    if not os.path.exists(processed_data_path):
        with open(raw_data_path, "r") as f:
            data = f.readlines()
        tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
        tokenized_data = np.array(
            [
                y
                for x in [
                    x
                    for x in [
                        tokenizer(line.strip())["input_ids"] for line in tqdm(data)
                    ]
                    if x
                ]
                for y in x
            ],
            dtype=np.int,
        )
        fp = np.memmap(
            processed_data_path, dtype=np.int, mode="w+", shape=tokenized_data.shape
        )
        print(fp.shape)
        fp[:] = tokenized_data[:]
        fp.flush()
    else:
        fp = np.memmap(processed_data_path, dtype=np.int, mode="r")
    return lm_mmap_dataset(fp, seq_length)


if __name__ == "__main__":
    dataset = wikitext_dataset(
        "/gpfs/alpine/csc452/scratch/ssingh37/axonn/examples/dataset/wikitext",
        seq_length=128,
        split="test",
    )
    print(len(dataset))
    print(dataset[0].shape)
    print(dataset[0])
