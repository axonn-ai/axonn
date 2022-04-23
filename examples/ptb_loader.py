# Copyright 2021 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from torch.utils.data import Dataset
import torch
import os


def init_vocab(path):
    with open(os.path.join(path, "ptb.train.txt"), "r") as f:
        lines = f.readlines()
    word2ind = {}
    for line in lines:
        for word in line.strip().split():
            if word not in word2ind:
                word2ind[word] = len(word2ind)
    word2ind["<eos>"] = len(word2ind)  # special eos token
    print(f"vocab size {len(word2ind)}")
    return word2ind


class ptb_dataset(Dataset):
    def __init__(self, path, seq_length, word2ind, split="train"):
        with open(os.path.join(path, f"ptb.{split}.txt"), "r") as f:
            lines = f.readlines()
        all_tokens = []
        for line in lines:
            tokens = [word2ind[word] for word in line.strip().split()]
            tokens.append(word2ind["<eos>"])
            all_tokens += tokens
        self.examples = []
        for i in range(0, len(all_tokens), seq_length + 1):
            batch = all_tokens[i : i + seq_length + 1]
            while len(batch) != seq_length + 1:
                batch.append(word2ind["<eos>"])
            self.examples.append(batch)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i]).long()


def get_collate_fn(seq_length=128):
    def collate_fn(batch):
        final_tensor = torch.zeros((len(batch), seq_length + 1), dtype=torch.long)
        mask = torch.ones((len(batch), seq_length + 1), dtype=torch.bool)
        for i, sent in enumerate(batch):
            length = min(sent.shape[0], seq_length + 1)
            final_tensor[i, :length] = sent[:length]
            mask[i, :length] = False
        return final_tensor

    return collate_fn


if __name__ == "__main__":
    dataset = ptb_dataset(
        "./dataset/PTB/", seq_length=128, word2ind=init_vocab("./dataset/PTB")
    )
    print(dataset[0])
