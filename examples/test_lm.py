# Copyright 2021 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from axonn import axonn as ax
from axonn import optim
from external.models.nvidia_transformer import DistributedGPT
from external.dataloaders.megatron_loader import wikitext_dataloader
import torch
from tqdm import tqdm
import torch.nn as nn
from ptb_loader import ptb_dataset, init_vocab
from wikitext_loader import wikitext_dataset
import numpy as np
import argparse
import time
import os


def create_dataset(dataset_name, seq_len):
    if dataset_name == "ptb":
        word2ind = init_vocab("./examples/dataset/PTB")
        train_dataset = ptb_dataset(
            "./examples/dataset/PTB/",
            seq_length=seq_len,
            word2ind=word2ind,
            split="train",
        )
        val_dataset = ptb_dataset(
            "./examples/dataset/PTB/",
            seq_length=seq_len,
            word2ind=word2ind,
            split="valid",
        )
        test_dataset = ptb_dataset(
            "./examples/dataset/PTB/",
            seq_length=seq_len,
            word2ind=word2ind,
            split="test",
        )
    elif dataset_name == "wikitext":
        train_dataset = wikitext_dataset(
            "./examples/dataset/wikitext/", seq_length=seq_len, split="train"
        )
        val_dataset = wikitext_dataset(
            "./examples/dataset/wikitext/", seq_length=seq_len, split="valid"
        )
        test_dataset = wikitext_dataset(
            "./examples/dataset/wikitext/", seq_length=seq_len, split="test"
        )
    return train_dataset, val_dataset, test_dataset


def create_megatron_loader(batch_size, seq_length, num_workers):
    summit_home = "/gpfs/alpine/csc452/scratch/ssingh37/"
    data_prefix = os.path.join(summit_home, "gpt2_data/wikitext_103_text_document")
    merge_file = os.path.join(summit_home, "gpt2_data/gpt2-merges.txt")
    vocab_file = os.path.join(summit_home, "gpt2_data/gpt2-vocab.json")
    dp_rank = ax.config.data_parallel_rank
    dp_size = ax.config.G_data
    batch_size = batch_size
    seq_len = seq_length
    num_workers = num_workers
    return wikitext_dataloader(
        data_prefix,
        merge_file,
        vocab_file,
        dp_rank,
        dp_size,
        batch_size,
        seq_len,
        num_workers,
    )


def init_weights(module):
    """Initialize the weights."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.01)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def get_loss_fn():
    criterion = torch.nn.CrossEntropyLoss()

    def loss_fn(logits, labels):
        return criterion(logits.view(-1, vocab_size), labels.view(-1))

    return loss_fn


def run_epoch(dataloader, optimizer, eval_mode=False):
    epoch_loss = 0
    start = time.time()
    for sent in tqdm(
        dataloader,
        disable=not (ilp_rank == 0 and ax.config.data_parallel_rank == 0),
    ):
        if not eval_mode:
            optimizer.zero_grad()
        src, trg = sent, sent
        if ilp_rank == 0:
            src = src["text"][:, :-1].cuda()
            trg = trg["text"][:, 1:].cuda()
        if G_inter > 1:
            if ilp_rank == 0:
                ax.comm_handle.send(trg, G_inter - 1, tag=0, async_op=False)
            elif ilp_rank == G_inter - 1:
                trg = torch.cuda.LongTensor(len(trg), seq_len)
                ax.comm_handle.recv(trg, 0, tag=0, async_op=False)
        epoch_loss += ax.run_batch(src, trg, eval_mode=eval_mode)
        if ilp_rank == 0:
            ax.print_status(f"Compute time = {time.time() - start}s")
        if not eval_mode:
            optimizer.step()
        end = time.time()
        if ilp_rank == 0:
            ax.print_status(f"Batch time = {end-start} s")
        start = time.time()
    return epoch_loss / len(dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--G-inter", help="degree of inter-layer parallelism", type=int, default=1
    )
    parser.add_argument(
        "--G-data", help="degree of data parallelism", type=int, default=1
    )
    parser.add_argument(
        "--micro-batch-size", help="micro batch size", type=int, default=4
    )
    parser.add_argument("--batch-size", help="global batch size", type=int, default=64)
    parser.add_argument(
        "--epochs", help="number of training epochs", type=int, default=20
    )
    parser.add_argument(
        "--sequence-length",
        help="sequence length of each data point",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--dataset",
        help="which dataset to use",
        choices=["wikitext", "ptb"],
        type=str,
        default="wikitext",
    )
    parser.add_argument(
        "-N",
        "--num-hidden-layers",
        help="number of transformer layers",
        type=int,
        default=24,
    )
    parser.add_argument(
        "-D", "--hidden-size", help="hidden size of transformer", type=int, default=1024
    )
    parser.add_argument(
        "-H",
        "--num-attention_heads",
        help="number of attention heads",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--cpu-offload",
        help="offload fp32 params, fp32 grads and optimizer states to cpu",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    num_epochs = args.epochs
    cpu_offload = args.cpu_offload
    N, D, H = args.num_hidden_layers, args.hidden_size, args.num_attention_heads
    seq_len = args.sequence_length
    dataset = args.dataset
    bs = args.batch_size
    mbs = args.micro_batch_size

    ax.init(
        G_data=args.G_data,
        G_inter=args.G_inter,
        mixed_precision=True,
        fp16_allreduce=True,
        cpu_offload=cpu_offload,
    )

    ilp_rank = ax.config.inter_layer_parallel_rank
    G_inter = ax.config.G_inter
    train_dataset, val_dataset, test_dataset = create_dataset(dataset, seq_len)

    if dataset == "wikitext":
        vocab_size, num_workers, lr = (
            51200,
            0,
            1e-3,
        )  # wikitext-103 dataset does not work with multiple workers
    else:
        vocab_size, num_workers, lr = 10000, 2, 1e-4

    if ilp_rank != 0:
        train_loader = ax.create_dataloader(
            train_dataset, batch_size=bs, micro_batch_size=mbs, num_workers=num_workers
        )

    else:
        train_loader = create_megatron_loader(bs // ax.config.G_data, seq_len, 2)

    val_loader = ax.create_dataloader(
        val_dataset, batch_size=bs, micro_batch_size=mbs, num_workers=num_workers
    )

    test_loader = ax.create_dataloader(
        test_dataset, batch_size=bs, micro_batch_size=mbs, num_workers=num_workers
    )

    model = DistributedGPT(
        N, D, H, vocab_size=vocab_size, seq_len=seq_len, ckp_coeff=4
    ).cuda()

    model.apply(init_weights)

    if cpu_offload:
        optimizer = optim.CPUAdam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    ax.register_model_and_optimizer(model, optimizer)
    ax.register_loss_fn(get_loss_fn())

    log_memory = False

    for epoch_number in range(num_epochs):
        epoch_loss = run_epoch(train_loader, optimizer, eval_mode=False)
        if not log_memory:
            ax.print_status(
                f"With cpu_offload = {cpu_offload}, "
                "Current memory utilisation = "
                f"{torch.cuda.memory_allocated() /1e9} GB, "
                "Max memory utilisation = "
                f"{torch.cuda.max_memory_allocated() /1e9} GB, "
            )
            log_memory = True
        val_loss = run_epoch(val_loader, optimizer, eval_mode=True)
        if ilp_rank == G_inter - 1 and ax.config.data_parallel_rank == 0:
            ax.print_status(
                f"Epoch {epoch_number+1} : train loss"
                f"{epoch_loss/len(train_loader)} | "
                f"val loss = {val_loss} val ppl = {np.exp(val_loss)}"
            )
    test_ppl = np.exp(run_epoch(test_loader, optimizer, eval_mode=True))
    ax.print_status(f"Final test ppl = {test_ppl}")
