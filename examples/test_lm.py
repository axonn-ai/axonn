from axonn import axonn as ax
from axonn import optim
from external.models.nvidia_transformer import DistributedGPT
import torch
from tqdm import tqdm
import torch.nn as nn
from ptb_loader import ptb_dataset, init_vocab
from wikitext_loader import wikitext_dataset
import numpy as np


bs_per_gpu = 16
num_gpus = 6
bs = num_gpus * bs_per_gpu
mbs = bs_per_gpu
num_epochs = 30
cpu_offload = False
N, D, H = 12, 768, 12
seq_len = 512
dataset = "wikitext"  # one of ptb or wikitext

if dataset == "ptb":
    word2ind = init_vocab("./examples/dataset/PTB")
    train_dataset = ptb_dataset(
        "./examples/dataset/PTB/", seq_length=seq_len, word2ind=word2ind, split="train"
    )
    val_dataset = ptb_dataset(
        "./examples/dataset/PTB/", seq_length=seq_len, word2ind=word2ind, split="valid"
    )
    test_dataset = ptb_dataset(
        "./examples/dataset/PTB/", seq_length=seq_len, word2ind=word2ind, split="test"
    )
    vocab_size = 10000
    num_workers = 2
    lr = 1e-4
elif dataset == "wikitext":
    train_dataset = wikitext_dataset(
        "./examples/dataset/wikitext/", seq_length=seq_len, split="train"
    )
    val_dataset = wikitext_dataset(
        "./examples/dataset/wikitext/", seq_length=seq_len, split="valid"
    )
    test_dataset = wikitext_dataset(
        "./examples/dataset/wikitext/", seq_length=seq_len, split="test"
    )
    vocab_size = 50257
    num_workers = 0  # segfaults with more than 0 workers
    lr = 1e-3

ax.init(
    G_data=6,
    G_inter=1,
    mixed_precision=True,
    fp16_allreduce=True,
    cpu_offload=cpu_offload,
)

ilp_rank = ax.config.inter_layer_parallel_rank
G_inter = ax.config.G_inter

train_loader = ax.create_dataloader(
    train_dataset, batch_size=bs, micro_batch_size=mbs, num_workers=num_workers
)


val_loader = ax.create_dataloader(
    val_dataset, batch_size=bs, micro_batch_size=mbs, num_workers=num_workers
)

test_loader = ax.create_dataloader(
    test_dataset, batch_size=bs, micro_batch_size=mbs, num_workers=2
)

model = DistributedGPT(
    N, D, H, vocab_size=vocab_size, seq_len=seq_len, ckp_coeff=4
).cuda()


def init_bert_weights(module):
    """Initialize the weights."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.01)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


model.apply(init_bert_weights)


def get_loss_fn():
    criterion = torch.nn.CrossEntropyLoss()

    def loss_fn(logits, labels):
        return criterion(logits.view(-1, vocab_size), labels.view(-1))

    return loss_fn


if cpu_offload:
    optimizer = optim.CPUAdam(model.parameters(), lr=lr)
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

ax.register_model_and_optimizer(model, optimizer)
ax.register_loss_fn(get_loss_fn())

log_memory = False


def evaluate(loader):
    val_loss = 0
    for sent in tqdm(
        loader,
        disable=not (ilp_rank == 0 and ax.config.data_parallel_rank == 0),
    ):
        src, trg = sent, sent
        if ilp_rank == 0:
            src = sent[:, :-1].cuda()
            trg = sent[:, 1:].cuda()
        if G_inter > 1:
            if ilp_rank == 0:
                ax.comm_handle.send(trg, G_inter - 1, tag=0, async_op=False)
            elif ilp_rank == G_inter - 1:
                trg = torch.cuda.LongTensor(len(sent), seq_len)
                ax.comm_handle.recv(trg, 0, tag=0, async_op=False)
        val_loss += ax.run_batch(src, trg, eval_mode=True)
    return val_loss / len(val_loader)


for epoch_number in range(num_epochs):
    epoch_loss = 0
    for sent in tqdm(
        train_loader,
        disable=not (ilp_rank == 0 and ax.config.data_parallel_rank == 0),
    ):

        optimizer.zero_grad()
        src, trg = sent, sent
        if ilp_rank == 0:
            src = sent[:, :-1].cuda()
            trg = sent[:, 1:].cuda()
        if G_inter > 1:
            if ilp_rank == 0:
                ax.comm_handle.send(trg, G_inter - 1, tag=0, async_op=False)
            elif ilp_rank == G_inter - 1:
                trg = torch.cuda.LongTensor(len(sent), seq_len)
                ax.comm_handle.recv(trg, 0, tag=0, async_op=False)
        batch_loss = ax.run_batch(src, trg)
        epoch_loss += batch_loss
        optimizer.step()
        if not log_memory:
            ax.print_status(
                f"With cpu_offload = {cpu_offload}, "
                "Current memory utilisation = "
                f"{torch.cuda.memory_allocated() /1e9} GB, "
                "Max memory utilisation = "
                f"{torch.cuda.max_memory_allocated() /1e9} GB, "
            )
            log_memory = True
    val_loss = evaluate(val_loader)
    if ilp_rank == G_inter - 1 and ax.config.data_parallel_rank == 0:
        ax.print_status(
            f"Epoch {epoch_number+1} : train loss"
            f"{epoch_loss/len(train_loader)} | "
            f"val loss = {val_loss} val ppl = {np.exp(val_loss)}"
        )
test_ppl = np.exp(evaluate(test_loader))
ax.print_status(f"Final test ppl = {test_ppl}")
