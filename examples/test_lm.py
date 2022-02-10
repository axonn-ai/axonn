from axonn import axonn as ax
from axonn import optim
from external.models.nvidia_transformer import DistributedGPT
import torch
from tqdm import tqdm
import torch.nn as nn
from ptb_loader import ptb_dataset, init_vocab


dataset = ptb_dataset(
    "./examples/dataset/PTB/",
    seq_length=128,
    word2ind=init_vocab("./examples/dataset/PTB"),
)
bs_per_gpu = 16
num_gpus = 6
bs = num_gpus * bs_per_gpu
mbs = bs_per_gpu
epochs = 10
cpu_offload = False
vocab_size = 10000
N, D, H = 24, 1024, 16
seq_len = 128

ax.init(
    G_data=6,
    G_inter=1,
    mixed_precision=True,
    fp16_allreduce=True,
    cpu_offload=cpu_offload,
)

ilp_rank = ax.config.inter_layer_parallel_rank
G_inter = ax.config.G_inter
num_epochs = 10

train_loader = ax.create_dataloader(
    dataset, batch_size=bs, micro_batch_size=mbs, num_workers=0
)


model = DistributedGPT(
    N, D, H, vocab_size=vocab_size, seq_len=seq_len, ckp_coeff=8
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
    optimizer = optim.CPUAdam(model.parameters(), lr=0.0001)
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

ax.register_model_and_optimizer(model, optimizer)
ax.register_loss_fn(get_loss_fn())

log_memory = False
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
    if ilp_rank == G_inter - 1 and ax.config.data_parallel_rank == 0:
        ax.print_status(
            f"Epoch {epoch_number+1} : epoch loss {epoch_loss/len(train_loader)}"
        )
