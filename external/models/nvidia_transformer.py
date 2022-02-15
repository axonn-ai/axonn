import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../megatron"))
import megatron  # noqa: E402
from megatron.myelin_compliant_model.transformer import (  # noqa: E402
    ParallelTransformer,  # noqa: E402
)  # noqa: E402
from megatron.model.enums import AttnMaskType  # noqa: E402
from dataclasses import dataclass  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from axonn import axonn as ax  # noqa: E402


@dataclass
class MegatronArgs:
    rank: int
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    kv_channels: int
    ffn_hidden_size: int
    checkpoint_activations: bool
    checkpoint_num_layers: int
    masked_softmax_fusion: bool = True
    fp16: bool = True
    bf16: bool = False
    apply_query_key_layer_scaling: bool = True
    attention_softmax_in_fp32: bool = True
    attention_dropout: float = 0.1
    init_method_std: float = 0.01
    openai_gelu: bool = False
    onnx_safe: bool = False
    bias_gelu_fusion: bool = False
    apply_residual_connection_post_layernorm: bool = False
    fp32_residual_connection: bool = False
    layernorm_epsilon: float = 1e-5
    hidden_dropout: float = 0.1
    bias_dropout_fusion: float = True


try:
    from apex.normalization.fused_layer_norm import FusedLayerNormAffineFunction

    APEX_IS_AVAILABLE = True
except ImportError:
    print(
        "Better speed can be achieved with apex"
        "installed from https://www.github.com/nvidia/apex."
    )
    # BertLayerNorm = BertNonFusedLayerNorm
    APEX_IS_AVAILABLE = False


class NvidiaLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(NvidiaLayerNorm, self).__init__()
        self.shape = torch.Size((hidden_size,))
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.apex_enabled = APEX_IS_AVAILABLE

    @torch.jit.unused
    def fused_layer_norm(self, x):
        return FusedLayerNormAffineFunction.apply(
            x, self.weight, self.bias, self.shape, self.eps
        )

    def forward(self, x):
        if self.apex_enabled and not torch.jit.is_scripting():
            x = self.fused_layer_norm(x)
        else:
            u = x.mean(-1, keepdim=True)
            s = x - u
            s = s * s
            s = s.mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight * x + self.bias
        return x


class GPTEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, vocab_size, hidden_size, seq_len, dropout_prob):
        super(GPTEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(seq_len, hidden_size)
        self.LayerNorm = NvidiaLayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        # embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class GPTLMPredictionHead(nn.Module):
    def __init__(self, embedding_weights):
        super(GPTLMPredictionHead, self).__init__()
        # self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            embedding_weights.size(1), embedding_weights.size(0), bias=False
        )
        self.decoder.weight = embedding_weights
        self.bias = nn.Parameter(torch.zeros(embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


def transformer_encoder(
    num_layers: int,
    hidden_size: int,
    num_attention_heads: int,
    checkpoint_activations: bool,
    checkpoint_num_layers: int,
    world_rank: int,
    pre_process: bool,
    post_process: bool,
    causal_attention: bool,
):
    megatron_args = MegatronArgs(
        rank=world_rank,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        ffn_hidden_size=hidden_size * 4,
        kv_channels=hidden_size // num_attention_heads,
        checkpoint_activations=checkpoint_activations,
        checkpoint_num_layers=checkpoint_num_layers,
    )

    megatron.fused_kernels.load(megatron_args)
    if causal_attention:
        mask_type = AttnMaskType.causal
    else:
        mask_type = AttnMaskType.padding
    return ParallelTransformer(
        megatron_args,
        pre_process=pre_process,
        post_process=post_process,
        self_attn_mask_type=mask_type,
    )


class DistributedGPT(nn.Module):
    def __init__(
        self,
        num_layers,
        hidden_size,
        num_attention_heads,
        vocab_size,
        seq_len,
        ckp_coeff,
    ):
        super(DistributedGPT, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.vocab_size = vocab_size
        self.ckp_coeff = ckp_coeff
        self.ilp_rank = ax.config.inter_layer_parallel_rank
        self.G_inter = ax.config.G_inter
        self.world_rank = ax.comm_handle.world_rank
        self.attn_mask = torch.tril(
            torch.ones((1, seq_len, seq_len), device="cuda").view(seq_len, seq_len)
        )
        self.attn_mask = self.attn_mask < 0.5
        self.seq_len = seq_len

        if self.ilp_rank == 0:
            self.embeddings = GPTEmbeddings(vocab_size, hidden_size, seq_len, 0.1)
        assert num_layers % self.G_inter == 0, "layers should be a multiple of G_inter"
        self.num_layers = num_layers // self.G_inter
        self.encoder = transformer_encoder(
            num_layers=self.num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            checkpoint_activations=True,
            checkpoint_num_layers=ckp_coeff,
            world_rank=self.world_rank,
            pre_process=(self.ilp_rank == 0),
            post_process=(self.ilp_rank == self.G_inter - 1),
            causal_attention=True,
        )
        if self.ilp_rank == self.G_inter - 1:
            if self.ilp_rank == 0:
                temp_embedding = self.embeddings
            else:
                temp_embedding = GPTEmbeddings(vocab_size, hidden_size, seq_len, 0.1)
            self.decoder = GPTLMPredictionHead(temp_embedding.word_embeddings.weight)

    def get_input_shape(self):
        return [self.seq_len, -1, self.hidden_size]

    def get_output_shape(self):
        return [self.seq_len, -1, self.hidden_size]

    def forward(self, x):
        if self.ilp_rank == 0:
            x = self.embeddings(x)
        x = self.encoder(x, self.attn_mask)
        if self.ilp_rank == self.G_inter - 1:
            x = self.decoder(x)
        return x


if __name__ == "__main__":
    hsize = 1024
    seq_len = 32
    batch_size = 16
    gpt_medium = (
        transformer_encoder(12, hsize, 16, True, 4, 0, True, True, True).cuda().half()
    )
    rand_input = torch.randn(
        size=(batch_size, seq_len, hsize), dtype=torch.half, device="cuda"
    )
    attention_mask = torch.tril(
        torch.ones((1, seq_len, seq_len), device="cuda").view(1, 1, seq_len, seq_len)
    )
    extended_attention_mask = attention_mask < 0.5
    print(extended_attention_mask[0][0])
    out = gpt_medium(rand_input, extended_attention_mask)
    print(out.shape, out.dtype, out.device)
