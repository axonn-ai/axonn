import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../megatron'))
import megatron
from megatron.myelin_compliant_model.transformer import ParallelTransformer
from megatron.model.enums import AttnMaskType
from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class MegatronArgs:
    rank : int
    num_layers : int
    hidden_size : int
    num_attention_heads : int
    kv_channels : int
    ffn_hidden_size : int
    checkpoint_activations : bool
    checkpoint_num_layers : int
    masked_softmax_fusion : bool = True
    fp16 : bool = True
    bf16 : bool = False
    apply_query_key_layer_scaling : bool = True
    attention_softmax_in_fp32 : bool = True
    attention_dropout : float =  0.1
    init_method_std : float = 0.01
    openai_gelu : bool = False
    onnx_safe : bool = False
    bias_gelu_fusion : bool = False
    apply_residual_connection_post_layernorm : bool = False
    fp32_residual_connection : bool = False
    layernorm_epsilon : float = 1e-5
    hidden_dropout : float = 0.1
    bias_dropout_fusion : float = True

class GPTEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, vocab_size, hidden_size, seq_len, dropout_prob):
        super(GPTEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(seq_len, hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class GPTLMPredictionHead(nn.Module):
    def __init__(self, embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        #self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


def transformer_encoder(num_layers: int, 
                   hidden_size: int, 
                   num_attention_heads: int, 
                   checkpoint_activations: bool,
                   checkpoint_num_layers: int,
                   world_rank: int,
                   pre_process: bool,
                   post_process: bool,
                   causal_attention: bool
                  ):
    megatron_args = MegatronArgs(
                        rank = world_rank,
                        num_layers = num_layers,
                        hidden_size = hidden_size,
                        num_attention_heads = num_attention_heads,
                        ffn_hidden_size = hidden_size * 4,
                        kv_channels = hidden_size // num_attention_heads,
                        checkpoint_activations = checkpoint_activations,
                        checkpoint_num_layers = checkpoint_num_layers
                    )

    megatron.fused_kernels.load(megatron_args)
    if causal_attention:
        mask_type = AttnMaskType.causal
    else:
        mask_type = AttnMaskType.padding
    return ParallelTransformer(megatron_args, pre_process=pre_process, post_process=post_process, self_attn_mask_type=AttnMaskType.causal)

class DistributedGPT(nn.Module):
    def __init__(self, num_layers, hidden_size, num_attention_heads, vocab_size, ckp_coeff, ilp_rank, G_inter):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.vocab_size = vocab_size
        self.ckp_coeff = ckp_coeff
        self.ilp_rank = ilp_rank
        self.G_inter = G_inter

        if ilp_rank == 0:
            self.embeddings = GPTEmbeddings(vocab_size, hidden_size)

if __name__ == '__main__':
    hsize = 1024
    seq_len = 32
    batch_size = 16
    gpt_medium = transformer_encoder(12, hsize, 16, True, 4, 0, True, True, True).cuda().half()
    rand_input = torch.randn(size=(batch_size, seq_len, hsize), dtype=torch.half, device='cuda')
    attention_mask = torch.tril(
                                torch.ones((1, seq_len, seq_len), device="cuda").view(1, 1, seq_len, seq_len)
                            )
    extended_attention_mask = (attention_mask < 0.5)
    print(extended_attention_mask[0][0])
    out = gpt_medium(rand_input, extended_attention_mask)
    print(out.shape, out.dtype, out.device)
