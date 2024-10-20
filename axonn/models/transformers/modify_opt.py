# Copyright 2024 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, ACT2FN
import torch.nn as nn
from axonn.intra_layer import Linear
from axonn import axonn as ax


def modified_attention_init(
    self,
    embed_dim: int,
    num_heads: int,
    dropout: float = 0.0,
    is_decoder: bool = False,
    bias: bool = True,
):
    super(OPTAttention, self).__init__()
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.dropout = dropout
    self.head_dim = embed_dim // num_heads

    if (self.head_dim * num_heads) != self.embed_dim:
        raise ValueError(
            f"embed_dim must be divisible by num_heads "
            f"(got `embed_dim`: {self.embed_dim} & `num_heads`: {num_heads})."
        )
    self.scaling = self.head_dim**-0.5
    self.is_decoder = is_decoder

    self.k_proj = Linear(embed_dim, embed_dim, bias=bias, easy_api=False)
    self.v_proj = Linear(embed_dim, embed_dim, bias=bias, easy_api=False)
    self.q_proj = Linear(embed_dim, embed_dim, bias=bias, easy_api=False)
    self.out_proj = Linear(
        embed_dim, embed_dim, bias=bias, easy_api=False, transpose=True
    )

    assert self.num_heads % ax.config.G_intra_r == 0
    self.num_heads //= ax.config.G_intra_r


def modified_decoder_init(self, config):
    super(OPTDecoderLayer, self).__init__()
    self.embed_dim = config.hidden_size
    self.self_attn = OPTAttention(
        embed_dim=self.embed_dim,
        num_heads=config.num_attention_heads,
        dropout=config.attention_dropout,
        is_decoder=True,
        bias=config.enable_bias,
    )
    self.do_layer_norm_before = config.do_layer_norm_before
    self.dropout = config.dropout
    self.activation_fn = ACT2FN[config.activation_function]

    self.self_attn_layer_norm = nn.LayerNorm(
        self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
    )
    self.fc1 = Linear(
        self.embed_dim, config.ffn_dim, bias=config.enable_bias, easy_api=False
    )
    self.fc2 = Linear(
        config.ffn_dim,
        self.embed_dim,
        bias=config.enable_bias,
        easy_api=False,
        transpose=True,
    )
    self.final_layer_norm = nn.LayerNorm(
        self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
    )


def monkey_patch_opt_with_axonn():
    original_inits = OPTAttention.__init__, OPTDecoderLayer.__init__
    OPTAttention.__init__ = modified_attention_init
    OPTDecoderLayer.__init__ = modified_decoder_init
    return original_inits


def reverse_monkey_patch_opt_with_axonn(original_attention_init, original_mlp_init):
    OPTAttention.__init__ = original_attention_init
    OPTDecoderLayer.__init__ = original_mlp_init
