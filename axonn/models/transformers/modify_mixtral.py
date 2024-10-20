# Copyright 2024 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from transformers.models.mixtral.modeling_mixtral import (
    MixtralAttention,
    MixtralRotaryEmbedding,
    MixtralBlockSparseTop2MLP,
    ACT2FN,
)
from axonn.intra_layer import Linear
from axonn import axonn as ax


def modified_attention_init(self, config, layer_idx):
    super(MixtralAttention, self).__init__()
    self.config = config
    self.layer_idx = layer_idx
    if layer_idx is None:
        import logger

        logger.warning_once(
            f"Instantiating {self.__class__.__name__} without passing a"
            f"`layer_idx` is not recommended and will"
            f"lead to errors during the forward call if"
            f"caching is used. Please make sure to provide a `layer_idx` "
            f"when creating this class."
        )

    self.hidden_size = config.hidden_size
    self.num_heads = config.num_attention_heads
    self.head_dim = self.hidden_size // self.num_heads
    self.num_key_value_heads = config.num_key_value_heads
    self.num_key_value_groups = self.num_heads // self.num_key_value_heads
    self.max_position_embeddings = config.max_position_embeddings
    self.rope_theta = config.rope_theta
    self.is_causal = True
    self.attention_dropout = config.attention_dropout

    if (self.head_dim * self.num_heads) != self.hidden_size:
        raise ValueError(
            f"hidden_size must be divisible by num_heads"
            f"(got `hidden_size`: {self.hidden_size}"
            f" and `num_heads`: {self.num_heads})."
        )
    self.q_proj = Linear(
        self.hidden_size, self.num_heads * self.head_dim, bias=False, use_easy_api=False
    )
    self.k_proj = Linear(
        self.hidden_size,
        self.num_key_value_heads * self.head_dim,
        bias=False,
        use_easy_api=False,
    )
    self.v_proj = Linear(
        self.hidden_size,
        self.num_key_value_heads * self.head_dim,
        bias=False,
        use_easy_api=False,
    )
    self.o_proj = Linear(
        self.num_heads * self.head_dim,
        self.hidden_size,
        bias=False,
        use_easy_api=False,
        transpose=True,
    )

    self.rotary_emb = MixtralRotaryEmbedding(
        self.head_dim,
        max_position_embeddings=self.max_position_embeddings,
        base=self.rope_theta,
    )

    assert self.num_heads % ax.config.G_intra_r == 0
    self.num_heads //= ax.config.G_intra_r

    assert self.num_key_value_heads % ax.config.G_intra_r == 0
    self.num_key_value_heads //= ax.config.G_intra_r

    assert self.hidden_size % ax.config.G_intra_r == 0
    self.hidden_size //= ax.config.G_intra_r


def modified_mlp_init(self, config):
    super(MixtralBlockSparseTop2MLP, self).__init__()
    self.ffn_dim = config.intermediate_size
    self.hidden_dim = config.hidden_size

    self.w1 = Linear(self.hidden_dim, self.ffn_dim, bias=False, use_easy_api=False)
    self.w2 = Linear(
        self.ffn_dim, self.hidden_dim, bias=False, use_easy_api=False, transpose=True
    )
    self.w3 = Linear(self.hidden_dim, self.ffn_dim, bias=False, use_easy_api=False)

    self.act_fn = ACT2FN[config.hidden_act]


def monkey_patch_mixtral_with_axonn():
    original_inits = MixtralAttention.__init__, MixtralBlockSparseTop2MLP.__init__
    MixtralAttention.__init__ = modified_attention_init
    MixtralBlockSparseTop2MLP.__init__ = modified_mlp_init
    return original_inits


def reverse_monkey_patch_mixtral_with_axonn(original_attention_init, original_mlp_init):
    MixtralAttention.__init__ = original_attention_init
    MixtralBlockSparseTop2MLP.__init__ = original_mlp_init
