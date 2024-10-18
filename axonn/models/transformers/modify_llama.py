# Copyright 2024 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP, ACT2FN
from axonn.intra_layer import Linear
from typing import Optional
from axonn import axonn as ax


def modified_attention_init(self, config, layer_idx: Optional[int] = None):
    super(LlamaAttention, self).__init__()
    self.config = config
    self.layer_idx = layer_idx
    if layer_idx is None:
        logger.warning_once(  # noqa: F821
            f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "  # noqa: E501
            "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "  # noqa: E501
            "when creating this class."
        )

    self.attention_dropout = config.attention_dropout
    self.hidden_size = config.hidden_size
    self.num_heads = config.num_attention_heads
    self.head_dim = self.hidden_size // self.num_heads
    self.num_key_value_heads = config.num_key_value_heads
    self.num_key_value_groups = self.num_heads // self.num_key_value_heads
    self.max_position_embeddings = config.max_position_embeddings
    self.rope_theta = config.rope_theta
    self.is_causal = True

    if (self.head_dim * self.num_heads) != self.hidden_size:
        raise ValueError(
            f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"  # noqa: E501
            f" and `num_heads`: {self.num_heads})."
        )

    self.q_proj = Linear(
        self.hidden_size,
        self.num_heads * self.head_dim,
        bias=config.attention_bias,
        use_easy_api=False,
    )
    self.k_proj = Linear(
        self.hidden_size,
        self.num_key_value_heads * self.head_dim,
        bias=config.attention_bias,
        use_easy_api=False,
    )
    self.v_proj = Linear(
        self.hidden_size,
        self.num_key_value_heads * self.head_dim,
        bias=config.attention_bias,
        use_easy_api=False,
    )
    self.o_proj = Linear(
        self.hidden_size,
        self.hidden_size,
        bias=config.attention_bias,
        use_easy_api=False,
        transpose=True,
    )

    assert self.num_heads % ax.config.G_intra_r == 0
    self.num_heads //= ax.config.G_intra_r

    assert self.num_key_value_heads % ax.config.G_intra_r == 0
    self.num_key_value_heads //= ax.config.G_intra_r

    assert self.hidden_size % ax.config.G_intra_r == 0
    self.hidden_size //= ax.config.G_intra_r

    self._init_rope()


def modified_mlp_init(self, config):
    super(LlamaMLP, self).__init__()
    self.config = config
    self.hidden_size = config.hidden_size
    self.intermediate_size = config.intermediate_size
    self.gate_proj = Linear(
        self.hidden_size, self.intermediate_size, bias=False, use_easy_api=False
    )
    self.up_proj = Linear(
        self.hidden_size, self.intermediate_size, bias=False, use_easy_api=False
    )
    self.down_proj = Linear(
        self.intermediate_size,
        self.hidden_size,
        bias=False,
        use_easy_api=False,
        transpose=True,
    )
    self.act_fn = ACT2FN[config.hidden_act]


def monkey_patch_llama_with_axonn():
    original_inits = LlamaAttention.__init__, LlamaMLP.__init__
    LlamaAttention.__init__ = modified_attention_init
    LlamaMLP.__init__ = modified_mlp_init
    return original_inits


def reverse_monkey_patch_llama_with_axonn(original_attention_init, original_mlp_init):
    LlamaAttention.__init__ = original_attention_init
    LlamaMLP.__init__ = original_mlp_init
