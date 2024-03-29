from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    MistralRotaryEmbedding,
    MistralMLP,
    ACT2FN,
)
from axonn.intra_layer import Linear


def modified_attention_init(self, config):
    super(MistralAttention, self).__init__()
    self.config = config
    self.hidden_size = config.hidden_size
    self.num_heads = config.num_attention_heads
    self.head_dim = self.hidden_size // self.num_heads
    self.num_key_value_heads = config.num_key_value_heads
    self.num_key_value_groups = self.num_heads // self.num_key_value_heads
    self.max_position_embeddings = config.max_position_embeddings
    self.rope_theta = config.rope_theta
    self.is_causal = True
    # This gives an attribute error, not sure why
    # self.attention_dropout = config.attention_dropout

    if (self.head_dim * self.num_heads) != self.hidden_size:
        raise ValueError(
            f"hidden_size must be divisible by num_heads "
            f"(got `hidden_size`: {self.hidden_size} & `num_heads`: {self.num_heads})."
        )
    self.q_proj = Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
    self.k_proj = Linear(
        self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
    )
    self.v_proj = Linear(
        self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
    )
    self.o_proj = Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    self.rotary_emb = MistralRotaryEmbedding(
        self.head_dim,
        max_position_embeddings=self.max_position_embeddings,
        base=self.rope_theta,
    )


def modified_mlp_init(self, config):
    super(MistralMLP, self).__init__()
    self.config = config
    self.hidden_size = config.hidden_size
    self.intermediate_size = config.intermediate_size
    self.gate_proj = Linear(self.hidden_size, self.intermediate_size, bias=False)
    self.up_proj = Linear(self.hidden_size, self.intermediate_size, bias=False)
    self.down_proj = Linear(self.intermediate_size, self.hidden_size, bias=False)
    self.act_fn = ACT2FN[config.hidden_act]


def monkey_patch_mistral_with_axonn():
    MistralAttention.__init__ = modified_attention_init
    MistralMLP.__init__ = modified_mlp_init
