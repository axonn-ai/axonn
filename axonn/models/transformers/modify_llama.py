from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP, ACT2FN
from axonn.intra_layer import Linear


def modified_attention_init(self, config):
    super(LlamaAttention, self).__init__()
    self.config = config
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
            f"hidden_size must be divisible by num_heads "
            f"(got `hidden_size`: {self.hidden_size} & `num_heads`: {self.num_heads})."
        )
    self.q_proj = Linear(
        self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
    )
    self.k_proj = Linear(
        self.hidden_size,
        self.num_key_value_heads * self.head_dim,
        bias=config.attention_bias,
    )
    self.v_proj = Linear(
        self.hidden_size,
        self.num_key_value_heads * self.head_dim,
        bias=config.attention_bias,
    )
    self.o_proj = Linear(
        self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
    )
    self._init_rope()


def modified_mlp_init(self, config):
    super(LlamaMLP, self).__init__()
    self.config = config
    self.hidden_size = config.hidden_size
    self.intermediate_size = config.intermediate_size
    self.gate_proj = Linear(self.hidden_size, self.intermediate_size, bias=False)
    self.up_proj = Linear(self.hidden_size, self.intermediate_size, bias=False)
    self.down_proj = Linear(self.intermediate_size, self.hidden_size, bias=False)
    self.act_fn = ACT2FN[config.hidden_act]


def monkey_patch_llama_with_axonn():
    original_inits = {
        "LlamaAttention": LlamaAttention.__init__,
        "LlamaMLP": LlamaMLP.__init__,
    }
    LlamaAttention.__init__ = modified_attention_init
    LlamaMLP.__init__ = modified_mlp_init
    return original_inits
