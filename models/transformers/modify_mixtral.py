from transformers.models.mixtral.modeling_mixtral import (
    MixtralRotaryEmbedding,
    MixtralAttention,
    MixtralBLockSparseTop2MLP,
    ACT2FN,
    Optional,
)
from transformers.utils import logging
from axonn.intra_layer import Linear

logger = logging.get_logger(__name__)


def modified_attention_init(self, config, layer_idx: Optional[int] = None):
    super(MixtralAttention, self).__init__()
    self.config = config
    self.layer_idx = layer_idx
    if layer_idx is None:
        logger.warning_once(
            f"Instantiating {self.__class__.__name__} without passing `layer_idx` "
            "is not recommended and will lead to errors during the forward call, "
            "if caching is used. Please make sure to provide a `layer_idx` "
            "when creating this class."
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

    self.rotary_emb = MixtralRotaryEmbedding(
        self.head_dim,
        max_position_embeddings=self.max_position_embeddings,
        base=self.rope_theta,
    )


def modified_mlp_init(self, config):
    super(MixtralBLockSparseTop2MLP, self).__init__()
    self.ffn_dim = config.intermediate_size
    self.hidden_dim = config.hidden_size

    self.w1 = Linear(self.hidden_dim, self.ffn_dim, bias=False)
    self.w2 = Linear(self.ffn_dim, self.hidden_dim, bias=False)
    self.w3 = Linear(self.hidden_dim, self.ffn_dim, bias=False)

    self.act_fn = ACT2FN[config.hidden_act]


def monkey_patch_mixtral_with_axonn():
    MixtralAttention.__init__ = modified_attention_init
    MixtralBLockSparseTop2MLP.__init__ = modified_mlp_init
