from contextlib import contextmanager
from transformers import AutoConfig
from .modify_opt import monkey_patch_opt_with_axonn, reverse_monkey_patch_opt_with_axonn
from .modify_llama import (
    monkey_patch_llama_with_axonn,
    reverse_monkey_patch_llama_with_axonn,
)
from .modify_mixtral import (
    monkey_patch_mixtral_with_axonn,
    reverse_monkey_patch_mixtral_with_axonn,
)

modify_dict = {
    "OPTForCausalLM": (
        monkey_patch_opt_with_axonn,
        reverse_monkey_patch_opt_with_axonn,
    ),
    "LlamaForCausalLM": (
        monkey_patch_llama_with_axonn,
        reverse_monkey_patch_llama_with_axonn,
    ),
    "MixtralForCausalLM": (
        monkey_patch_mixtral_with_axonn,
        reverse_monkey_patch_mixtral_with_axonn,
    ),
}


@contextmanager
def parallelize(model_id):
    config = AutoConfig.from_pretrained(model_id)
    architecture = config.architectures[0]
    # config.architectures is a list, not sure what to do
    # if it has multiple elements
    assert (
        architecture in modify_dict
    ), f"{architecture} has not been parallelized within AxoNN"

    monkey_patch_fn, reverse_monkey_patch_fn = modify_dict[architecture]
    original_attention_init, original_mlp_init = monkey_patch_fn()
    try:
        yield None
    finally:
        reverse_monkey_patch_fn(original_attention_init, original_mlp_init)
