from contextlib import contextmanager
from modify_opt import monkey_patch_opt_with_axonn
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer
from modify_llama import monkey_patch_llama_with_axonn
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP

modify_dict = {
    "facebook/opt-125m": monkey_patch_opt_with_axonn,
    "facebook/opt-350m": monkey_patch_opt_with_axonn,
    "facebook/opt-1.3b": monkey_patch_opt_with_axonn,
    "codellama/CodeLlama-70b-hf": monkey_patch_llama_with_axonn,
    "codellama/CodeLlama-34b-hf": monkey_patch_llama_with_axonn,
    "codellama/CodeLlama-13b-hf": monkey_patch_llama_with_axonn,
    "codellama/CodeLlama-7b-hf": monkey_patch_llama_with_axonn,
    "deepseek-ai/deepseek-coder-6.7b-base": monkey_patch_llama_with_axonn,
    "meta-llama/Llama-2-7b-hf": monkey_patch_llama_with_axonn,
}


@contextmanager
def parallelize(model_id):
    original_inits = modify_dict[model_id]()  # call to monkey patch
    try:
        yield None
    finally:
        OPTAttention.__init__ = original_inits["OPTAttention"]
        OPTDecoderLayer.__init__ = original_inits["OPTDecoderLayer"]
        LlamaAttention.__init__ = original_inits["LlamaAttention"]
        LlamaMLP.__init__ = original_inits["LlamaMLP"]
