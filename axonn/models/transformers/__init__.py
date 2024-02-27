from contextlib import contextmanager
from modify_opt import monkey_patch_opt_with_axonn
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer

modify_dict = {
    "facebook/opt-125m": monkey_patch_opt_with_axonn,
    "facebook/opt-350m": monkey_patch_opt_with_axonn,
    "facebook/opt-1.3b": monkey_patch_opt_with_axonn,
}


@contextmanager
def parallelize(model_id):
    original_inits = modify_dict[model_id]()  # call to monkey patch
    try:
        yield None
    finally:
        OPTAttention.__init__ = original_inits["OPTAttention"]
        OPTDecoderLayer.__init__ = original_inits["OPTDecoderLayer"]
