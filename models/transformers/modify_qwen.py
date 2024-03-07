# import sys
# sys.path.append("/nfshomes/jwendlan/Qwen-7B")
# need to have local version of Qwen for next import statement, along with above code
from modeling_qwen import QWenAttention, FlashSelfAttention, QWenMLP
import torch.nn as nn
from axonn.intra_layer import Linear
import torch
import math
import warnings
import pathlib
from flash_attn import flash_attn_unpadded_func


def modified_attention_init(self, config):
    super(QWenAttention, self).__init__()

    self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)
    self.seq_length = config.seq_length

    self.hidden_size = config.hidden_size
    self.split_size = config.hidden_size
    self.num_heads = config.num_attention_heads
    self.head_dim = self.hidden_size // self.num_heads

    self.use_flash_attn = config.use_flash_attn
    self.scale_attn_weights = True

    self.projection_size = config.kv_channels * config.num_attention_heads

    assert self.projection_size % config.num_attention_heads == 0
    self.hidden_size_per_attention_head = (
        self.projection_size // config.num_attention_heads
    )

    self.c_attn = Linear(config.hidden_size, 3 * self.projection_size)

    self.c_proj = Linear(
        config.hidden_size, self.projection_size, bias=not config.no_bias
    )

    self.is_fp32 = not (config.bf16 or config.fp16)
    if (
        self.use_flash_attn
        and flash_attn_unpadded_func is not None
        and not self.is_fp32
    ):
        self.core_attention_flash = FlashSelfAttention(
            causal=True, attention_dropout=config.attn_dropout_prob
        )
    self.bf16 = config.bf16

    self.use_dynamic_ntk = config.use_dynamic_ntk
    self.use_logn_attn = config.use_logn_attn

    logn_list = [
        math.log(i, self.seq_length) if i > self.seq_length else 1
        for i in range(1, 32768)
    ]
    logn_tensor = torch.tensor(logn_list)[None, :, None, None]
    self.register_buffer("logn_tensor", logn_tensor, persistent=False)

    self.attn_dropout = nn.Dropout(config.attn_dropout_prob)
    self.softmax_in_fp32 = (
        config.softmax_in_fp32 if hasattr(config, "softmax_in_fp32") else False
    )
    self.use_cache_quantization = (
        config.use_cache_quantization
        if hasattr(config, "use_cache_quantization")
        else False
    )
    self.use_cache_kernel = (
        config.use_cache_kernel if hasattr(config, "use_cache_kernel") else False
    )
    cache_dtype = torch.float
    if self.bf16:
        cache_dtype = torch.bfloat16
    elif config.fp16:
        cache_dtype = torch.float16
    self.cache_qmax = torch.tensor(torch.iinfo(torch.uint8).max, dtype=cache_dtype)
    self.cache_qmin = torch.tensor(torch.iinfo(torch.uint8).min, dtype=cache_dtype)

    if config.use_cache_quantization and config.use_cache_kernel:
        # pre check if the support files existing
        module_root = pathlib.Path(__file__).parent
        src_files = ("cache_autogptq_cuda_256.cpp", "cache_autogptq_cuda_kernel_256.cu")
        if any(not (module_root / src).is_file() for src in src_files):
            warnings.warn("KV cache kernel source files (.cpp and .cu) not found.")
            self.cache_kernels = None
        else:
            try:
                from .cpp_kernels import cache_autogptq_cuda_256

                self.cache_kernels = cache_autogptq_cuda_256
            except ImportError:
                warnings.warn("Failed to import KV cache kernels.")
                self.cache_kernels = None


def modified_mlp_init(self, config):
    super(QWenMLP, self).__init__()
    self.w1 = Linear(
        config.hidden_size, config.intermediate_size // 2, bias=not config.no_bias
    )
    self.w2 = Linear(
        config.hidden_size, config.intermediate_size // 2, bias=not config.no_bias
    )
    ff_dim_in = config.intermediate_size // 2
    self.c_proj = Linear(ff_dim_in, config.hidden_size, bias=not config.no_bias)


def monkey_patch_qwen_with_axonn():
    QWenAttention.__init__ = modified_attention_init
    QWenMLP.__init__ = modified_mlp_init
