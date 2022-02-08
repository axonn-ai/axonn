import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../megatron'))
import megatron
from megatron.myelin_compliant_model.transformer import ParallelTransformer
from megatron.model.enums import AttnMaskType
from dataclasses import dataclass

@dataclass
class MegatronArgs:
    rank : int
    num_layers : int
    hidden_size : int
    num_attention_heads : int
    kv_channels : int
    ffn_hidden_size : int
    checkpoint_activations : bool
    checkpoint_num_layers : int
    masked_softmax_fusion : bool = True
    fp16 : bool = True
    bf16 : bool = False
    apply_query_key_layer_scaling : bool = True
    attention_softmax_in_fp32 : bool = True
    attention_dropout : float =  0.1
    init_method_std : float = 0.01
    openai_gelu : bool = False
    onnx_safe : bool = False
    bias_gelu_fusion : bool = False
    apply_residual_connection_post_layernorm : bool = False
    fp32_residual_connection : bool = False
    layernorm_epsilon : float = 1e-5
    hidden_dropout : float = 0.1
    bias_dropout_fusion : float = True


def gpt_encoder(num_layers: int, 
                   hidden_size: int, 
                   num_attention_heads: int, 
                   checkpoint_activations: bool,
                   checkpoint_num_layers: int,
                   world_rank: int,
                   pre_process: bool,
                   post_process: bool
                  ):
    megatron_args = MegatronArgs(
                        rank = world_rank,
                        num_layers = num_layers,
                        hidden_size = hidden_size,
                        num_attention_heads = num_attention_heads,
                        ffn_hidden_size = hidden_size * 4,
                        kv_channels = hidden_size // num_attention_heads,
                        checkpoint_activations = checkpoint_activations,
                        checkpoint_num_layers = checkpoint_num_layers
                    )

    megatron.fused_kernels.load(megatron_args)
    return ParallelTransformer(megatron_args, pre_process=pre_process, post_process=post_process, self_attn_mask_type=AttnMaskType.causal)


if __name__ == '__main__':
    gpt_medium = gpt_encoder(12, 1024, 16, True, 4, 0, True, True)
    print(gpt_medium)
