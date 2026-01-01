# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is:w
#  distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Protocol, Any

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe import get_moe_runner_backend
from sglang.srt.utils import (
    is_hip,
    is_cuda,
    is_npu,
)

_is_hip = is_hip()
_is_cuda = is_cuda()
_is_npu = is_npu()


class AWQDequantizeFunc(Protocol):
    """Protocol for AWQ dequantization functions.
    
    All implementations must accept at least 3 tensor arguments (qweight, scales, zeros).
    Some implementations may accept additional optional arguments.
    """
    def __call__(
        self,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        *args: Any,
    ) -> torch.Tensor:
        ...


def awq_dequantize_func() -> AWQDequantizeFunc | None:
    """Get the appropriate AWQ dequantization function based on the hardware backend.
    
    This function performs lazy import of the AWQ dequantization implementation
    that matches the current hardware platform. AWQ (Activation-aware Weight Quantization)
    is a quantization method that requires dequantization during weight loading.
    
    Returns:
        Optional[Callable]: The AWQ dequantization function for the current backend:
            - For CUDA: Returns `sgl_kernel.awq_dequantize`
            - For HIP/ROCm: Returns `awq_dequantize_triton`
            - For NPU: Returns `awq_dequantize_decomposition`
            - For other backends: Returns `None` (AWQ not supported)
    """
    if _is_cuda:
        from sgl_kernel import awq_dequantize
        return awq_dequantize
    elif _is_hip:
        from sglang.srt.layers.quantization.awq_triton import awq_dequantize_triton
        return awq_dequantize_triton
    elif _is_npu:
        from sglang.srt.layers.quantization.awq_triton import awq_dequantize_decomposition
        return awq_dequantize_decomposition
    else:
        return None


def enable_nextn_moe_bf16_cast_to_fp8(quant_config):
    """Check if nextn MoE weights should be cast from bf16 to fp8.
    
    This is used for DeepSeek nvfp4 checkpoint optimization where MoE weights
    in the nextn speculative decoding layer are quantized to fp8 for better performance.
    """
    return (
        envs.SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE.get()
        and quant_config is not None
        and quant_config.get_name() == "modelopt_fp4"
        and get_moe_runner_backend().is_deep_gemm()
    )
