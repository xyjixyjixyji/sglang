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

from sglang.srt.utils import (
    is_hip,
    is_cuda,
    is_npu,
)

_is_hip = is_hip()
_is_cuda = is_cuda()
_is_npu = is_npu()

def awq_dequantize_func():
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
