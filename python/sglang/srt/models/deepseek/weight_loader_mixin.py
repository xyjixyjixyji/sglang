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

import torch
from typing import Iterable, Tuple, Optional, List

class DeepSeekV2WeightLoaderMixin:
    def __init__(self, is_nextn: bool = False):
        '''
        Args:
            is_nextn: Whether the model is a nextn model for next n token prediction in speculative decoding
        '''
        self.is_nextn = is_nextn

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        '''
        Load weights for the model

        Args:
            weights: Iterable of tuples of (weight_name, weight_tensor)
        '''
        pass

    def _post_load_weights(self, weight_names: Optional[List[str]] = None):
        '''
        Post processing the model weights after loading

        Args:
            weight_names: List of weight names to post process
        '''
        pass
