# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [PACKAGE DOC]
# Package for the qwen2 model family. Re-exports the submodules model (network
# definition and config) and params (checkpoint loading), and defines
# BACKEND_MAPPINGS: a dict mapping rollout backend name strings ('vllm_jax',
# 'sglang_jax') to the weight-name mapping objects defined in mapping_vllm_jax
# (VLLM_JAX_MAPPING) and mapping_sglang_jax (SGLANG_JAX_MAPPING). These mappings
# are used to translate tunix parameter names/layouts into the naming used by the
# corresponding inference engine when syncing weights.
# Inputs/outputs: none at import time beyond the exported names in __all__
# (BACKEND_MAPPINGS, model, params). Side effects: imports the four submodules.
# Raises: only import errors from submodules.
"""Qwen2 API."""

from tunix.models.qwen2 import mapping_sglang_jax
from tunix.models.qwen2 import mapping_vllm_jax
from tunix.models.qwen2 import model
from tunix.models.qwen2 import params

BACKEND_MAPPINGS = {
    'vllm_jax': mapping_vllm_jax.VLLM_JAX_MAPPING,
    'sglang_jax': mapping_sglang_jax.SGLANG_JAX_MAPPING,
}


__all__ = ['BACKEND_MAPPINGS', 'model', 'params']
