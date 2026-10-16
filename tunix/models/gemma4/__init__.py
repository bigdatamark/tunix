# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [PACKAGE DOC]
# Gemma 4 model package. Re-exports the submodules model (NNX Gemma4
# implementation and configs) and params_safetensors (Hugging Face
# safetensors loader), and defines BACKEND_MAPPINGS, a dict from backend name
# to weight-mapping spec; currently 'vllm_jax' maps to
# mapping_vllm_jax.VLLM_JAX_MAPPING (used to export/sync weights to the vLLM
# JAX backend). Other package modules: moe (mixture-of-experts layer), params
# (Orbax checkpoint loading and tokenizer). Importing the package imports
# mapping_vllm_jax, model and params_safetensors (model import configures
# the sharding environment).
"""Gemma4 API."""

from tunix.models.gemma4 import mapping_vllm_jax
from tunix.models.gemma4 import model
from tunix.models.gemma4 import params_safetensors

BACKEND_MAPPINGS = {
    'vllm_jax': mapping_vllm_jax.VLLM_JAX_MAPPING,
}

__all__ = ['BACKEND_MAPPINGS', 'model', 'params_safetensors']
