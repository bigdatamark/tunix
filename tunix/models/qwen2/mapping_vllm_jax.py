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

# [MODULE DOC]
# Static weight mappings from tunix Qwen2 parameters to the vLLM JAX backend,
# which here uses Hugging Face style names ('...weight'). Defines
# TO_HF_MAPPINGS (base params), LORA_TO_HF_MAPPINGS (LoRA params, tunix
# '<kernel|w>_lora_a/_lora_b' -> '*.weight_lora_a/_lora_b') and
# VLLM_JAX_MAPPING = {'to_hf_mappings', 'lora_to_hf_mappings',
# 'to_hf_transpose_keys' (any key containing 'embedding' transposed (1, 0)),
# 'to_hf_hook_fns' (None)}. Sharding axis name is 'model': embedding on vocab
# dim, MLP down on input dim, gate/up on output dim, q/k/v on heads dim 1, o on
# dim 0, lm_head on dim 1, norms replicated, q/k/v biases sharded on dim 0.
# Mapping entries are (target name pattern, sharding spec), where "*" in a
# tunix path stands for the layer index and is substituted into the target, and
# the sharding spec is a tuple of mesh axis names (or None for replicated) per
# tensor dimension. The consumer (tunix weight-sync utilities) uses these to
# rename, reshard and optionally transpose tunix parameters when pushing them
# into the inference engine. No runtime behavior beyond building these dicts at
# import time; nothing raises.
# The LoRA map also covers q/k/v bias LoRA params (attn.*_bias_lora_a/b ->
# self_attn.*_proj.bias_lora_a/b).
# KNOWN ISSUE: the attention q/k/v/o LoRA targets lack the 'model.' prefix
# ('layers.*.self_attn...') unlike all other targets, so they likely do not
# match engine parameter names.
"""vLLM JAX backend mappings for Qwen2 models."""

from __future__ import annotations

from typing import Any, Dict, Tuple

Sharding = Tuple[str | None, ...]
MappingEntry = Tuple[str, Sharding]


TO_HF_MAPPINGS: Dict[str, MappingEntry] = {
    'embedder.input_embedding': ('model.embed_tokens.weight', ('model', None)),
    'layers.*.input_layernorm.w': (
        'model.layers.*.input_layernorm.weight',
        (None,),
    ),
    'layers.*.mlp.down_proj.kernel': (
        'model.layers.*.mlp.down_proj.weight',
        ('model', None),
    ),
    'layers.*.mlp.gate_proj.kernel': (
        'model.layers.*.mlp.gate_proj.weight',
        (None, 'model'),
    ),
    'layers.*.mlp.up_proj.kernel': (
        'model.layers.*.mlp.up_proj.weight',
        (None, 'model'),
    ),
    'layers.*.post_attention_layernorm.w': (
        'model.layers.*.post_attention_layernorm.weight',
        (None,),
    ),
    'layers.*.attn.k_proj.w': (
        'model.layers.*.self_attn.k_proj.weight',
        (None, 'model', None),
    ),
    'layers.*.attn.o_proj.w': (
        'model.layers.*.self_attn.o_proj.weight',
        ('model', None, None),
    ),
    'layers.*.attn.q_proj.w': (
        'model.layers.*.self_attn.q_proj.weight',
        (None, 'model', None),
    ),
    'layers.*.attn.v_proj.w': (
        'model.layers.*.self_attn.v_proj.weight',
        (None, 'model', None),
    ),
    'layers.*.attn.q_bias': (
        'model.layers.*.self_attn.q_proj.bias',
        ('model', None),
    ),
    'layers.*.attn.k_bias': (
        'model.layers.*.self_attn.k_proj.bias',
        ('model', None),
    ),
    'layers.*.attn.v_bias': (
        'model.layers.*.self_attn.v_proj.bias',
        ('model', None),
    ),
    'final_norm.w': ('model.norm.weight', (None,)),
    'lm_head.w': ('lm_head.weight', (None, 'model')),
}


LORA_TO_HF_MAPPINGS: Dict[str, MappingEntry] = {
    'layers.*.mlp.gate_proj.kernel_lora_a': (
        'model.layers.*.mlp.gate_proj.weight_lora_a',
        (None, None),
    ),
    'layers.*.mlp.gate_proj.kernel_lora_b': (
        'model.layers.*.mlp.gate_proj.weight_lora_b',
        (None, 'model'),
    ),
    'layers.*.mlp.up_proj.kernel_lora_a': (
        'model.layers.*.mlp.up_proj.weight_lora_a',
        (None, None),
    ),
    'layers.*.mlp.up_proj.kernel_lora_b': (
        'model.layers.*.mlp.up_proj.weight_lora_b',
        (None, 'model'),
    ),
    'layers.*.mlp.down_proj.kernel_lora_a': (
        'model.layers.*.mlp.down_proj.weight_lora_a',
        ('model', None),
    ),
    'layers.*.mlp.down_proj.kernel_lora_b': (
        'model.layers.*.mlp.down_proj.weight_lora_b',
        (None, None),
    ),
    'layers.*.attn.q_proj.w_lora_a': (
        'layers.*.self_attn.q_proj.weight_lora_a',
        ('model', None),
    ),
    'layers.*.attn.q_proj.w_lora_b': (
        'layers.*.self_attn.q_proj.weight_lora_b',
        (None, None),
    ),
    'layers.*.attn.k_proj.w_lora_a': (
        'layers.*.self_attn.k_proj.weight_lora_a',
        ('model', None),
    ),
    'layers.*.attn.k_proj.w_lora_b': (
        'layers.*.self_attn.k_proj.weight_lora_b',
        (None, None),
    ),
    'layers.*.attn.v_proj.w_lora_a': (
        'layers.*.self_attn.v_proj.weight_lora_a',
        ('model', None),
    ),
    'layers.*.attn.v_proj.w_lora_b': (
        'layers.*.self_attn.v_proj.weight_lora_b',
        (None, None),
    ),
    'layers.*.attn.o_proj.w_lora_a': (
        'layers.*.self_attn.o_proj.weight_lora_a',
        ('model', None),
    ),
    'layers.*.attn.o_proj.w_lora_b': (
        'layers.*.self_attn.o_proj.weight_lora_b',
        (None, None),
    ),
    'layers.*.attn.q_bias_lora_a': (
        'model.layers.*.self_attn.q_proj.bias_lora_a',
        ('model', None),
    ),
    'layers.*.attn.q_bias_lora_b': (
        'model.layers.*.self_attn.q_proj.bias_lora_b',
        ('model', None),
    ),
    'layers.*.attn.k_bias_lora_a': (
        'model.layers.*.self_attn.k_proj.bias_lora_a',
        ('model', None),
    ),
    'layers.*.attn.k_bias_lora_b': (
        'model.layers.*.self_attn.k_proj.bias_lora_b',
        ('model', None),
    ),
    'layers.*.attn.v_bias_lora_a': (
        'model.layers.*.self_attn.v_proj.bias_lora_a',
        ('model', None),
    ),
    'layers.*.attn.v_bias_lora_b': (
        'model.layers.*.self_attn.v_proj.bias_lora_b',
        ('model', None),
    ),
}


VLLM_JAX_MAPPING: Dict[str, Any] = {
    'to_hf_mappings': TO_HF_MAPPINGS,
    'lora_to_hf_mappings': LORA_TO_HF_MAPPINGS,
    'to_hf_transpose_keys': {'embedding': (1, 0)},
    'to_hf_hook_fns': None,
}

__all__ = [
    'VLLM_JAX_MAPPING',
]