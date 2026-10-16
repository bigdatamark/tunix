# Copyright 2025 Google LLC
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

# [MODULE DOC]
# HELPER: small utility module for loading original (linen / Orbax) Gemma
# checkpoints into a nested parameter dict suitable for
# model.Gemma.from_params. Defines the alias Params (a string-keyed mapping).
# Uses orbax StandardCheckpointer for both restoring params and reading
# metadata (reads from disk / remote storage).
"""Utils for loading Gemma params."""

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

Params = Mapping[str, Any]


def load_and_format_params(path: str) -> Params:
  # [FUNCTION DOC]
  # Loads an Orbax checkpoint at path and returns a nested params dict.
  # Steps: restore the flat checkpoint dict (keys like 'transformer/layer_0/
  # attn/...' separated by '/'), convert every leaf to a jnp array (moves data
  # to the default device), remap MLP entries via _param_remapper, then nest the
  # '/'-separated keys via _unflatten_params.
  # Output: nested dict of jnp arrays. Exceptions propagate from Orbax.
  """Loads parameters and formats them for compatibility."""
  params = _load_params(path)
  param_state = jax.tree.map(jnp.array, params)
  remapped_params = _param_remapper(param_state)
  nested_params = _unflatten_params(remapped_params)
  return nested_params


def load_metadata(path: str) -> Any | None:
  # [FUNCTION DOC]
  # Returns the Orbax StandardCheckpointer metadata (tree of shapes/dtypes) for
  # the checkpoint at path, without loading array data.
  """Loads metadata from a checkpoint path."""
  checkpointer = ocp.StandardCheckpointer()
  metadata = checkpointer.metadata(path)
  return metadata


def _load_params(path: str) -> Params:
  # [FUNCTION DOC]
  # HELPER: restores the raw checkpoint at path with Orbax StandardCheckpointer
  # and returns it unchanged.
  """Loads parameters from a checkpoint path."""
  checkpointer = ocp.StandardCheckpointer()
  params = checkpointer.restore(path)
  return params


def _param_remapper(orig_params: Params) -> Params:
  # [FUNCTION DOC]
  # HELPER: flattens MLP sub-modules. For every top-level key containing
  # 'mlp/', splits off the last path segment (e.g. 'gating_einsum' or 'linear')
  # as the param name and stores value v['w'] under new_params[prefix][param],
  # where prefix is the key without that last segment. Other keys are copied as is.
  # Output: new dict (input not modified).
  # KNOWN ISSUE: if an MLP entry has no 'w' field it is silently dropped
  # (only an empty dict for its prefix is created).
  """Remaps params to new module layout.

  This is needed here because the model definition  does not have a separate
  `mlp` module.

  Args:
    orig_params: original dict of parameters in Gemma format.

  Returns:
    dict of params with different names.
  """
  new_params = {}
  for k, v in orig_params.items():
    if 'mlp/' in k:
      layer_name, param = k.rsplit('/', maxsplit=1)
      if layer_name not in new_params:
        new_params[layer_name] = {}
      if 'w' in v:
        new_params[layer_name][param] = v['w']
    else:
      new_params[k] = v
  return new_params


def _unflatten_params(params: Params) -> Params:
  # [FUNCTION DOC]
  # HELPER: turns a dict with '/'-separated string keys into nested dicts, the
  # last segment being the leaf key. Returns the nested dict.
  """Unflatten keys of params as nested dicts."""
  nested_params = {}
  for path, param in params.items():
    *path, leaf = path.split('/')
    subdict = nested_params
    for key in path:
      subdict = subdict.setdefault(key, {})
    subdict[leaf] = param
  return nested_params
