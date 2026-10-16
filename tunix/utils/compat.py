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
# HELPER: Small compatibility shim exposing version-independent aliases for
# Flax and JAX APIs that were renamed/added across releases.
# Module-level names (computed at import time):
# - ModuleList: the Flax NNX list container type when the installed flax
#   provides one (flax 0.12+), otherwise the builtin list type. Used to hold
#   sub-modules in a way NNX tracks.
# - set_mesh: the context-manager/function that activates a device mesh. Uses
#   the sharding "use_mesh" function when the jax sharding namespace has it,
#   otherwise falls back to the top-level jax "set_mesh".
# Inputs/outputs: none beyond the exported aliases. Exceptions: an
# AttributeError at import if neither mesh API exists. Side effects: imports
# flax and jax.
"""Compatibility utils for old jax and flax versions."""

from flax import nnx
import jax


# Flax version compatibility.
ModuleList = list

# To accomodate github requirements. nnx.List is available in flax 0.12.0 and
# later.
if hasattr(nnx, "List"):
  ModuleList = nnx.List  # noqa: N816 (public alias)


# JAX version compatibility
if hasattr(jax.sharding, "use_mesh"):
  set_mesh = jax.sharding.use_mesh
else:
  set_mesh = jax.set_mesh
