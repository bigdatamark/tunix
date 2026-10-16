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

# [MODULE DOC]
# HELPER: Small sharding utility module providing a single function that applies
# a named sharding to a JAX array using the currently active physical mesh
# (taken from the thread-local mesh resource environment).
"""Sharding utilities."""

from typing import Tuple

import jax
from jax import numpy as jnp
from jax.interpreters import pxla
import jax.sharding as shd


# TODO(abheesht17): Use this function for all models and unify with the fn in
# sft/sharding_utils.py.
def shard(x: jnp.ndarray, s: Tuple[str, ...], eager: bool = False):
  # [FUNCTION DOC]
  # Applies a sharding to an array.
  # Inputs: x - array-like; s - tuple of mesh axis names (or None entries) used
  #   as the partition spec, one per array dimension; eager - choose eager
  #   placement vs. a traced sharding constraint.
  # Behavior: read the current physical mesh from the thread-local resource
  #   environment. If that mesh is empty or the first JAX device is a CPU,
  #   simply convert x to a JAX array and return it (no sharding). Otherwise build
  #   a named sharding of the mesh and a partition spec from s; if eager, place
  #   the array with device_put; else return a sharding constraint (for use
  #   inside jit tracing).
  # Output: the (possibly) sharded JAX array.
  # Exceptions: propagates JAX errors for invalid specs/axis names.
  # Side effects: eager mode transfers data to devices.
  """Shards a JAX array.

  Args:
    x: The JAX array to shard.
    s: The sharding spec.
    eager: If True, sharding is done eagerly via jax.device_put. Otherwise,
      sharding is done lazily via jax.lax.with_sharding_constraint.

  Returns:
    The sharded JAX array.
  """
  mesh = pxla.thread_resources.env.physical_mesh
  if mesh.empty or jax.devices()[0].platform == 'cpu':
    return jnp.asarray(x)
  sharding = shd.NamedSharding(mesh, shd.PartitionSpec(*s))
  if eager:
    return jax.device_put(x, sharding)
  return jax.lax.with_sharding_constraint(x, sharding)
