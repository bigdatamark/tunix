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
# HELPER: small utility module for placing host input batches onto the
#   current device mesh with a data-parallel PartitionSpec.
# Uses the mesh from the active "with mesh:" context
#   (pxla.thread_resources.env.physical_mesh). No side effects at import.
"""Utilities for sharding tensors."""

from typing import Tuple

import jax
from jax.interpreters import pxla
import jax.sharding as shd
import numpy as np


def shard_input(
    input_data: jax.Array, data_sharding_axis: Tuple[str, ...]
) -> jax.Array:
  # [FUNCTION DOC]
  # Shards a pytree of input arrays over the active mesh.
  # Inputs: input_data (any pytree of arrays, e.g. a TrainingInput),
  #   data_sharding_axis (tuple of mesh axis names forming the
  #   PartitionSpec, e.g. ("fsdp",)).
  # Steps: if there is no active mesh return input unchanged. Build pspec.
  #   If every leaf is a jax.Array already sharded on this mesh with this
  #   exact spec, return unchanged. If any leaf is a jax.Array that is not
  #   fully addressable by this process, return unchanged. Otherwise,
  #   within transfer_guard("allow"), map each leaf to
  #   make_array_from_process_local_data(get_sharding(leaf), leaf).
  # Output: pytree of the same structure with global sharded arrays.
  # Exceptions: from JAX array creation. Side effects: host-to-device copy.
  # KNOWN ISSUE: type hint and docstring disagree (jax.Array vs
  #   TrainingInput); any pytree works.
  """Shards the input data across the available devices.

  Args:
    input_data: The input data to be sharded, expected to be a TrainingInput
      dataclass.
    data_sharding_axis: The sharding axis for the input data, e.g. ("fsdp",).

  Returns:
    The sharded TrainingInput.
  """
  mesh = pxla.thread_resources.env.physical_mesh
  if mesh.empty:
    return input_data

  pspec = shd.PartitionSpec(*data_sharding_axis)
  # Check if the input is already sharded with the target mesh to avoid
  # re-sharding.
  is_sharded = jax.tree.map(
      lambda x: isinstance(x, jax.Array)
      and hasattr(x, "sharding")
      and hasattr(x.sharding, "mesh")
      and x.sharding.mesh == mesh
      and hasattr(x.sharding, "spec")
      and x.sharding.spec == pspec,
      input_data,
  )
  if all(jax.tree.leaves(is_sharded)):
    return input_data

  # Skip sharding if the input contains globally distributed arrays that are
  # not fully addressable by the local process. Attempting to recreate them via
  # `jax.make_array_from_process_local_data` would fail with a ValueError.
  is_global = jax.tree.map(
      lambda x: isinstance(x, jax.Array) and not x.is_fully_addressable,
      input_data,
  )
  if any(jax.tree.leaves(is_global)):
    return input_data

  with jax.transfer_guard("allow"):
    return jax.tree.map(
        lambda x: jax.make_array_from_process_local_data(
            get_sharding(x, mesh=mesh, pspec=pspec), x
        ),
        input_data,
    )


def get_sharding(x: jax.Array, mesh: shd.Mesh, pspec: shd.PartitionSpec):
  # [FUNCTION DOC]
  # HELPER: chooses a NamedSharding for one leaf. Inputs: x, mesh, pspec.
  # Returns fully replicated sharding if x is not a numpy/jax array, is a
  #   scalar (ndim 0), has fewer dims than pspec entries, or any sharded
  #   dimension i is not divisible by the product... more precisely by
  #   each individual mesh axis size named for that dimension (a tuple
  #   entry is checked axis by axis). Otherwise NamedSharding(mesh, pspec).
  # Exceptions: KeyError if pspec names an axis absent from the mesh.
  # KNOWN ISSUE: for tuple entries divisibility is checked per axis, not
  #   against the product of the axis sizes, so a dimension divisible by
  #   each but not by their product is still sharded and may fail later.
  # Only shard arrays with rank > 0.
  """Get a sharding for an tensor given a mesh and partition spec."""
  if not isinstance(x, (np.ndarray, jax.Array)) or x.ndim == 0:
    return shd.NamedSharding(mesh, shd.PartitionSpec())  # Replicated

  # Don't shard if rank is not sufficient.
  if x.ndim < len(pspec):
    return shd.NamedSharding(mesh, shd.PartitionSpec())  # Replicated

  # Check for divisibility for all sharded axes.
  for i, axis_name in enumerate(pspec):
    if axis_name is not None:
      axis_names = axis_name if isinstance(axis_name, tuple) else (axis_name,)
      for name in axis_names:
        axis_size = mesh.shape[name]
        if x.shape[i] % axis_size != 0:
          # Replicate if not evenly divisible.
          return shd.NamedSharding(mesh, shd.PartitionSpec())
  return shd.NamedSharding(mesh, pspec)
