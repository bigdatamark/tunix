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
# HELPER: Small utility module that builds a model instance whose parameters are
# random normal values instead of loaded checkpoint weights (useful for tests,
# benchmarking and dry runs). It mirrors the shape/sharding approach of the
# safetensors loader: build an abstract (shape-only) model, optionally compute
# per-parameter named shardings from a mesh, generate each tensor (shard by shard
# when sharded), then merge the generated state back into the graph definition.
"""Utilities for creating randomly initialized model weights.

This mirrors the shape/sharding handling of `safetensors_loader.load_and_create_model`
but generates random parameters instead of loading them from files.
"""

import contextlib
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx


def create_dummy_model(
    model_class,
    config,
    mesh=None,
    dtype: jnp.dtype | None = None,
    random_seed: int = 0,
    scale: float = 0.02,
):
  # [FUNCTION DOC]
  # Creates a model of class model_class from config with random weights.
  # Inputs: model_class (callable taking config and an nnx.Rngs keyword 'rngs'),
  # config (passed unchanged), mesh (optional JAX mesh, also used as a context
  # manager), dtype (optional override dtype for all params), random_seed (int seed
  # for a fresh nnx.Rngs stream), scale (multiplier of the standard normal values).
  # Behavior: inside the mesh context (or a null context) it shape-evaluates the
  # model constructor (with a dummy params seed 0) to get an abstract model; splits
  # it into graph definition and abstract state; converts state to a plain nested
  # dict. If a mesh is given, computes the named sharding of each abstract leaf as a
  # parallel nested dict. Then maps over every leaf (with its path, and its sharding
  # when present) producing a random tensor: dtype is the override dtype, else the
  # leaf's dtype, else float32. Unsharded leaves get one full tensor; sharded leaves
  # are built via a make-array-from-callback that generates a tensor of the local
  # shard shape for each device index. Finally merges graph definition and the new
  # state into a concrete model and returns it.
  # Outputs: the model instance with random parameters (sharded if mesh given).
  # Raises: whatever model construction / JAX raises; no explicit checks.
  # Side effects: consumes the RNG stream; allocates device memory.
  # KNOWN ISSUE: each shard is generated independently from the RNG stream, so
  # different shards of one array are not slices of a single coherent random
  # tensor, and the nested jitted helper mutates the RNG stream inside jit, which
  # may yield repeated keys across calls depending on nnx tracing behavior.
  """Create a model with random-initialized parameters.

  Args:
    model_class: Model class to instantiate.
    config: Model configuration.
    mesh: Optional JAX mesh or mesh-like context for sharding.
    dtype: Optional dtype for parameter initialization.
    random_seed: RNG seed for initialization.
    scale: Scaling factor applied to the random normal values.

  Returns:
    Model instance with randomly initialized weights (sharded if a mesh is provided).

  """
  context_manager = mesh if mesh is not None else contextlib.nullcontext()

  with context_manager:
    # Build abstract model to obtain param shapes without allocating full tensors.
    abs_model = nnx.eval_shape(lambda: model_class(config, rngs=nnx.Rngs(params=0)))

  graph_def, abs_state = nnx.split(abs_model)

  state_dict = abs_state.to_pure_dict()
  if mesh is not None:
    sharding_dict = nnx.get_named_sharding(abs_state, mesh).to_pure_dict()
  else:
    sharding_dict = None

  rngs = nnx.Rngs(random_seed)

  @partial(nnx.jit, static_argnums=(2, 3,))
  def make_param(rngs, scale, shape, dt):
    # [FUNCTION DOC]
    # HELPER: Jitted generator (shape and dtype are static args 2 and 3). Draws a
    # fresh key from the rngs 'params' stream and returns scale times a standard
    # normal array of the given shape and dtype. Side effect: advances the stream.
    # Call the stream to get a unique JAX key, then use jax.random
    key = rngs.params() 
    return scale * jax.random.normal(key, shape, dtype=dt)

  def make_random_tensor(path, param, shard=None):
    # [FUNCTION DOC]
    # HELPER: Tree-map callback. Inputs: path (unused), abstract param leaf (needs
    # .shape and possibly .dtype), optional sharding. Returns a full random tensor if
    # shard is None, otherwise a globally sharded array assembled from per-shard
    # random tensors of the sharding's shard shape.
    shape = param.shape
    dt = dtype or getattr(param, "dtype", None) or jnp.float32

    if shard is None:
      return make_param(rngs, scale, shape, dt)
    else:
      shard_shape = shard.shard_shape(shape)

      def _callback(index):
        # [FUNCTION DOC]
        # HELPER: Per-device callback; ignores the index and returns a random tensor of
        # the shard shape.
        return make_param(rngs, scale, shard_shape, dt)

      return jax.make_array_from_callback(shape, shard, _callback)

  if sharding_dict is not None:
    state_dict = jax.tree.map_with_path(make_random_tensor, state_dict, sharding_dict)
  else:
    state_dict = jax.tree.map_with_path(make_random_tensor, state_dict)

  return nnx.merge(graph_def, state_dict)
