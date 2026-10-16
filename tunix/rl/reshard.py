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
# Utilities for moving pytrees of JAX arrays (model weights) between shardings
# and meshes, e.g. from trainer devices to rollout devices. Tries the
# Pathways experimental reshard API first (requires pathwaysutils and the
# "proxy" JAX platform) and falls back to jax.device_put. Completion is
# logged asynchronously from a background thread. Also reshards a whole nnx
# model onto a given mesh using its partition specs.
"""Resharding functions."""

from concurrent import futures
import functools
# Keep this import for google internal usage.
import math  # pylint: disable=unused-import
import os
import threading
import time
from typing import Any, Callable

from absl import logging
import jax
import jaxtyping
from flax import nnx
from tunix.rl import utils

# TODO(tsbao): move this to util
def callback_on_ready(
    x: jaxtyping.PyTree,
    success: Callable[[], None],
    failure: Callable[[Exception], None],
):
  # [FUNCTION DOC]
  # HELPER: Inputs: pytree x, zero-arg success callback, one-arg failure
  # callback. Creates a Future, attaches a done-callback that calls success if
  # the future has no exception else failure with the exception, and starts a
  # new (non-daemon) thread that blocks until x is ready and then resolves the
  # future with x or with the raised exception. Returns None immediately; the
  # callbacks run on the background thread.
  """Callback to invoke when the Jax array is ready."""
  fut = futures.Future()

  def callback(f):
    # [FUNCTION DOC]
    # HELPER: Done-callback: reads the future's exception and dispatches to
    # success or failure.
    e = f.exception()
    if e is None:
      success()
    else:
      failure(e)

  fut.add_done_callback(callback)

  def wait():
    # [FUNCTION DOC]
    # HELPER: Thread body: waits for x to be ready; sets the future's exception
    # on any error, else sets its result to x.
    try:
      jax.block_until_ready(x)
    except Exception as e:  # pylint: disable=broad-exception-caught
      fut.set_exception(e)
    else:
      fut.set_result(x)

  threading.Thread(target=wait).start()


#


def _get_reshard_fn_pathwaysutils(
    *,
    cache_resharding_plans: bool,
    donate: bool,
    use_experimental_pre_reshard: bool,
):
  # [FUNCTION DOC]
  # HELPER: Keyword-only inputs: cache_resharding_plans, donate,
  # use_experimental_pre_reshard. Imports the pathwaysutils experimental reshard
  # module; on ImportError logs and re-raises. Raises EnvironmentError if the
  # JAX_PLATFORMS environment variable does not contain "proxy". Otherwise
  # returns a closure reshard_fn(x, sharding).
  # KNOWN ISSUE: the docstring says use_experimental_pre_reshard is ignored, but
  # the returned function does use it to choose the API.
  """Returns a reshard function using pathwaysutils.

  Args:
    cache_resharding_plans: Whether to cache resharding plans.
    donate: Whether to donate the input buffer.
    use_experimental_pre_reshard: Ignored.

  Returns:
    A reshard function.
  """
  # This import is expected to fail sometimes internally if pathwaysutils is
  # not linked to the binary.
  try:
    from pathwaysutils.experimental import reshard as experimental_reshard  # pylint: disable=g-import-not-at-top # pytype: disable=import-error
  except ImportError:
    logging.info(
        'Cannot import PathwaysUtils and experimental reshard API.'
    )
    raise
  else:
    if 'proxy' not in os.getenv('JAX_PLATFORMS', ''):
      raise EnvironmentError(
          'Pathways proxy is not available. Make sure you have enabled Pathways'
          ' proxy as jax backend, e.g. os.environ["JAX_PLATFORMS"] = "proxy".'
      )

    def reshard_fn(
        x: Any,
        sharding: jax.sharding.Sharding | Any,
    ):
      # [FUNCTION DOC]
      # HELPER: Reshards pytree x to the target sharding pytree. If
      # use_experimental_pre_reshard, collects each leaf's current sharding and
      # calls the side-channel reshard with an intermediate sharding (passing
      # donate and plan caching); otherwise calls the experimental reshard with
      # donate, no aliasing hint and plan caching. Returns the resharded pytree.

      # TODO(b/476149699): Migrate to new API once it's verified.
      if use_experimental_pre_reshard:
        in_sharding = jax.tree_util.tree_map(
            lambda x: x.sharding,
            x,
        )
        return (
            experimental_reshard.sidechannel_reshard_with_intermediate_sharding(
                x,
                in_sharding,
                sharding,
                donate=donate,
                cache_resharding_plans=cache_resharding_plans,
            )
        )
      else:
        return experimental_reshard.reshard(
            x,
            sharding,
            donate=donate,
            may_alias=None,
            cache_resharding_plans=cache_resharding_plans,
        )

  return reshard_fn


def _get_reshard_fn_jax_device_put(
    *,
    donate: bool,
    cache_resharding_plans: bool = False,  # pylint: disable=unused-argument
    use_experimental_pre_reshard: bool = False,  # pylint: disable=unused-argument
):
  # [FUNCTION DOC]
  # HELPER: Returns jax.device_put partially applied with donate; the plan
  # caching and pre-reshard options are accepted for interface compatibility
  # and ignored. Never raises.
  return functools.partial(
      jax.device_put,
      donate=donate,
  )


def _get_reshard_fn(
    cache_resharding_plans: bool,
    donate: bool,
    use_experimental_pre_reshard: bool,
    get_reshard_fns: list[Callable[..., Any]],
):
  # [FUNCTION DOC]
  # HELPER: Tries each factory in get_reshard_fns in order with the given
  # options; factories raising ImportError or EnvironmentError are skipped
  # (debug-logged). Returns the first reshard function obtained. Raises
  # ValueError if none succeeded.
  # KNOWN ISSUE: the debug and error messages contain literal "{...=}" text
  # because they are not f-strings.
  """Returns a reshard function.

  Args:
    cache_resharding_plans: Whether to cache resharding plans.
    donate: Whether to donate the input buffer.
    use_experimental_pre_reshard: Whether to use experimental pre-reshard.
    get_reshard_fns: A list of reshard functions to try to use.

  Returns:
    A reshard function.
  """
  for get_reshard_fn in get_reshard_fns:
    try:
      reshard_fn = get_reshard_fn(
          cache_resharding_plans=cache_resharding_plans,
          donate=donate,
          use_experimental_pre_reshard=use_experimental_pre_reshard,
      )
    except (ImportError, EnvironmentError):
      logging.debug('Could not support {get_reshard_fn=}.', exc_info=True)
    else:
      return reshard_fn

  raise ValueError('Could not find a reshard function from {get_reshard_fns=}.')


def reshard_pytree(
    source: jaxtyping.PyTree,
    target: jaxtyping.PyTree,
    cache_plan: bool = True,
    donate_input: bool = False,
    use_experimental_pre_reshard: bool = True,
) -> jaxtyping.PyTree:
  # [FUNCTION DOC]
  # Inputs: source pytree of arrays, target pytree of arrays or shardings,
  # cache_plan (default True), donate_input (default False),
  # use_experimental_pre_reshard (default True). Converts every target leaf to a
  # destination sharding: NamedSharding or SingleDeviceSharding leaves are used
  # as is, arrays become a NamedSharding with their mesh, spec and memory kind.
  # Selects a reshard function (Pathways first, then device_put), records the
  # start time, reshards source, registers an asynchronous readiness callback
  # that logs the elapsed time on success or an error on failure, and returns
  # the resharded pytree (possibly not yet ready). Raises ValueError if no
  # reshard backend is available; may invalidate source when donating.
  """Reshard input pytree from source sharding and mesh to target sharding and mesh.

  From source to target, both the sharding and mesh can be different.

  Args:
    source: The input source pytree to reshard.
    target: The target pytree to reshard to. Contains target mesh and named
      sharding information. This can be a pytree containing jax.Array or
      jax.sharding.NamedSharding.
    cache_plan: Whether to cache the resharding plan. This can largely speed up
      the resharding process. Turn off with caution.
    donate_input: Whether to donate the input (source) to the reshard.
    use_experimental_pre_reshard: Whether to use the experimental pre-reshard
      API.

  Returns:
    The resharded pytree.
  """

  def _get_dst_sharding(x):
    # [FUNCTION DOC]
    # HELPER: Maps one target leaf to a sharding as described above; raises
    # AttributeError if the leaf is neither a supported sharding nor has a
    # NamedSharding-like sharding attribute.
    if isinstance(
        x, jax.sharding.NamedSharding | jax.sharding.SingleDeviceSharding
    ):
      return x
    else:
      return jax.sharding.NamedSharding(
          x.sharding.mesh,
          x.sharding.spec,
          memory_kind=x.sharding.memory_kind,
      )

  dst_shardings = jax.tree_util.tree_map(
      _get_dst_sharding,
      target,
  )

  reshard_fn = _get_reshard_fn(
      cache_resharding_plans=cache_plan,
      donate=donate_input,
      use_experimental_pre_reshard=use_experimental_pre_reshard,
      get_reshard_fns=[
          #
          _get_reshard_fn_pathwaysutils,
          _get_reshard_fn_jax_device_put,
      ],
  )

  start = time.time()

  resharded_array = reshard_fn(source, dst_shardings)

  callback_on_ready(
      resharded_array,
      lambda: logging.info('Reshard finished in %.2fs', time.time() - start),
      lambda e: logging.error(
          'Reshard failed in %.2fs: %s', time.time() - start, e
      ),
  )
  return resharded_array


def reshard_model_to_mesh(model: nnx.Module, mesh: jax.sharding.Mesh):
  # [FUNCTION DOC]
  # Inputs: nnx model, mesh (may be None). Determines the mesh of the model's
  # state via utils.get_pytree_mesh_info. If mesh is given and differs, then
  # within the mesh context splits the model, builds NamedShardings on that
  # mesh from each variable's partition spec using the first device's default
  # memory kind, reshards the state with reshard_pytree and merges a new model.
  # Returns the new model, or the original model unchanged otherwise.
  # KNOWN ISSUE: docstring mentions only LoRA models but the logic applies to
  # any model.
  """Reshard the lora model if the mesh is specified and the lora model mesh is not the same as the input mesh."""
  model_mesh = utils.get_pytree_mesh_info(nnx.state(model))
  if mesh is not None and model_mesh != mesh:
    with mesh:
      graph_def, state = nnx.split(model)
      default_memory_kind = jax.devices()[0].default_memory().kind
      dst_shardings = jax.tree_util.tree_map(
          lambda x: jax.sharding.NamedSharding(
              mesh,
              x,
              memory_kind=default_memory_kind,
          ),
          nnx.get_partition_spec(state),
      )
      model = nnx.merge(graph_def, reshard_pytree(state, dst_shardings))
  return model
