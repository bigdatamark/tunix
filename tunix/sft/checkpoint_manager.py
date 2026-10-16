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
# Orbax (v0 API) checkpoint manager wrapper used by PEFT/SFT trainers.
# Saves/restores a composite checkpoint with items "model_params"
#   (nnx.state of the model, optionally only LoRA params),
#   "optimizer_state" (nnx OptState of the optimizer, optional) and
#   "custom_metadata" (JSON). Module constant _DEFAULT_CHECKPOINTING_OPTIONS:
#   continuous saving at most every 180 s, keep 3 checkpoints.
# No side effects at import beyond constructing the default options.
"""Checkpoint manager for PEFT."""

import os
import time
from typing import Any, Tuple

from absl import logging
from flax import nnx
import jax
import orbax.checkpoint as ocp

_DEFAULT_CHECKPOINTING_OPTIONS = ocp.CheckpointManagerOptions(
    save_decision_policy=ocp.checkpoint_managers.ContinuousCheckpointingPolicy(
        minimum_interval_secs=180,
    ),
    max_to_keep=3,
)


class CheckpointManager:
  # [CLASS DOC]
  # Wraps ocp.CheckpointManager (attribute _checkpoint_manager, None when
  # disabled). All methods are no-ops / neutral returns when disabled.
  """Checkpoint manager for PEFT."""

  def __init__(
      self,
      root_directory: str | None = None,
      options: ocp.CheckpointManagerOptions | None = None,
  ):
    # [FUNCTION DOC]
    # Inputs: root_directory (None disables checkpointing), options (Orbax
    #   v0 CheckpointManagerOptions; None uses the module default).
    # Behavior: if root_directory is given, choose item handlers: when the
    #   JAX_PLATFORMS env var contains "proxy" (Pathways), use
    #   PyTreeCheckpointHandler with use_ocdbt=False and use_zarr3=False for
    #   model_params and optimizer_state and log info if
    #   ENABLE_PATHWAYS_PERSISTENCE is set, else warn about OOM risk;
    #   otherwise default PyTreeCheckpointHandlers. Add a
    #   JsonCheckpointHandler for custom_metadata and create the Orbax
    #   CheckpointManager at root_directory.
    # Exceptions: from Orbax. Side effects: may create directories; logging.
    """Initializes the checkpoint manager.

    Args:
      root_directory: The root directory for the checkpoint manager. If None,
        the checkpoint manager will be disabled.
      options: The options for the checkpoint manager.
    """
    self._checkpoint_manager: ocp.CheckpointManager | None = None
    if root_directory is not None:
      # When using Pathways, the checkpoint manager only supports persistence
      # APIs now.
      if 'proxy' in os.getenv('JAX_PLATFORMS', ''):
        item_handlers = {
            'model_params': ocp.PyTreeCheckpointHandler(
                use_ocdbt=False,
                use_zarr3=False,
            ),
            'optimizer_state': ocp.PyTreeCheckpointHandler(
                use_ocdbt=False,
                use_zarr3=False,
            ),
        }
        if os.getenv('ENABLE_PATHWAYS_PERSISTENCE', ''):
          logging.info(
              'Using persistence API for checkpointing with Pathways.'
          )
        else:
          logging.warning(
              'Checkpointing without the persistence API, be aware of potential'
              ' OOM.'
          )
      else:
        item_handlers = {
            'model_params': ocp.PyTreeCheckpointHandler(),
            'optimizer_state': ocp.PyTreeCheckpointHandler(),
        }
      item_handlers['custom_metadata'] = ocp.JsonCheckpointHandler()
      self._checkpoint_manager = ocp.CheckpointManager(
          root_directory,
          item_handlers=item_handlers,
          options=options or _DEFAULT_CHECKPOINTING_OPTIONS,
      )

  def latest_step(self) -> int | None:
    # [FUNCTION DOC]
    # Returns the latest saved step from Orbax, or None if disabled or no
    # checkpoints exist.
    """Returns the latest step."""
    if self._checkpoint_manager is None:
      return None
    return self._checkpoint_manager.latest_step()

  def save(
      self,
      step: int,
      model: nnx.Module,
      optimizer: nnx.Optimizer | None = None,
      save_only_lora_params: bool = False,
      force: bool = False,
      custom_metadata: dict[str, Any] | None = None,
  ) -> bool:
    # [FUNCTION DOC]
    # Inputs: step, model, optimizer (optional), save_only_lora_params,
    #   force, custom_metadata (optional dict).
    # Steps: return False if disabled, or if not force and Orbax
    #   should_save(step) is False. params = nnx.state(model, LoRAParam) if
    #   save_only_lora_params else full nnx.state(model). Build PyTreeSave
    #   args with default SaveArgs per leaf; if optimizer given, also save
    #   nnx.state(optimizer, OptState) the same way. Call Orbax save with a
    #   Composite of those, custom_metadata (or {}), and force.
    # Output: bool returned by Orbax save (whether saved).
    # Side effects: writes checkpoint files (possibly asynchronously).
    """Saves the params for the given step.

    Args:
      step: The step to save the params for.
      model: The model to save the params for.
      optimizer: The optimizer to save the params for. If None, the optimizer
        will not be saved.
      save_only_lora_params: Whether to save only the LoRA params.
      force: Whether to save the checkpoint regardless of the save decision
        policy.
      custom_metadata: Custom metadata to save with the checkpoint.

    Returns:
      Whether the checkpoint was saved.
    """
    if self._checkpoint_manager is None:
      return False
    if not force and not self._checkpoint_manager.should_save(step):
      return False
    if save_only_lora_params:
      params = nnx.state(model, nnx.LoRAParam)
    else:
      params = nnx.state(model)

    model_cp_args = ocp.args.PyTreeSave(
        item=params, save_args=jax.tree.map(lambda _: ocp.SaveArgs(), params)
    )

    cp_save_args = {
        'model_params': model_cp_args,
    }
    if optimizer is not None:
      optimizer_state = nnx.state(optimizer, nnx.optimizer.OptState)
      optimizer_cp_args = ocp.args.PyTreeSave(
          item=optimizer_state,
          save_args=jax.tree.map(lambda _: ocp.SaveArgs(), optimizer_state),
      )
      cp_save_args['optimizer_state'] = optimizer_cp_args
    return self._checkpoint_manager.save(
        step,
        args=ocp.args.Composite(**cp_save_args),
        custom_metadata=custom_metadata or {},
        force=force,
    )

  def maybe_restore(
      self,
      model: nnx.Module,
      optimizer: nnx.Optimizer | None = None,
      step: int | None = None,
      restore_only_lora_params: bool = False,
  ) -> Tuple[int, dict[str, Any]]:
    # [FUNCTION DOC]
    # Inputs: model, optimizer (optional), step (None = latest),
    #   restore_only_lora_params.
    # Steps: if disabled return (0, {}); if step None use latest, and if
    #   none exists return (0, {}). Read checkpoint metadata. Abstract
    #   target = nnx.state(model[, LoRAParam]) with restore args built from
    #   it (keeps target shardings). If optimizer given and checkpoint has
    #   optimizer_state, target the optimizer OptState with shardings fixed
    #   by fix_sharding, restore both and nnx.update the optimizer; else
    #   restore model params only. nnx.update the model; log duration.
    # Output: (restored step, checkpoint custom_metadata or {}).
    # Exceptions: Orbax errors (e.g. missing step, structure mismatch).
    # Side effects: mutates model and optimizer in place.
    # KNOWN ISSUE: docstring says it returns only the step and raises
    #   RuntimeError; it returns a tuple and raises whatever Orbax raises.
    """Restores the params from the latest checkpoint if available and updates the model provided.

    Args:
      model: The model to restore the params for.
      optimizer: The optimizer to restore the params for. If None or if
        optimizer state is not found in the checkpoint, the optimizer will not
        be restored.
      step: The step to restore the params from. If None, the latest step will
        be used.
      restore_only_lora_params: Whether to restore only the LoRA params.

    Returns:
      The step of the restored checkpoint or 0 if no checkpoint is available.

    Raises:
      RuntimeError: If the checkpoint cannot be restored.
    """
    restore_start = time.time()
    if self._checkpoint_manager is None:
      return 0, {}
    if step is None:
      step = self._checkpoint_manager.latest_step()
      # If no checkpoint is available, return 0.
      if step is None:
        return 0, {}

    metadata = self._checkpoint_manager.metadata(step)

    # Load the params from the checkpoint.
    if restore_only_lora_params:
      abstract_params = nnx.state(model, nnx.LoRAParam)
    else:
      abstract_params = nnx.state(model)

    model_cp_args = ocp.args.PyTreeRestore(
        item=abstract_params,
        restore_args=ocp.checkpoint_utils.construct_restore_args(
            target=abstract_params
        ),
    )

    def fix_sharding(state):
      # [FUNCTION DOC]
      # HELPER: nested. Input: state (optimizer state pytree). Maps leaves
      #   to their shardings; if any leaf has a NamedSharding, returns
      #   nnx.get_named_sharding(optimizer_state, that sharding's mesh) so
      #   scalar leaves (step/count on a single device) become mesh-placed
      #   per their partition annotations; otherwise returns the raw
      #   sharding tree.
      # KNOWN ISSUE: uses the enclosing variable optimizer_state instead of
      #   its state argument (they are the same object at the only call).
      # Scalar values in optimizer states like step and count is initialized as
      # SingleDeviceSharding, which will fail if optimizer is sharded. To fix
      # it, we will replicate the scalar values.
      shardings = jax.tree_util.tree_map(lambda x: x.sharding, state)
      try:
        named_sharding = next(
            s
            for s in jax.tree_util.tree_leaves(shardings)
            if isinstance(s, jax.sharding.NamedSharding)
        )
        return nnx.get_named_sharding(optimizer_state, named_sharding.mesh)
      except StopIteration:
        return shardings

    if optimizer is not None and 'optimizer_state' in metadata.item_metadata:
      optimizer_state = nnx.state(optimizer, nnx.optimizer.OptState)
      fixed_sharding = fix_sharding(optimizer_state)
      optimizer_cp_args = ocp.args.PyTreeRestore(
          item=optimizer_state,
          restore_args=ocp.checkpoint_utils.construct_restore_args(
              target=optimizer_state, sharding_tree=fixed_sharding
          ),
      )
      ckpt = self._checkpoint_manager.restore(
          step,
          args=ocp.args.Composite(
              model_params=model_cp_args,
              optimizer_state=optimizer_cp_args,
          ),
      )
      nnx.update(optimizer, ckpt.optimizer_state)
    else:
      ckpt = self._checkpoint_manager.restore(
          step,
          args=ocp.args.Composite(
              model_params=model_cp_args,
          ),
      )
    # Update the model state with params from the restored checkpoint.
    nnx.update(model, ckpt.model_params)
    logging.info(
        'Restored params from step: %d in %.3f seconds',
        step,
        time.time() - restore_start,
    )
    custom_metadata = metadata.custom_metadata if metadata else {}
    return step, custom_metadata

  def close(self):
    # [FUNCTION DOC]
    # Closes the Orbax manager (waits for pending async saves) if enabled;
    # otherwise no-op.
    """Closes the checkpoint manager."""
    if self._checkpoint_manager is None:
      return
    self._checkpoint_manager.close()
