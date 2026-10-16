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
# Unit tests for tunix.sft.checkpoint_manager.CheckpointManager (Orbax-based
# saving/restoring of nnx model params, optionally LoRA-only params,
# optimizer state and custom metadata; returns (step, metadata) on restore).
# Module setup: sets XLA_FLAGS to force 4 host CPU devices and disables
# flax_always_shard_variable if that config flag exists.
# KNOWN ISSUE: XLA_FLAGS is set after importing jax; it only takes effect if
# the backend is not yet initialized.
# Side effects: writes checkpoints under temporary directories; reads
# committed checkpoints from test_data/checkpoints next to this file.
"""Peft Checkpoint manager unittest."""

import os
import tempfile
from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from flax import config as flax_config
from flax import nnx
import jax
import jax.numpy as jnp
import jax.sharding as shd
import numpy as np
import optax
import qwix
from tunix.sft import checkpoint_manager

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'


if hasattr(flax_config, 'flax_always_shard_variable'):
  flax_config.update('flax_always_shard_variable', False)


def assert_close(path, x, y, atol=1e-5, rtol=1e-5):
  # [FUNCTION DOC]
  # HELPER: tree-map callback asserting x and y are all-close, reporting the
  # tree path on mismatch. Inputs: path, two arrays, atol and rtol (1e-5).
  # Raises AssertionError on mismatch.
  # KNOWN ISSUE: atol and rtol are passed positionally into numpy's
  # assert_allclose whose positional order is rtol then atol, so they are
  # swapped (harmless while both defaults are equal).
  np.testing.assert_allclose(
      x, y, atol, rtol, err_msg=f'Mismatch at path: {path}'
  )


def assert_not_equal(path, x, y):
  # [FUNCTION DOC]
  # HELPER: tree-map callback asserting at least one element of x differs from
  # y; raises AssertionError naming the path otherwise.
  np.testing.assert_(
      np.any(np.not_equal(x, y)), msg=f'Unexpected match at path: {path}'
  )


class TestModel(nnx.Module):
  # [CLASS DOC]
  # HELPER: tiny two-layer nnx MLP with residual used as a checkpoint subject.
  # w1 is Linear 2->4 partitioned ('fsdp','tp'), w2 is Linear 4->2 partitioned
  # ('tp','fsdp'), both lecun_normal initialized.

  def __init__(self, rngs: nnx.Rngs):
    # [FUNCTION DOC]
    # HELPER: builds w1 and w2 with the given nnx.Rngs and partitioning metadata.
    kernel_init_fn = nnx.initializers.lecun_normal()
    self.w1 = nnx.Linear(
        in_features=2,
        out_features=4,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(kernel_init_fn, ('fsdp', 'tp')),
    )
    self.w2 = nnx.Linear(
        in_features=4,
        out_features=2,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(kernel_init_fn, ('tp', 'fsdp')),
    )

  def __call__(self, x):
    # [FUNCTION DOC]
    # HELPER: forward pass; returns w2(relu(w1(x))) + x (input dim must be 2).
    h = nnx.relu(self.w1(x))
    h = self.w2(h) + x
    return h


def create_sharded_model(model_ctor, rngs, mesh):
  # [FUNCTION DOC]
  # HELPER: builds a model whose state is sharded on the mesh. Inside a jitted
  # inner function (model_ctor static) constructs the model, applies sharding
  # constraints from its partition specs, and updates the model; runs it under
  # the mesh. Inputs: model constructor, nnx.Rngs, mesh.
  # Outputs: (model, NamedSharding tree for the state).
  @nnx.jit(static_argnums=(0,))
  def _create_sharded_model(model_ctor, rngs):
    # [FUNCTION DOC]
    # HELPER: jitted inner builder; returns the constrained model and its
    # (pre-constraint) state object.
    model = model_ctor(rngs)
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)
    return model, state

  with mesh:
    model, state = _create_sharded_model(model_ctor, rngs)
  state_sharding = nnx.get_named_sharding(state, mesh)
  return model, state_sharding


class CheckpointManagerTest(parameterized.TestCase):
  # [CLASS DOC]
  # Test case for CheckpointManager. Fixture provides a temp path and a
  # (2, device_count/2) mesh with axes ('fsdp','tp').

  def setUp(self):
    # [FUNCTION DOC]
    # HELPER: fixture creating self.temp_path, self.device_count and self.mesh.
    super().setUp()
    try:
      self.temp_path = self.create_tempdir().full_path
    except Exception:
      self.temp_path = tempfile.TemporaryDirectory().name
    self.device_count = jax.device_count()
    self.mesh = jax.sharding.Mesh(
        devices=np.array(jax.devices()).reshape(2, self.device_count // 2),
        axis_names=('fsdp', 'tp'),
    )

  def test_empty_root_directory(self):
    # [FUNCTION DOC]
    # Verifies a manager with root_directory None is a no-op: latest_step None,
    # save returns False, maybe_restore returns (0, {}).
    cp_manager = checkpoint_manager.CheckpointManager(root_directory=None)
    self.assertIsNone(cp_manager.latest_step())
    self.assertFalse(cp_manager.save(1, None))
    self.assertEqual(cp_manager.maybe_restore(None), (0, {}))

  def test_checkpoint_manager_options_none_sets_default(self):
    # [FUNCTION DOC]
    # Verifies options None results in the module's default checkpointing
    # options being used by the underlying Orbax manager.
    cp_path = f'{self.temp_path}/{self.id()}'
    cp_manager = checkpoint_manager.CheckpointManager(cp_path, options=None)
    self.assertIsNotNone(cp_manager._checkpoint_manager)
    self.assertEqual(
        cp_manager._checkpoint_manager._options,  # pytype: disable=attribute-error
        checkpoint_manager._DEFAULT_CHECKPOINTING_OPTIONS,
    )

  def test_save(self):
    # [FUNCTION DOC]
    # Verifies save returns True, latest_step becomes 1, and
    # '<path>/1/model_params' exists after close.
    cp_path = f'{self.temp_path}/{self.id()}'
    cp_manager = checkpoint_manager.CheckpointManager(cp_path)
    model, _ = create_sharded_model(TestModel, nnx.Rngs(0), self.mesh)

    # Save the model state.
    self.assertTrue(cp_manager.save(1, model))
    cp_manager._checkpoint_manager.wait_until_finished()  # pytype: disable=attribute-error
    self.assertEqual(cp_manager.latest_step(), 1)

    cp_manager.close()
    model_param_path = epath.Path(cp_path) / '1' / 'model_params'
    # Verify the model params are saved.
    self.assertTrue(model_param_path.exists())

  def test_restore(self):
    # [FUNCTION DOC]
    # Verifies params modified after saving are restored to the saved values and
    # maybe_restore returns (1, {}).
    cp_path = f'{self.temp_path}/{self.id()}'
    cp_manager = checkpoint_manager.CheckpointManager(cp_path)
    model, _ = create_sharded_model(TestModel, nnx.Rngs(0), self.mesh)
    expected_state = nnx.state(model)

    # Save the model params.
    self.assertTrue(cp_manager.save(1, model))
    cp_manager._checkpoint_manager.wait_until_finished()  # pytype: disable=attribute-error

    # Change the model state.
    changed_state = jax.tree.map(lambda x: x + 1, nnx.state(model))
    nnx.update(model, changed_state)

    # Restore the model params.
    self.assertEqual(cp_manager.maybe_restore(model), (1, {}))
    # Check the model params are restored correctly.
    jax.tree.map_with_path(
        assert_close,
        expected_state,
        nnx.state(model),
    )

  def test_restore_different_sharding(self):
    # [FUNCTION DOC]
    # Verifies restore honours the target model's sharding: restoring into an
    # unsharded model yields SingleDeviceSharding, into the sharded model yields
    # the declared PartitionSpecs.
    cp_path = f'{self.temp_path}/{self.id()}'
    cp_manager = checkpoint_manager.CheckpointManager(cp_path)
    unsharded_model = TestModel(nnx.Rngs(0))
    model, _ = create_sharded_model(TestModel, nnx.Rngs(0), self.mesh)

    # Save the model params.
    self.assertTrue(cp_manager.save(1, unsharded_model))
    cp_manager._checkpoint_manager.wait_until_finished()  # pytype: disable=attribute-error

    # Restore the model without shardings.
    self.assertEqual(cp_manager.maybe_restore(unsharded_model), (1, {}))
    unsharded_variables = nnx.state(unsharded_model, nnx.Param)
    # Check the model shardings are restored correctly.
    self.assertIsInstance(
        unsharded_variables.w1.kernel.value.sharding,
        jax.sharding.SingleDeviceSharding,
    )
    self.assertIsInstance(
        unsharded_variables.w2.kernel.value.sharding,
        jax.sharding.SingleDeviceSharding,
    )

    # Restore the model with shardings.
    self.assertEqual(cp_manager.maybe_restore(model), (1, {}))
    # Check the model shardings are restored correctly.
    variables = nnx.state(model, nnx.Param)

    self.assertEqual(
        variables.w1.kernel.value.sharding.spec,
        shd.PartitionSpec('fsdp', 'tp'),
    )
    self.assertEqual(
        variables.w2.kernel.value.sharding.spec,
        shd.PartitionSpec('tp', 'fsdp'),
    )

  def test_restore_with_lora(self):
    # [FUNCTION DOC]
    # Verifies LoRA-only save/restore: after applying qwix LoRA to w1 and saving
    # with save_only_lora_params, perturbing all params and restoring with
    # restore_only_lora_params restores LoRA params only; non-LoRA params stay
    # perturbed.
    cp_path = f'{self.temp_path}/{self.id()}'
    cp_manager = checkpoint_manager.CheckpointManager(cp_path)
    model, _ = create_sharded_model(TestModel, nnx.Rngs(0), self.mesh)
    lora_provider = qwix.LoraProvider(
        module_path='.*w1',
        rank=4,
        alpha=2.0,
    )
    dummy_model_input = {
        'x': jnp.ones(2, dtype=jnp.int32),
    }
    model = qwix.apply_lora_to_model(model, lora_provider, **dummy_model_input)
    expected_lora_state = nnx.clone(nnx.state(model, nnx.LoRAParam))
    old_non_lora_state = nnx.clone(
        nnx.state(model, (nnx.filterlib.Not(nnx.LoRAParam)))
    )

    # Save the model params.
    self.assertTrue(cp_manager.save(1, model, save_only_lora_params=True))
    cp_manager._checkpoint_manager.wait_until_finished()  # pytype: disable=attribute-error

    # Change the model state.
    changed_state = jax.tree.map(lambda x: x + 1, nnx.state(model))
    nnx.update(model, changed_state)

    # Restore the model lora params.
    self.assertEqual(
        cp_manager.maybe_restore(model, restore_only_lora_params=True),
        (1, {}),
    )
    # Check the model lora params are restored correctly.
    jax.tree.map_with_path(
        assert_close,
        expected_lora_state,
        nnx.state(model, nnx.LoRAParam),
    )
    # Check the rest of the params are not restored.
    jax.tree.map_with_path(
        assert_not_equal,
        old_non_lora_state,
        nnx.state(model, nnx.filterlib.Not(nnx.LoRAParam)),
    )

  def test_save_and_restore_with_custom_metadata(self):
    # [FUNCTION DOC]
    # Verifies custom_metadata given to save is returned by maybe_restore.
    cp_path = f'{self.temp_path}/{self.id()}'
    ckpt_manager = checkpoint_manager.CheckpointManager(cp_path)
    model, _ = create_sharded_model(TestModel, nnx.Rngs(0), self.mesh)
    custom_metadata = {'foo': 1, 'bar': 2}
    ckpt_manager.save(1, model, custom_metadata=custom_metadata)
    ckpt_manager._checkpoint_manager.wait_until_finished()  # pytype: disable=attribute-error
    restored_step, restored_metadata = ckpt_manager.maybe_restore(model)
    self.assertEqual(restored_step, 1)
    self.assertEqual(restored_metadata, custom_metadata)

  def test_save_and_restore_with_optimizer_state(self):
    # [FUNCTION DOC]
    # Verifies optimizer state (incl. injected learning_rate hyperparam 1e-3) is
    # restored into a fresh optimizer created with lr 1e-5, plus metadata.
    cp_path = f'{self.temp_path}/{self.id()}'
    ckpt_manager = checkpoint_manager.CheckpointManager(cp_path)
    model, _ = create_sharded_model(TestModel, nnx.Rngs(0), self.mesh)
    optimizer = nnx.Optimizer(
        model,
        optax.inject_hyperparams(optax.adamw)(learning_rate=1e-3),
        wrt=nnx.Param,
    )
    custom_metadata = {'foo': 1, 'bar': 2}
    ckpt_manager.save(1, model, optimizer, custom_metadata=custom_metadata)
    ckpt_manager._checkpoint_manager.wait_until_finished()  # pytype: disable=attribute-error

    new_optimizer = nnx.Optimizer(
        model,
        optax.inject_hyperparams(optax.adamw)(learning_rate=1e-5),
        wrt=nnx.Param,
    )
    self.assertEqual(
        new_optimizer.opt_state.hyperparams['learning_rate'].value, 1e-5
    )
    restored_step, restored_metadata = ckpt_manager.maybe_restore(
        model, new_optimizer
    )
    self.assertEqual(restored_step, 1)
    self.assertEqual(restored_metadata, custom_metadata)
    jax.tree.map_with_path(
        assert_close,
        nnx.state(new_optimizer, nnx.optimizer.OptState),
        nnx.state(optimizer, nnx.optimizer.OptState),
    )
    self.assertEqual(
        new_optimizer.opt_state.hyperparams['learning_rate'].value, 1e-3
    )

  def test_restore_without_optimizer(self):
    # [FUNCTION DOC]
    # Verifies restoring a checkpoint that contains optimizer state works when no
    # optimizer is passed (must not raise).
    cp_path = f'{self.temp_path}/{self.id()}'
    ckpt_manager = checkpoint_manager.CheckpointManager(cp_path)
    model, _ = create_sharded_model(TestModel, nnx.Rngs(0), self.mesh)
    optimizer = nnx.Optimizer(
        model,
        optax.inject_hyperparams(optax.adamw)(learning_rate=1e-3),
        wrt=nnx.Param,
    )
    ckpt_manager.save(1, model, optimizer)
    ckpt_manager._checkpoint_manager.wait_until_finished()  # pytype: disable=attribute-error
    ckpt_manager.maybe_restore(model)

  @parameterized.parameters(['test_data/checkpoints'])
  def test_restore_with_backward_compatibility(self, ckpt_path):
    # [FUNCTION DOC]
    # Verifies backward compatibility with checkpoints saved via StandardSave in
    # test_data: restoring into a perturbed Rngs(0) model reproduces the initial
    # Rngs(0) params. Input: relative checkpoint path.
    # The checkpoints in test_data is saved with StandardSave. The test is to
    # verify the checkpoint manager with PyTreeRestore can still restore the
    # checkpoints saved with StandardSave.
    ckpt_manager = checkpoint_manager.CheckpointManager(
        os.path.join(os.path.dirname(__file__), ckpt_path)
    )
    model, _ = create_sharded_model(TestModel, nnx.Rngs(0), self.mesh)
    expected_state = nnx.state(model)
    # Change the model state.
    changed_state = jax.tree.map(lambda x: x + 1, nnx.state(model))
    nnx.update(model, changed_state)

    # Restore the model params.
    self.assertEqual(ckpt_manager.maybe_restore(model), (1, {}))
    # Check the model params are restored correctly.
    jax.tree.map_with_path(
        assert_close,
        expected_state,
        nnx.state(model),
    )


if __name__ == '__main__':
  absltest.main()
