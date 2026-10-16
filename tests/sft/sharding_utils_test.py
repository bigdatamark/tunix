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
# Unit tests for tunix.sft.sharding_utils.shard_input, which places an input
# array onto the current (context-manager) mesh with the batch dimension
# sharded over the given mesh axes, falling back to replication when the
# batch is not divisible, skipping work when already correctly sharded, and
# skipping non-fully-addressable (global) arrays.
# Side effects: at import time sets the XLA_FLAGS environment variable to
# force 8 host CPU devices (must happen before JAX backend initialization).
# KNOWN ISSUE: XLA_FLAGS is set after jax is imported; it only takes effect
# if the JAX backend has not been initialized yet at that point.
import os
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import jax.sharding as shd
import numpy as np
from tunix.sft import sharding_utils
from unittest import mock

# CPU environment setup to simulate multi device env.
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'


class ShardingUtilsTest(parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for shard_input across meshes and partition specs.

  @parameterized.named_parameters(
      dict(
          testcase_name='happy_path',
          batch_size=8,
          expected_pspec=shd.PartitionSpec(*('fsdp',)),
      ),
      dict(
          testcase_name='not_evenly_divisible',
          batch_size=9,
          expected_pspec=shd.PartitionSpec(),
      ),
      dict(
          testcase_name='not_enough_size',
          batch_size=7,
          expected_pspec=shd.PartitionSpec(),
      ),
  )
  def test_simple_reshard(self, batch_size, expected_pspec):
    # [FUNCTION DOC]
    # Verifies sharding a single-device (batch,4,8) array on a (devices,1) mesh
    # with axes ('fsdp','tp'): batch 8 gets PartitionSpec('fsdp'); batch 9 (not
    # divisible) and batch 7 (smaller than device count) get an empty
    # (replicated) PartitionSpec. Result is NamedSharding on the active mesh.
    # Inputs: batch_size and expected_pspec.
    device_cnt = jax.device_count()
    mesh = shd.Mesh(
        np.array(jax.devices()).reshape(device_cnt, 1),
        axis_names=('fsdp', 'tp'),
    )
    x = jnp.ones((batch_size, 4, 8))
    self.assertIsInstance(x.sharding, shd.SingleDeviceSharding)
    with mesh:
      x_sharded = sharding_utils.shard_input(x, ('fsdp',))
    assert isinstance(x_sharded.sharding, shd.NamedSharding)
    self.assertEqual(x_sharded.sharding.mesh, mesh)
    self.assertEqual(x_sharded.sharding.spec, expected_pspec)

  def test_reshard_across_meshes(self):
    # [FUNCTION DOC]
    # Verifies resharding an array that lives on mesh1 (first half of devices,
    # spec 'tp') onto mesh2 (second half, shape (n/2,1)) with 'fsdp': result
    # uses mesh2, spec 'fsdp', and each shard has first dim device_cnt/split_idx.
    device_cnt = jax.device_count()
    split_idx = device_cnt // 2
    mesh1 = shd.Mesh(
        np.array(jax.devices()[:split_idx]).reshape(1, split_idx),
        axis_names=('fsdp', 'tp'),
    )
    mesh2 = shd.Mesh(
        np.array(jax.devices()[split_idx:]).reshape(split_idx, 1),
        axis_names=('fsdp', 'tp'),
    )
    pspec = shd.PartitionSpec(*('tp',))
    x = jnp.ones((device_cnt, 4, 8))
    sharding = shd.NamedSharding(mesh1, pspec)
    x_sharded = jax.device_put(x, sharding)
    assert isinstance(x_sharded.sharding, shd.NamedSharding)
    self.assertEqual(x_sharded.sharding.mesh, mesh1)
    self.assertEqual(x_sharded.sharding.spec, pspec)

    with mesh2:
      x_resharded = sharding_utils.shard_input(x_sharded, ('fsdp',))
    assert isinstance(x_resharded.sharding, shd.NamedSharding)
    self.assertEqual(x_resharded.sharding.mesh, mesh2)
    self.assertEqual(
        x_resharded.sharding.spec,
        shd.PartitionSpec(*('fsdp',)),
    )
    self.assertEqual(
        x_resharded.addressable_data(0).shape, (device_cnt / split_idx, 4, 8)
    )

  def test_reshard_same_mesh_different_pspec(self):
    # [FUNCTION DOC]
    # Verifies that on the same mesh an array sharded with 'tp' is resharded to
    # the requested 'fsdp' PartitionSpec.
    device_cnt = jax.device_count()
    mesh = shd.Mesh(
        np.array(jax.devices()).reshape(1, device_cnt),
        axis_names=('fsdp', 'tp'),
    )
    pspec1 = shd.PartitionSpec(*('tp',))
    pspec2 = shd.PartitionSpec(*('fsdp',))
    x = jnp.ones((device_cnt, 4, 8))
    sharding = shd.NamedSharding(mesh, pspec1)
    x_sharded = jax.device_put(x, sharding)
    assert isinstance(x_sharded.sharding, shd.NamedSharding)
    self.assertEqual(x_sharded.sharding.mesh, mesh)
    self.assertEqual(x_sharded.sharding.spec, pspec1)

    with mesh:
      x_resharded = sharding_utils.shard_input(x_sharded, ('fsdp',))
    assert isinstance(x_resharded.sharding, shd.NamedSharding)
    self.assertEqual(x_resharded.sharding.mesh, mesh)
    self.assertEqual(x_resharded.sharding.spec, pspec2)

  def test_noop_on_already_sharded_input(self):
    # [FUNCTION DOC]
    # Verifies shard_input returns the very same object when the array already
    # has the target sharding, and a new object when a different axis ('tp') is
    # requested.
    device_cnt = jax.device_count()
    mesh = shd.Mesh(
        np.array(jax.devices()).reshape(1, device_cnt),
        axis_names=('fsdp', 'tp'),
    )
    pspec = shd.PartitionSpec('fsdp')
    x = jnp.ones((device_cnt, 4, 8))
    sharding = shd.NamedSharding(mesh, pspec)
    x_sharded = jax.device_put(x, sharding)
    with mesh:
      x_resharded = sharding_utils.shard_input(x_sharded, ('fsdp',))
    self.assertIs(x_sharded, x_resharded)

    # If resharding happened, a new obj will be created
    with mesh:
      x_resharded = sharding_utils.shard_input(x_sharded, ('tp',))
    self.assertIsNot(x_sharded, x_resharded)

  @mock.patch('tunix.sft.sharding_utils.jax.make_array_from_process_local_data')
  def test_global_array_noop(self, mock_make_array):
    # [FUNCTION DOC]
    # Verifies non-fully-addressable arrays are returned unchanged: a MagicMock
    # jax.Array with is_fully_addressable False and ndim 3 is passed; the result
    # must be the same object and jax.make_array_from_process_local_data (patched
    # to raise) must never be called. Input: the patched mock.
    mock_make_array.side_effect = ValueError(
        "device_put's first argument must be a fully addressable array, but got value with devices {TpuDevice(id=0..."
    )
    device_cnt = jax.device_count()
    mesh = shd.Mesh(
        np.array(jax.devices()).reshape(1, device_cnt),
        axis_names=('fsdp', 'tp'),
    )

    x = mock.MagicMock(spec=jax.Array)
    x.is_fully_addressable = False
    x.ndim = 3

    with mesh:
      x_resharded = sharding_utils.shard_input(x, ('fsdp',))

    self.assertIs(x, x_resharded)
    mock_make_array.assert_not_called()

if __name__ == '__main__':
  absltest.main()
