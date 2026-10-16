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
# Unit tests for tunix.rl.utils: mesh discovery from a pytree's shardings,
# weight-sharing detection between models, slice chunking, batch slicing of
# nested pytrees, merging micro-batches by concatenation, critic model
# creation from an actor, moving params to a memory kind, and sequence
# packing of TrainExamples into fixed token budgets with segment ids and
# positions. Uses 4 simulated CPU devices.
# Side effects: sets XLA_FLAGS env var at import and calls
# chex.set_n_cpu_devices in setUp.
# KNOWN ISSUE: XLA_FLAGS is set after jax is imported, so it may have no
# effect; the device count relies on chex.set_n_cpu_devices instead.

import os
from absl.testing import absltest
import chex
from flax import nnx
import jax
from jax import sharding
import jax.numpy as jnp
import numpy as np
from tunix.rl import common
from tunix.rl import utils
from tunix.tests import test_common as tc

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'


class UtilsTest(absltest.TestCase):
  # [CLASS DOC]
  # Tests for tunix.rl.utils helpers.

  def setUp(self):
    # [FUNCTION DOC]
    # HELPER: forces 4 CPU devices via chex and records jax.device_count.
    super().setUp()
    self.num_cpus = 4
    chex.set_n_cpu_devices(self.num_cpus)
    self.device_count = jax.device_count()

  def test_get_pytree_mesh_info(self):
    # [FUNCTION DOC]
    # Verifies get_pytree_mesh_info returns the mesh a LoRA toy model was
    # sharded on (two disjoint half-device meshes give distinct results) and
    # None for an unsharded model.
    mesh1 = sharding.Mesh(
        np.array(jax.devices()[: self.device_count // 2]).reshape(
            1, self.device_count // 2
        ),
        ('fsdp', 'tp'),
    )
    model1 = tc.get_lora_model(
        tc.ToyTransformer(
            config=tc.ModelConfig(vocab_size=tc.MockVocab().GetPieceSize()),
            rngs=nnx.Rngs(0),
        ),
        mesh=mesh1,
    )
    self.assertEqual(utils.get_pytree_mesh_info(nnx.state(model1)), mesh1)

    mesh2 = sharding.Mesh(
        np.array(jax.devices()[self.device_count // 2 :]).reshape(
            1, self.device_count // 2
        ),
        ('fsdp', 'tp'),
    )
    model2 = tc.get_lora_model(
        tc.ToyTransformer(
            config=tc.ModelConfig(vocab_size=tc.MockVocab().GetPieceSize()),
            rngs=nnx.Rngs(0),
        ),
        mesh=mesh2,
    )
    self.assertEqual(utils.get_pytree_mesh_info(nnx.state(model2)), mesh2)

    self.assertNotEqual(mesh1, mesh2)

    model3 = tc.get_lora_model(
        tc.ToyTransformer(
            config=tc.ModelConfig(vocab_size=tc.MockVocab().GetPieceSize()),
            rngs=nnx.Rngs(0),
        ),
    )
    self.assertIsNone(utils.get_pytree_mesh_info(nnx.state(model3)))

  def test_is_sharing_weights(self):
    # [FUNCTION DOC]
    # Verifies is_sharing_weights is False for two independently built
    # models (even with the same seed) and True for a model and its
    # nnx.clone, even though their state objects are distinct.
    # KNOWN ISSUE: the expectation that a clone "shares weights" suggests
    # the helper compares by value or underlying buffers, not identity;
    # the name may mislead.
    m1 = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=tc.MockVocab().GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    m2 = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=tc.MockVocab().GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    m3 = nnx.clone(m1)
    self.assertIsNot(nnx.state(m1), nnx.state(m2))
    self.assertIsNot(nnx.state(m1), nnx.state(m3))
    self.assertIsNot(nnx.state(m2), nnx.state(m3))
    self.assertFalse(utils.is_sharing_weights(m1, m2))
    self.assertFalse(utils.is_sharing_weights(m2, m3))
    self.assertTrue(utils.is_sharing_weights(m1, m3))

  def test_chunk_slices_by_size(self):
    # [FUNCTION DOC]
    # Verifies chunk_slices_by_size(stop 5, step 2) yields slices covering
    # [0,2), [2,4), [4,5).
    x = [0, 1, 2, 3, 4]
    y = [x[s] for s in utils.chunk_slices_by_size(stop=len(x), step=2)]
    self.assertEqual(y, [[0, 1], [2, 3], [4]])

  def test_get_batch_slice(self):
    # [FUNCTION DOC]
    # Verifies get_batch_slice applies a slice to the leading dim of every
    # leaf in a nested dict pytree.
    x = {
        'a': np.array([[1], [2], [3], [4], [5], [6]]),
        'b': {'c': np.array([[7], [8], [9], [10], [11], [12]])},
    }
    y = [
        utils.get_batch_slice(x, s)
        for s in utils.chunk_slices_by_size(stop=6, step=2)
    ]
    expected = [
        {'a': np.array([[1], [2]]), 'b': {'c': np.array([[7], [8]])}},
        {'a': np.array([[3], [4]]), 'b': {'c': np.array([[9], [10]])}},
        {'a': np.array([[5], [6]]), 'b': {'c': np.array([[11], [12]])}},
    ]
    jax.tree_util.tree_map(np.testing.assert_array_equal, expected, y)

  def test_merge_micro_batches(self):
    # [FUNCTION DOC]
    # Verifies merge_micro_batches concatenates matching leaves across a
    # list of batch dicts: lists are extended and arrays concatenated, with
    # varying per-batch lengths allowed, recursing into nested dicts.
    batches = [
        {
            'a': [1, 2],
            'b': {'c': np.array([3, 4]), 'd': np.array([5])},
            'e': np.array([6, 7]),
        },
        {
            'a': [10, 11],
            'b': {'c': np.array([12, 13]), 'd': np.array([14, 15])},
            'e': np.array([16]),
        },
    ]
    merged = utils.merge_micro_batches(batches)
    self.assertEqual(merged['a'], [1, 2, 10, 11])
    jax.tree_util.tree_map(
        np.testing.assert_array_equal,
        merged['b'],
        {'c': np.array([3, 4, 12, 13]), 'd': np.array([5, 14, 15])},
    )
    jax.tree_util.tree_map(
        np.testing.assert_array_equal, merged['e'], np.array([6, 7, 16])
    )

  def test_create_critic_model(self):
    # [FUNCTION DOC]
    # Verifies create_critic_model turns an actor toy transformer into a
    # model whose output for a (2, 3) token batch with causal mask has
    # shape (2, 3, 1), i.e. a scalar value per token.
    actor_model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=tc.MockVocab().GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    critic_model = utils.create_critic_model(actor_model)

    x = jnp.array([[1, 2, 3], [4, 5, 6]])
    positions = jnp.arange(x.shape[1])
    attn_mask = common.make_causal_attn_mask(jnp.ones_like(x))
    out, _ = critic_model(x, positions, None, attn_mask)
    self.assertEqual(out.shape, (2, 3, 1))

  def test_put_params_on_memory_kind(self):
    # [FUNCTION DOC]
    # Verifies put_params_on_memory_kind moves all leaves to pinned_host,
    # returns the same object when already there, returns an empty tree
    # unchanged, and raises ValueError "memory_kind must be one of" for an
    # unknown kind.
    # Test valid memory kind
    params = {'a': jnp.array([1.0, 2.0]), 'b': jnp.array([3.0])}
    updated_params = utils.put_params_on_memory_kind(params, 'pinned_host')
    self.assertEqual(
        jax.tree.map(lambda x: x.sharding.memory_kind, updated_params),
        {'a': 'pinned_host', 'b': 'pinned_host'},
    )

    # Test already on requested memory kind
    updated_params_2 = utils.put_params_on_memory_kind(
        updated_params, 'pinned_host'
    )
    self.assertIs(updated_params, updated_params_2)

    # Test empty tree
    empty_params = {}
    updated_empty = utils.put_params_on_memory_kind(empty_params, 'device')
    self.assertEqual(updated_empty, {})

    # Test invalid memory kind
    with self.assertRaisesRegex(ValueError, 'memory_kind must be one of'):
      utils.put_params_on_memory_kind(params, 'invalid_kind')

  def test_pack_sequences(self):
    # [FUNCTION DOC]
    # Verifies pack_sequences greedily packs sequences (prompt plus
    # completion lengths 5, 3, 7) into rows of budget 10: the first pack
    # holds the first two, the second holds the third. In each pack prompt
    # ids are empty, completion ids hold concatenated prompt and completion
    # tokens padded to 10, segment ids are 1-based per sequence with 0 for
    # padding, positions restart at 0 per sequence (0 for padding), and the
    # completion mask is 1 only on completion tokens.

    def _create_mock_train_example(
        prompt_len: int, completion_len: int
    ) -> common.TrainExample:
      # [FUNCTION DOC]
      # HELPER: builds a single-row TrainExample with prompt ids of ones,
      # completion ids of twos, all-ones masks, advantage 1.5 and all
      # optional fields None.
      return common.TrainExample(
          prompt_ids=jnp.ones((1, prompt_len), dtype=jnp.int32),
          prompt_mask=jnp.ones((1, prompt_len), dtype=jnp.int32),
          completion_ids=jnp.ones((1, completion_len), dtype=jnp.int32) * 2,
          completion_mask=jnp.ones((1, completion_len), dtype=jnp.int32),
          advantages=jnp.array([1.5], dtype=jnp.float32),
          ref_per_token_logps=None,
          old_per_token_logps=None,
          segment_ids=None,
          segment_positions=None,
      )

    # 3 sequences with lengths (P+C): (2+3=5), (1+2=3), (3+4=7)
    example1 = _create_mock_train_example(2, 3)
    example2 = _create_mock_train_example(1, 2)
    example3 = _create_mock_train_example(3, 4)

    item_iterator = iter([[example1], [example2], [example3]])

    # Budget of 10. We expect item 1 (5) and item 2 (3) to fit in the first pack (8).
    # Item 3 (7) will go to the second pack (because 8+7 > 10).
    packed_iterator = utils.pack_sequences(
        item_iterator, max_token_budget=10, pad_id=0
    )

    packed_batches = list(packed_iterator)
    with self.subTest('pack_counts'):
      self.assertLen(packed_batches, 2)

    pack1 = packed_batches[0][0]
    # Segment IDs should be (5 ones, 3 twos, 2 padding zeros)
    expected_segments_1 = jnp.array(
        [[1] * 5 + [2] * 3 + [0] * 2], dtype=jnp.int32
    )
    # Positions should be (0..4, 0..2, 0, 0)
    expected_positions_1 = jnp.array(
        [[0, 1, 2, 3, 4, 0, 1, 2, 0, 0]], dtype=jnp.int32
    )
    # Completion mask should be 0 for prompts, 1 for completions, 0 for padding
    # Seq 1: 2 prompts (0), 3 completions (1)
    # Seq 2: 1 prompt (0), 2 completions (1)
    expected_mask_1 = jnp.array(
        [[0, 0, 1, 1, 1, 0, 1, 1, 0, 0]], dtype=jnp.int32
    )

    with self.subTest('pack1_contents'):
      self.assertEqual(pack1.prompt_ids.shape, (1, 0))  # prompt_ids is empty
      self.assertEqual(pack1.completion_ids.shape, (1, 10))  # filled + padded
      self.assertEqual(pack1.segment_ids.shape, (1, 10))
      self.assertEqual(pack1.segment_positions.shape, (1, 10))
      np.testing.assert_array_equal(pack1.segment_ids, expected_segments_1)
      np.testing.assert_array_equal(pack1.segment_positions, expected_positions_1)
      np.testing.assert_array_equal(pack1.completion_mask, expected_mask_1)

    pack2 = packed_batches[1][0]
    expected_segments_2 = jnp.array([[1] * 7 + [0] * 3], dtype=jnp.int32)
    # Positions should be (0..6, 0, 0, 0)
    expected_positions_2 = jnp.array(
        [[0, 1, 2, 3, 4, 5, 6, 0, 0, 0]], dtype=jnp.int32
    )
    # Completion mask: 3 prompts (0), 4 completions (1), 3 padding (0)
    expected_mask_2 = jnp.array(
        [[0, 0, 0, 1, 1, 1, 1, 0, 0, 0]], dtype=jnp.int32
    )

    with self.subTest('pack2_contents'):
      self.assertEqual(pack2.prompt_ids.shape, (1, 0))
      self.assertEqual(pack2.completion_ids.shape, (1, 10))
      self.assertEqual(pack2.segment_ids.shape, (1, 10))
      self.assertEqual(pack2.segment_positions.shape, (1, 10))
      np.testing.assert_array_equal(pack2.segment_ids, expected_segments_2)
      np.testing.assert_array_equal(pack2.segment_positions, expected_positions_2)
      np.testing.assert_array_equal(pack2.completion_mask, expected_mask_2)


if __name__ == '__main__':
  absltest.main()
