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
# Unit tests for the Dr. GRPO variant (tunix.rl.grpo.drgrpo_learner):
# DrGRPOConfig defaults, the "drgrpo" advantage estimator (group mean
# subtraction without std scaling), the registered drgrpo policy loss, and
# algo_core.compute_drgrpo_advantages against fixed expected values.
# Side effects: at import, globally sets jax_threefry_partitionable to False.
# KNOWN ISSUE: algo_core is imported with an unused-import pylint disable but
# is actually used directly in test_compute_advantages.

from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.rl import algo_core  # pylint: disable=unused-import
from tunix.rl import function_registry as fr
from tunix.rl.grpo import drgrpo_learner as drgrpo_lib
from tunix.rl.grpo import grpo_learner as grpo_lib
from tunix.tests import test_common as tc

jax.config.update("jax_threefry_partitionable", False)


class DrGRPOlearnerTest(parameterized.TestCase):
  # [CLASS DOC]
  # Tests for Dr. GRPO config, advantage estimation and loss function.

  def setUp(self):
    # [FUNCTION DOC]
    # HELPER: builds shared fixtures: a MagicMock model, pad_id 0, eos_id 1,
    # batch 2 x seq 4 prompt ids (zeros) and completion ids (ones), a float
    # completion mask with 3 and 2 valid tokens, advantages [0.5, -0.2],
    # reference logps all -0.2 and old logps all -0.15.
    super().setUp()
    self.mock_model = mock.MagicMock()
    self.pad_id = 0
    self.eos_id = 1

    # Common data shapes
    self.batch_size = 2
    self.seq_len = 4
    self.prompt_ids = jnp.zeros(
        (self.batch_size, self.seq_len), dtype=jnp.int32
    )
    self.completion_ids = jnp.ones(
        (self.batch_size, self.seq_len), dtype=jnp.int32
    )
    self.completion_mask = jnp.array(
        [[1, 1, 1, 0], [1, 1, 0, 0]], dtype=jnp.float32
    )
    self.advantages = jnp.array([0.5, -0.2], dtype=jnp.float32)
    self.ref_per_token_logps = (
        jnp.ones_like(self.completion_ids, dtype=jnp.float32) * -0.2
    )
    self.old_per_token_logps = (
        jnp.ones_like(self.completion_ids, dtype=jnp.float32) * -0.15
    )

  def create_train_example(self):
    # [FUNCTION DOC]
    # HELPER: returns a MagicMock training example carrying the setUp
    # tensors as attributes, with segment_ids, segment_positions and
    # sampler_is_weights set to None.
    example = mock.MagicMock()
    example.prompt_ids = self.prompt_ids
    example.completion_ids = self.completion_ids
    example.completion_mask = self.completion_mask
    example.advantages = self.advantages
    example.ref_per_token_logps = self.ref_per_token_logps
    example.old_per_token_logps = self.old_per_token_logps
    example.segment_ids = None
    example.segment_positions = None
    example.sampler_is_weights = None
    return example

  def test_create_config(self):
    # [FUNCTION DOC]
    # Verifies DrGRPOConfig fixes algo_variant and advantage_estimator to
    # "drgrpo" and loss_agg_mode to "sequence-mean-token-scale", while
    # storing the passed epsilon, num_generations, num_iterations and beta.
    drgrpo_config = drgrpo_lib.DrGRPOConfig(
        epsilon=0.1, num_generations=5, num_iterations=3, beta=0.123
    )
    self.assertEqual(drgrpo_config.algo_variant, "drgrpo")
    self.assertEqual(drgrpo_config.advantage_estimator, "drgrpo")
    self.assertEqual(drgrpo_config.loss_agg_mode, "sequence-mean-token-scale")
    self.assertEqual(drgrpo_config.num_generations, 5)
    self.assertEqual(drgrpo_config.num_iterations, 3)
    self.assertEqual(drgrpo_config.epsilon, 0.1)
    self.assertEqual(drgrpo_config.beta, 0.123)

  def test_drgrpo_advantage_estimator(self):
    # [FUNCTION DOC]
    # Verifies, for three groups of size 2 with identical within-group
    # spread, that Dr. GRPO advantages equal GRPO advantages multiplied by
    # (sample std with ddof 1 of the group plus 1e-6), i.e. Dr. GRPO skips
    # std normalization. Estimators are fetched from the function registry.
    drgrpo_config = drgrpo_lib.DrGRPOConfig()
    grpo_config = grpo_lib.GRPOConfig()

    grpo_advantage_estimator = fr.get_advantage_estimator(
        grpo_config.advantage_estimator
    )
    drgrpo_advantage_estimator = fr.get_advantage_estimator(
        drgrpo_config.advantage_estimator
    )

    # Batch size 3 with group size 2.
    n_generations = 2
    rewards = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    num_generations = n_generations
    grpo_advantages = grpo_advantage_estimator(
        rewards=rewards.ravel(), num_generations=num_generations
    )
    drgrpo_advantages = drgrpo_advantage_estimator(
        rewards=rewards.ravel(), num_generations=num_generations
    )
    # Dr. GRPO advantages are not scaled by the standard deviation.
    # Std. across groups above is the same by construction.
    std_factor = jnp.array([1.0, 2.0]).std(ddof=1) + 1e-6
    np.testing.assert_allclose(grpo_advantages * std_factor, drgrpo_advantages)

  def test_drgrpo_loss_fn(self):
    # [FUNCTION DOC]
    # Verifies the registered drgrpo policy loss, applied to a toy
    # transformer and the mock training example with temperature 1.0,
    # returns a finite loss and an aux dict containing "kl".
    drgrpo_config = drgrpo_lib.DrGRPOConfig()
    drgrpo_config.temperature = 1.0

    drgrpo_loss_fn_impl = fr.default_registry.get(
        "policy_loss_fn", drgrpo_config.policy_loss_fn
    )

    # Create the same input for both functions
    train_example = self.create_train_example()
    pad_id = self.pad_id
    eos_id = self.eos_id
    vocab = tc.MockVocab()
    model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )

    # Call DrGRPO loss function
    drgrpo_loss, drgrpo_aux = drgrpo_loss_fn_impl(
        model, train_example, drgrpo_config, pad_id, eos_id
    )

    self.assertIn("kl", drgrpo_aux)
    self.assertTrue(jnp.isfinite(drgrpo_loss).all())

  def test_compute_advantages(self):
    # [FUNCTION DOC]
    # Verifies compute_drgrpo_advantages on 6 fixed rewards in groups of 3
    # (reward minus group mean) matches precomputed values within 1e-5.
    rewards = jnp.array(
        [[0.57450044, 0.09968603, 0.7419659, 0.8941783, 0.59656656, 0.45325184]]
    )
    advantages = algo_core.compute_drgrpo_advantages(rewards, num_generations=3)
    expected_array = jnp.array([
        [0.10245, -0.372365, 0.269915, 0.246179, -0.051432, -0.194747],
    ])
    np.testing.assert_allclose(advantages, expected_array, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  absltest.main()
