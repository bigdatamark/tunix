# Copyright 2026 Google LLC
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
# Unit tests for tunix.rl.algo_core advantage estimators.
# Covers RLOO (leave-one-out) advantages and GRPO group-normalized advantages.
# Inputs: hard-coded reward arrays and a fixed JAX PRNG key.
# Outputs: none; assertions fail on mismatch with precomputed expected values.
# Side effects: one test temporarily flips the global JAX config flag
# jax_threefry_partitionable to False and restores it via cleanup.

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from tunix.rl import algo_core


class AlgoCoreTest(absltest.TestCase):
  # [CLASS DOC]
  # Test case grouping checks of algo_core advantage functions.
  # No shared fixtures; each test builds its own reward arrays.

  def test_compute_rloo_advantages(self):
    # [FUNCTION DOC]
    # Verifies RLOO advantages for six rewards 1..6 split into groups of 3.
    # Each advantage is reward minus the mean of the other group members, so
    # each group yields -1.5, 0, 1.5. Raises AssertionError on mismatch.
    rewards = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    advantages = algo_core.compute_rloo_advantages(rewards, num_generations=3)
    expected_value = jnp.array([-1.5, 0.0, 1.5, -1.5, 0.0, 1.5])
    np.testing.assert_allclose(advantages, expected_value)

  def test_compute_rloo_advantages_low_generations(self):
    # [FUNCTION DOC]
    # Verifies that with num_generations equal to 1 (no peers to leave out),
    # RLOO advantages are all zeros with the same shape as the rewards.
    rewards = jnp.array([1.0, 2.0])
    advantages = algo_core.compute_rloo_advantages(rewards, num_generations=1)
    np.testing.assert_allclose(advantages, jnp.zeros_like(rewards))

  def test_grpo_compute_advantages(self):
    # [FUNCTION DOC]
    # Verifies GRPO advantages (per-group mean subtraction and std
    # normalization, groups of 3) on uniform random rewards of shape (1, 6)
    # drawn from PRNGKey(0) against precomputed values within 1e-3 tolerance.
    # Side effects: sets jax_threefry_partitionable to False so the random
    # values are reproducible; registers a cleanup restoring the prior value.
    prev_val = jax.config.jax_threefry_partitionable
    self.addCleanup(jax.config.update, 'jax_threefry_partitionable', prev_val)
    jax.config.update('jax_threefry_partitionable', False)
    self.assertFalse(jax.config.jax_threefry_partitionable)

    rng = jax.random.PRNGKey(0)
    rewards = jax.random.uniform(rng, shape=(1, 6))
    advantages = algo_core.compute_advantages(rewards, num_generations=3)
    expected_value = jnp.array(
        [[0.307498, -1.117636, 0.810138, 1.094526, -0.228671, -0.865855]]
    )
    np.testing.assert_allclose(advantages, expected_value, rtol=1e-3, atol=1e-3)


if __name__ == '__main__':
  absltest.main()
