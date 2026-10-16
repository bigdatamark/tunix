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
# Unit tests for tunix.rl.grpo.dapo_learner: DAPO shares GRPO's registered
# policy loss function but differs in loss aggregation and has no KL term;
# DAPOConfig post-init validation (epsilon_high at least epsilon; when
# overlong_buffer is enabled it must include non-None
# overlong_buffer_length, overlong_buffer_penalty and max_response_length,
# and the penalty must be non-negative); and reward_shaping, the overlong
# length penalty. No side effects.

from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax.numpy as jnp
from tunix.rl import function_registry as fr
from tunix.rl.grpo import dapo_learner as dapo_lib
from tunix.rl.grpo import grpo_learner as grpo_lib
from tunix.tests import test_common as tc


class DAPOlearnerTest(parameterized.TestCase):
  # [CLASS DOC]
  # Compares DAPO and GRPO loss behavior on identical inputs.

  def setUp(self):
    # [FUNCTION DOC]
    # HELPER: builds shared fixtures: mock model, pad_id 0, eos_id 1,
    # batch 2 x seq 4 prompt/completion ids, a completion mask with 3 and 2
    # valid tokens, advantages [0.5, -0.2], reference logps -0.2 and old
    # logps -0.15 everywhere.
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
    # tensors, with segment fields and sampler_is_weights set to None.
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

  def test_diff_loss(self):
    # [FUNCTION DOC]
    # Verifies DAPO and GRPO resolve to the same registered loss function,
    # yet with temperature 1.0 on a toy transformer the DAPO loss (with
    # reference logps None) differs from the GRPO loss, both aux dicts have
    # "kl", and DAPO's kl is 0.0.
    dapo_config = dapo_lib.DAPOConfig()
    grpo_config = grpo_lib.GRPOConfig()
    dapo_config.temperature = 1.0
    grpo_config.temperature = 1.0

    dapo_loss_fn_impl = fr.default_registry.get(
        "policy_loss_fn", dapo_config.policy_loss_fn
    )
    grpo_loss_fn_impl = fr.default_registry.get(
        "policy_loss_fn", grpo_config.policy_loss_fn
    )

    # Test that the functions is same
    self.assertEqual(dapo_loss_fn_impl, grpo_loss_fn_impl)

    # Create the same input for both functions
    train_example = self.create_train_example()
    pad_id = self.pad_id
    eos_id = self.eos_id
    vocab = tc.MockVocab()
    model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )

    # Call DAPO loss function (DAPO sets ref_per_token_logps to None as it doesn't fetch it)
    dapo_train_example = self.create_train_example()
    dapo_train_example.ref_per_token_logps = None
    dapo_loss, dapo_aux = dapo_loss_fn_impl(
        model, dapo_train_example, dapo_config, pad_id, eos_id
    )

    # Call GRPO loss function
    grpo_loss, grpo_aux = grpo_loss_fn_impl(
        model, train_example, grpo_config, pad_id, eos_id
    )

    # Assert that the loss values are different
    self.assertNotEqual(
        dapo_loss.item(),
        grpo_loss.item(),
        msg=(
            "DAPO and GRPO loss values should be different for the same input"
            " due to different loss aggregation logics."
        ),
    )

    self.assertIn("kl", dapo_aux)
    self.assertIn("kl", grpo_aux)
    self.assertEqual(dapo_aux["kl"], 0.0)  # DAPO does not have KL term.


class TestDAPOConfigPostInit(parameterized.TestCase):
  # [CLASS DOC]
  # Tests for DAPOConfig post-init validation.

  def test_valid_default(self):
    # [FUNCTION DOC]
    # Verifies default DAPOConfig construction raises no ValueError.
    """Tests that default values pass validation."""
    try:
      dapo_lib.DAPOConfig()
    except ValueError as e:
      self.fail(f"DAPOConfig raised ValueError on default initialization: {e}")

  @parameterized.named_parameters(
      dict(testcase_name="custom_epsilons", epsilon=0.1, epsilon_high=0.15),
      dict(testcase_name="epsilons_equal", epsilon=0.1, epsilon_high=0.1),
      dict(
          testcase_name="buffer_disabled",
          overlong_buffer={"enable": False},
      ),
      dict(testcase_name="buffer_none", overlong_buffer=None),
      dict(
          testcase_name="valid_buffer",
          overlong_buffer={
              "enable": True,
              "overlong_buffer_length": 2000,
              "overlong_buffer_penalty": 0.5,
              "max_response_length": 10000,
          },
      ),
  )
  def test_valid_configurations(self, **kwargs):
    # [FUNCTION DOC]
    # Verifies valid configs construct: epsilon_high above or equal to
    # epsilon, overlong_buffer disabled or None, or a complete buffer dict.
    """Tests various valid custom configurations."""
    try:
      dapo_lib.DAPOConfig(**kwargs)
    except ValueError as e:
      self.fail(f"DAPOConfig raised ValueError for valid case {kwargs}: {e}")

  @parameterized.named_parameters(
      dict(
          testcase_name="invalid_epsilon_high",
          config_kwargs=dict(epsilon=0.2, epsilon_high=0.1),
          expected_regex=(
              "epsilon_high must be greater than or equal to epsilon."
          ),
      ),
      dict(
          testcase_name="buffer_missing_length",
          config_kwargs=dict(
              overlong_buffer={
                  "enable": True,
                  "overlong_buffer_penalty": 1.0,
                  "max_response_length": 20480,
              }
          ),
          expected_regex=(
              "overlong_buffer is enabled but missing.*overlong_buffer_length.*"
          ),
      ),
      dict(
          testcase_name="buffer_missing_penalty",
          config_kwargs=dict(
              overlong_buffer={
                  "enable": True,
                  "overlong_buffer_length": 4096,
                  "max_response_length": 20480,
              }
          ),
          expected_regex=(
              "overlong_buffer is enabled but missing"
              ".*overlong_buffer_penalty.*"
          ),
      ),
      dict(
          testcase_name="buffer_missing_max_length",
          config_kwargs=dict(
              overlong_buffer={
                  "enable": True,
                  "overlong_buffer_length": 4096,
                  "overlong_buffer_penalty": 1.0,
              }
          ),
          expected_regex=(
              "overlong_buffer is enabled but missing.*max_response_length.*"
          ),
      ),
      dict(
          testcase_name="buffer_length_is_none",
          config_kwargs=dict(
              overlong_buffer={
                  "enable": True,
                  "overlong_buffer_length": None,
                  "overlong_buffer_penalty": 1.0,
                  "max_response_length": 20480,
              }
          ),
          expected_regex=(
              "overlong_buffer is enabled but missing.*overlong_buffer_length.*"
          ),
      ),
      dict(
          testcase_name="negative_penalty",
          config_kwargs=dict(
              overlong_buffer={
                  "enable": True,
                  "overlong_buffer_length": 4096,
                  "overlong_buffer_penalty": -0.5,
                  "max_response_length": 20480,
              }
          ),
          expected_regex="overlong_buffer_penalty must be non-negative",
      ),
  )
  def test_invalid_configurations(self, config_kwargs, expected_regex):
    # [FUNCTION DOC]
    # Verifies ValueError with a matching message for epsilon_high below
    # epsilon, an enabled buffer missing (or None for) any required key, and
    # a negative overlong penalty.
    """Tests various invalid configurations that should raise ValueError."""
    with self.assertRaisesRegex(ValueError, expected_regex):
      dapo_lib.DAPOConfig(**config_kwargs)


class RewardShapingTest(parameterized.TestCase):
  # [CLASS DOC]
  # Tests for dapo_learner.reward_shaping overlong penalty.

  def setUp(self):
    # [FUNCTION DOC]
    # HELPER: creates a MagicMock cluster whose Mode attribute is passed as
    # the mode argument.
    super().setUp()
    self.mock_cluster = mock.MagicMock()

  def test_raises_error_on_none_buffer(self):
    # [FUNCTION DOC]
    # Verifies reward_shaping raises ValueError when overlong_buffer is None.
    with self.assertRaisesRegex(
        ValueError, "reward_shaping is called but with empty overlong_buffer."
    ):

      dapo_lib.reward_shaping(
          prompts=["test prompt"],
          completions=["test completion"],
          mode=self.mock_cluster.Mode,
          overlong_buffer=None,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="under_length",
          lengths=[70],
          expected_scores=[0.0],
      ),
      dict(
          testcase_name="at_expected_length",
          lengths=[80],
          expected_scores=[0.0],
      ),
      dict(
          testcase_name="in_buffer_zone",
          lengths=[90],
          expected_scores=[-5.0],
      ),
      dict(
          testcase_name="at_max_length",
          lengths=[100],
          expected_scores=[-10.0],
      ),
      dict(
          testcase_name="over_max_length",
          lengths=[110],
          expected_scores=[-15.0],
      ),
      dict(
          testcase_name="mixed_lengths",
          lengths=[70, 80, 90, 100, 110],
          expected_scores=[0.0, 0.0, -5.0, -10.0, -15.0],
      ),
      dict(
          testcase_name="zero_penalty",
          lengths=[110],
          expected_scores=[0.0],
          penalty=0,
      ),
  )
  def test_reward_scores(self, lengths, expected_scores, penalty=10):
    # [FUNCTION DOC]
    # With max_response_length 100 and buffer length 20 (expected length
    # 80), verifies the score per completion is 0 up to length 80 and
    # otherwise negative (length - 80) / 20 times penalty, unclipped beyond
    # the max (110 gives -1.5 times penalty); penalty 0 gives 0.
    # Completion length is measured in characters here.
    completions = ["a" * length for length in lengths]
    overlong_buffer = {
        "overlong_buffer_length": 20,
        "overlong_buffer_penalty": penalty,
        "max_response_length": 100,
    }
    # expected_response_length = 100 - 20 = 80

    scores = dapo_lib.reward_shaping(
        prompts=[""] * len(completions),
        completions=completions,
        mode=self.mock_cluster.Mode,
        overlong_buffer=overlong_buffer,
    )

    self.assertSequenceAlmostEqual(expected_scores, scores, places=4)


if __name__ == "__main__":
  absltest.main()
