# [MODULE DOC]
# Unit tests for tunix.rl.rl_learner.RLLearner micro-batch size resolution.
# Defines a minimal concrete RLLearner subclass and drives train() against a
# MagicMock RL cluster to check how rollout and compute-logps micro-batch sizes
# are derived from (or validated against) mini_batch_size and
# train_micro_batch_size in RLTrainingConfig.
# Inputs: parameterized config values. Outputs: assertions only.
# Side effects: none beyond in-memory mocks.
"""Tests for `rl_learner`."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import optax
from tunix.rl import algorithm_config as algo_config_lib
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import rl_learner

class DummyModel(nnx.Module):
  # [CLASS DOC]
  # HELPER: empty nnx module used as stand-in actor and rollout model.
  pass


class DummyConfig(algo_config_lib.AlgorithmConfig):
  # [CLASS DOC]
  # HELPER: AlgorithmConfig subclass with all default values, no additions.
  pass


class DummyLearner(rl_learner.RLLearner[DummyConfig]):
  # [CLASS DOC]
  # HELPER: minimal concrete RLLearner implementing the abstract hooks with
  # trivial behavior so that train() can run against a mocked cluster.

  def _generate_and_compute_advantage(self, training_input, mode):
    # [FUNCTION DOC]
    # HELPER: no-op; returns None regardless of training_input and mode.
    pass

  def _compute_trajectory_ids(self, example, steps):
    # [FUNCTION DOC]
    # HELPER: returns a list of empty strings, one per entry in
    # example['prompts']; steps is ignored.
    return [''] * len(example['prompts'])

  def _num_iterations(self):
    # [FUNCTION DOC]
    # HELPER: returns 1 (single policy-update iteration per batch).
    return 1

  def _num_generations(self):
    # [FUNCTION DOC]
    # HELPER: returns 1 (one completion per prompt).
    return 1


class RLLearnerTest(parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized tests of micro-batch size defaults and validation in
  # RLLearner.

  @parameterized.named_parameters(
      ('1', None, None, None, None, [32, 32], False),
      ('2', 8, None, None, None, [8, 8], False),
      ('3', 8, 2, None, None, [2, 2], False),
      ('4', 8, 4, 4, 4, [4, 4], False),
      ('5', 8, 4, 3, 4, [], True),
      ('6', 8, 4, 4, 3, [], True),
  )
  def test_micro_batching(
      self,
      mini_batch_size,
      train_micro_batch_size,
      rollout_micro_batch_size,
      compute_logps_micro_batch_size,
      expected_values,
      expect_failure,
  ):
    # [FUNCTION DOC]
    # Builds an RLTrainingConfig (sgd optimizer, max_steps 1) with the given
    # mini/train/rollout/compute-logps micro-batch sizes, a MagicMock cluster
    # exposing that config and dummy models, and a DummyLearner with a
    # constant reward function, then trains on one batch of 32 prompts.
    # If expect_failure, asserts train raises ValueError (cases where rollout
    # or logps micro-batch size differs from train micro-batch size).
    # Otherwise asserts the learner's resolved rollout and compute-logps
    # micro-batch sizes equal expected_values: they default to
    # train_micro_batch_size, else mini_batch_size, else full batch size 32.
    config = rl_cluster_lib.RLTrainingConfig(
        actor_optimizer=optax.sgd(1e-3),
        mini_batch_size=mini_batch_size,
        train_micro_batch_size=train_micro_batch_size,
        rollout_micro_batch_size=rollout_micro_batch_size,
        compute_logps_micro_batch_size=compute_logps_micro_batch_size,
        eval_every_n_steps=1,
        max_steps=1,
    )

    actor_model = DummyModel()
    rollout_model = DummyModel()
    mock_cluster = mock.MagicMock()
    mock_cluster.actor_trainer.model = actor_model
    mock_cluster.rollout.model.return_value = rollout_model
    mock_cluster.cluster_config.training_config = config
    mock_cluster.actor_trainer.train_steps = 0
    mock_cluster.actor_trainer.iter_steps = 0

    learner = DummyLearner(
        rl_cluster=mock_cluster,
        algo_config=DummyConfig(),
        reward_fns=lambda prompts, completions, **kwargs: [1.0] * len(prompts),
    )

    full_batch_size = 32
    train_ds = [{'prompts': [''] * full_batch_size}]

    if expect_failure:
      with self.assertRaises(ValueError):
        learner.train(train_ds)
    else:
      learner.train(train_ds)
      (
          expected_rollout_micro,
          expected_compute_logps_micro,
      ) = expected_values

      self.assertEqual(
          learner._rollout_micro_batch_size, expected_rollout_micro
      )
      self.assertEqual(
          learner._compute_logps_micro_batch_size, expected_compute_logps_micro
      )


if __name__ == '__main__':
  absltest.main()
