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
# Unit tests for tunix.rl.reward_manager. SequenceRewardManager calls each
# reward function with (prompts, completions, plus kwargs), injecting
# algorithm-config fields whose names match the function's signature,
# sums per-sample rewards across functions with NaN treated as 0, and
# returns a dict with "rewards" (per-sample array) and "log_metrics"
# (per-function "rewards/<fn name>", "rewards/sum", "rewards/min",
# "rewards/max" and others; each value is a list whose first element is the
# per-sample array). It raises RuntimeError for None results or length
# mismatches and ValueError for an empty reward_fns list.
# AgenticSequenceRewardManager additionally logs "trajectory_rewards/*"
# metrics from passed trajectory_rewards and allows reward_fns None.
# No side effects.

import dataclasses
import inspect
from typing import Any, List
from unittest import mock
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import numpy.testing as npt
from tunix.rl import algorithm_config as algo_config_lib
from tunix.rl import reward_manager


# --- Test Reward Functions ---
def len_reward(
    prompts: List[str], completions: List[str], **kwargs: Any
) -> List[float]:
  # [FUNCTION DOC]
  # HELPER: reward fn returning the character length of each completion as
  # a float.
  del prompts, kwargs  # Unused
  res = [float(len(c)) for c in completions]
  return res


len_reward.__name__ = "len_reward"


def prompt_len_reward(
    prompts: List[str],
    completions: List[str],
    custom_param: float = 1.0,
    **kwargs: Any,
) -> List[float]:
  # [FUNCTION DOC]
  # HELPER: reward fn returning custom_param (default 1.0) times the length
  # of each prompt; custom_param is meant to be injected from the config.
  del completions, kwargs  # Unused
  res =[custom_param * len(p) for p in prompts]
  return res


prompt_len_reward.__name__ = "prompt_len_reward"


def nan_reward(
    prompts: List[str], completions: List[str], **kwargs: Any
) -> List[float]:
  # [FUNCTION DOC]
  # HELPER: reward fn returning NaN for every prompt.
  del completions, kwargs  # Unused
  return [np.nan] * len(prompts)


nan_reward.__name__ = "nan_reward"


@dataclasses.dataclass(slots=True, kw_only=True)
class TestAlgoConfig(algo_config_lib.AlgorithmConfig):
  # [CLASS DOC]
  # HELPER: slotted kw-only AlgorithmConfig adding reward_manager
  # ("sequence-level") and custom_param (2.0), the latter for injection
  # into reward fns.
  # KNOWN ISSUE: the "Test" prefix may cause pytest to try collecting this
  # dataclass as a test class (it emits a collection warning).
  """Test Algorithm Config."""

  reward_manager: str = "sequence-level"
  custom_param: float = 2.0


# --- Test Class ---
class SequenceRewardManagerTest(parameterized.TestCase):
  # [CLASS DOC]
  # Tests for SequenceRewardManager aggregation, metrics, param injection
  # and errors.

  def setUp(self):
    # [FUNCTION DOC]
    # HELPER: creates the test config, two prompts and two completions.
    super().setUp()
    self.test_algo_config = TestAlgoConfig()
    self.prompts = ["p1", "p22"]
    self.completions = ["c1_long", "c2"]

  def test_initialization(self):
    # [FUNCTION DOC]
    # Verifies a single callable reward_fns is wrapped into a list and the
    # config is stored.
    manager = reward_manager.SequenceRewardManager(
        reward_fns=len_reward,
        algo_config=self.test_algo_config,
    )
    self.assertEqual(manager.reward_fns, [len_reward])
    self.assertEqual(manager.algo_config, self.test_algo_config)

  def test_single_reward_fn(self):
    # [FUNCTION DOC]
    # Verifies rewards equal the single fn's outputs and exactly 7 log
    # metric entries are produced.
    manager = reward_manager.SequenceRewardManager(
        reward_fns=[len_reward],
        algo_config=self.test_algo_config,
    )
    rewards_info = manager(
        self.prompts,
        self.completions,
    )

    expected_rewards = np.array([float(len("c1_long")), float(len("c2"))])
    np.testing.assert_array_equal(rewards_info["rewards"], expected_rewards)
    self.assertLen(rewards_info["log_metrics"], 7)

  def test_multiple_reward_fns(self):
    # [FUNCTION DOC]
    # Verifies rewards are the per-sample sum of two fns (prompt_len_reward
    # receiving custom_param 2.0 from config); every "rewards/" metric holds
    # 2 per-sample values; sum and per-fn metrics match; and min/max are the
    # per-sample min/max across fns.
    # KNOWN ISSUE: the min/max loop iterates over the number of reward fns
    # (matrix rows) but uses it as a sample (column) index; it only works
    # because both counts are 2 here.
    manager = reward_manager.SequenceRewardManager(
        reward_fns=[len_reward, prompt_len_reward],
        algo_config=self.test_algo_config,
    )
    rewards_info = manager(
        self.prompts,
        self.completions,
    )

    # custom_param is 2.0 from test_algo_config
    r1 = np.array(len_reward(self.prompts, self.completions))
    r2 = np.array(
        prompt_len_reward(self.prompts, self.completions, custom_param=2.0)
    )
    expected_rewards = r1 + r2
    rewards_matrix = np.array([r1, r2])
    np.testing.assert_array_almost_equal(
        rewards_info["rewards"], expected_rewards
    )
    test_metrics = rewards_info["log_metrics"]
    for metric_name, v in test_metrics.items():
      if metric_name.startswith("rewards/"):
        self.assertLen(v[0], 2)
    npt.assert_allclose(
        test_metrics["rewards/sum"][0],
        expected_rewards,
        err_msg="rewards/sum mismatch",
    )
    npt.assert_allclose(
        test_metrics["rewards/len_reward"][0],
        r1,
        err_msg="rewards/len_reward mismatch",
    )
    npt.assert_allclose(
        test_metrics["rewards/prompt_len_reward"][0],
        r2,
        err_msg="rewards/prompt_len_reward mismatch",
    )
    for col_idx in range(rewards_matrix.shape[0]):
      npt.assert_allclose(
          test_metrics["rewards/min"][0][col_idx],
          np.min(rewards_matrix[:, col_idx]),
      )
      npt.assert_allclose(
          test_metrics["rewards/max"][0][col_idx],
          np.max(rewards_matrix[:, col_idx]),
      )

  def test_algo_config_param_passing(self):
    # [FUNCTION DOC]
    # Wraps prompt_len_reward in a spying Mock carrying its name and
    # signature, and verifies the manager calls it once with custom_param
    # 2.0 taken from the config, without unrelated parameters.
    # KNOWN ISSUE: the "another_param" absence check is vacuous because no
    # such field exists on the config.
    # Mock the reward function to spy on its call arguments
    mock_fn = mock.Mock(wraps=prompt_len_reward)
    mock_fn.__name__ = prompt_len_reward.__name__
    # Restore the signature for introspection
    mock_fn.__signature__ = inspect.signature(prompt_len_reward)

    manager = reward_manager.SequenceRewardManager(
        reward_fns=[mock_fn],
        algo_config=self.test_algo_config,
    )
    manager(
        self.prompts,
        self.completions,
    )

    mock_fn.assert_called_once()
    _, kwargs = mock_fn.call_args
    self.assertEqual(kwargs["custom_param"], 2.0)
    self.assertNotIn(
        "another_param", kwargs
    )  # Not in prompt_len_reward signature

  def test_nan_handling(self):
    # [FUNCTION DOC]
    # Verifies NaN rewards are ignored in the sum (nansum), while the
    # per-fn metric for the NaN fn keeps the NaNs.
    manager = reward_manager.SequenceRewardManager(
        reward_fns=[len_reward, nan_reward],
        algo_config=self.test_algo_config,
    )
    rewards_info = manager(
        self.prompts,
        self.completions,
    )
    # np.nansum should treat nan as 0 for summation
    expected_rewards = np.array([float(len(c)) for c in self.completions])
    np.testing.assert_array_almost_equal(
        rewards_info["rewards"], expected_rewards
    )
    # Check logged metrics for NaN
    test_metrics = rewards_info["log_metrics"]
    self.assertTrue(np.isnan(test_metrics["rewards/nan_reward"][0]).all())
    np.testing.assert_allclose(
        test_metrics["rewards/sum"][0],
        expected_rewards,
        err_msg="rewards/sum mismatch",
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="reward_fn_returns_none",
          reward_fns=[lambda prompts, completions, **kw: None],
          expected_regex="Failed to obtain result.*Result is None",
          error_type=RuntimeError,
      ),
      dict(
          testcase_name="reward_fn_bad_length",
          reward_fns=[
              lambda prompts, completions, **kw: [1.0] * (len(prompts) + 1)
          ],
          expected_regex="Length mismatch",
          error_type=RuntimeError,
      ),
  )
  def test_errors(
      self, expected_regex, error_type, kwargs=None, reward_fns=None
  ):
    # [FUNCTION DOC]
    # Verifies calling the manager raises RuntimeError "Failed to obtain
    # result ... Result is None" when a fn returns None, and "Length
    # mismatch" when a fn returns one value too many. Defaults reward_fns
    # to len_reward and names unnamed fns test_fn_<i>.
    # KNOWN ISSUE: lambdas always have a __name__ ("<lambda>"), so the
    # naming fallback never triggers.
    if reward_fns is None:
      reward_fns = [len_reward]
    for i, fn in enumerate(reward_fns):
      if not hasattr(fn, "__name__"):
        fn.__name__ = f"test_fn_{i}"

    manager = reward_manager.SequenceRewardManager(
        reward_fns=reward_fns,
        algo_config=self.test_algo_config,
    )
    with self.assertRaisesRegex(error_type, expected_regex):
      manager(
          self.prompts,
          self.completions,
          **(kwargs or {}),
      )

  def test_no_reward_fns_raises_error(self):
    # [FUNCTION DOC]
    # Verifies an empty reward_fns list raises ValueError at construction.
    with self.assertRaisesRegex(ValueError, "reward_fns cannot be empty"):
      reward_manager.SequenceRewardManager(
          reward_fns=[],
          algo_config=self.test_algo_config,
      )


class AgenticSequenceRewardManagerTest(parameterized.TestCase):
  # [CLASS DOC]
  # Tests that trajectory reward metrics and reward-fn metrics coexist in
  # AgenticSequenceRewardManager output.

  def setUp(self):
    # [FUNCTION DOC]
    # HELPER: creates the test config, two prompts and two completions.
    super().setUp()
    self.test_algo_config = TestAlgoConfig()
    self.prompts = ["p1", "p22"]
    self.completions = ["c1_long", "c2"]

  def test_log_metrics_non_interference_with_reward_fns(self):
    # [FUNCTION DOC]
    # Verifies both trajectory_rewards/sum and /mean and rewards/sum and
    # rewards/len_reward metrics are present when reward fns are given.
    manager = reward_manager.AgenticSequenceRewardManager(
        reward_fns=[len_reward],
        algo_config=self.test_algo_config,
    )
    traj_rewards = [10.0, 20.0]
    rewards_info = manager(
        self.prompts, self.completions, trajectory_rewards=traj_rewards
    )
    log_metrics = rewards_info["log_metrics"]
    # Verify trajectory metrics exist and are correctly prefixed
    self.assertIn("trajectory_rewards/sum", log_metrics)
    self.assertIn("trajectory_rewards/mean", log_metrics)
    # Verify general reward metrics exist and preserve their own prefix
    self.assertIn("rewards/sum", log_metrics)
    self.assertIn("rewards/len_reward", log_metrics)

  def test_log_metrics_non_interference_no_reward_fns(self):
    # [FUNCTION DOC]
    # Verifies with reward_fns None only trajectory_rewards metrics are
    # logged and no rewards/sum appears.
    manager = reward_manager.AgenticSequenceRewardManager(
        reward_fns=None,
        algo_config=self.test_algo_config,
    )
    traj_rewards = [5.0, 5.0]
    rewards_info = manager(
        self.prompts, self.completions, trajectory_rewards=traj_rewards
    )
    log_metrics = rewards_info["log_metrics"]
    # With no reward_fns, only trajectory log metrics should be populated
    self.assertIn("trajectory_rewards/sum", log_metrics)
    self.assertIn("trajectory_rewards/mean", log_metrics)
    self.assertNotIn("rewards/sum", log_metrics)


if __name__ == "__main__":
  absltest.main()
