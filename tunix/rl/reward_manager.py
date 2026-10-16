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
# Reward managers: objects that run one or more reward functions over a batch
# of (prompt, completion) pairs, combine their outputs into one scalar reward
# per completion, and prepare logging metrics. Defines the RewardFn alias, an
# abstract base, the "sequence-level" manager and the "agentic-sequence-level"
# manager; both are registered in function_registry at import time (side
# effect). Setting environment variable TUNIX_DEBUG_REWARDS logs one example.
# Metric dicts map metric name to a pair of value and aggregation function
# (None for non-numeric values such as prompt text).
"""Reward output for RL."""

import abc
from dataclasses import asdict
import inspect
import os
from typing import Any, Callable, Dict, List, Sequence
from absl import logging
import numpy as np
from tunix.rl import algorithm_config as algo_config_lib
from tunix.rl import function_registry


RewardFn = Callable[..., Any]


def _calculate_scalar_reward_log_metrics(
    rewards: np.ndarray,
    prefix: str = "rewards",
    axis: int = 1,
) -> Dict[str, Any]:
  # [FUNCTION DOC]
  # HELPER: Inputs: reward array, prefix (default "rewards"), axis (default 1).
  # Returns a dict with keys prefix plus "/sum", "/mean", "/min", "/max", each
  # mapping to a pair of the NaN-aware sum / NaN-aware mean / min / max along
  # axis and the aggregation (mean for sum and mean, min for min, max for max).
  # Min and max are not NaN-aware.
  """Helper to calculate sum, mean, min, and max log metrics for rewards."""
  return {
      f"{prefix}/sum": (np.nansum(rewards, axis=axis), np.mean),
      f"{prefix}/mean": (np.nanmean(rewards, axis=axis), np.mean),
      f"{prefix}/min": (np.min(rewards, axis=axis), np.min),
      f"{prefix}/max": (np.max(rewards, axis=axis), np.max),
  }


class AbstractRewardManager(abc.ABC):
  # [CLASS DOC]
  # Abstract base. State: reward_fns (list of callables) and algo_config.
  # Subclasses implement __call__ returning a dict with "rewards" (one float per
  # completion) and "log_metrics".
  """Abstract base class for managing and orchestrating multiple reward function outputs."""

  def __init__(
      self,
      reward_fns: RewardFn | List[RewardFn],
      algo_config: algo_config_lib.AlgorithmConfig,
  ):
    # [FUNCTION DOC]
    # Wraps a non-sequence reward_fns into a one-element list (sequences are kept
    # as is). Raises ValueError if the resulting list is empty. Stores
    # algo_config.
    """Initializes the manager with a list of callable reward function objects.

    Args:
        reward_fns: A list of reward functions or models.
        algo_config: The algorithm config to use for reward function
          configuration.
    """
    self.reward_fns = (
        [reward_fns] if not isinstance(reward_fns, Sequence) else reward_fns
    )

    if not self.reward_fns:
      raise ValueError(
          "reward_fns cannot be empty. You must provide at least one reward"
          " function."
      )
    self.algo_config = algo_config

  @abc.abstractmethod
  def __call__(
      self,
      prompts: List[str],
      completions: List[str],
      **kwargs,
  ) -> Dict[str, Any]:
    # [FUNCTION DOC]
    # Abstract. Inputs: prompts, completions, extra keyword args for reward
    # functions. Must return a rewards info dict. Base body does nothing.
    """Computes the rewards for completions using the provided reward functions.

    Args:
        prompts: A list of input prompts.
        completions: A list of generated text completions.
        **kwargs: Additional keyword arguments passed to the reward functions.

    Returns:
        A dictionary of rewards information, including the final rewards for
        advantage computation and intermediate rewards for logging.
    """
    pass


@function_registry.register_reward_manager("sequence-level")
class SequenceRewardManager(AbstractRewardManager):
  # [CLASS DOC]
  # Registered as "sequence-level". Computes one reward per completion by
  # summing (NaN-aware) the outputs of all reward functions.
  """Reward manager for sequence-level rewards only."""

  def __init__(
      self,
      reward_fns: RewardFn | List[RewardFn],
      algo_config: algo_config_lib.AlgorithmConfig,
      **kwargs,
  ):
    # [FUNCTION DOC]
    # Accepts and ignores extra kwargs; delegates to the base initializer.
    """Initializes the manager with a list of callable reward function objects."""
    super().__init__(reward_fns, algo_config)

  def __call__(
      self,
      prompts: List[str],
      completions: List[str],
      **kwargs,
  ) -> Dict[str, Any]:
    # [FUNCTION DOC]
    # Returns _compute_rewards(prompts, completions, kwargs).
    """Computes the rewards for completions using the provided reward function, and return the sequence-level rewards information for advantage computationand logging."""
    return self._compute_rewards(prompts, completions, **kwargs)

  def _compute_rewards(
      self,
      prompts: List[str],
      completions: List[str],
      **kwargs,
  ) -> Dict[str, Any]:
    # [FUNCTION DOC]
    # Inputs: prompts, completions, extra kwargs (e.g. dataset columns, mode).
    # Converts algo_config to a dict. Builds a zeros matrix of prompt count by
    # number of reward functions. For each reward function: inspects its
    # signature; for each parameter other than self, prompts, completions and
    # kwargs whose name is an algo_config field and is not already in the
    # caller's kwargs, injects the config value. Calls the function with
    # prompts, completions and the merged kwargs. Raises RuntimeError if it
    # returns None or a list of the wrong length. Stores the result as a column.
    # Then builds log metrics, computes per-row NaN-aware sums as final rewards,
    # and if TUNIX_DEBUG_REWARDS is set logs the first example of each metric.
    # Returns a dict with "rewards" (1-D array) and "log_metrics".
    # KNOWN ISSUE: the comment says parameter named kwargs is skipped, but a
    # VAR_KEYWORD parameter is named by the function author, so this only
    # matches the literal name; non-list results of wrong length are only caught
    # by NumPy broadcasting errors.
    """Computes the rewards for completions using the provided reward functions."""

    algo_config_params = asdict(self.algo_config)
    base_kwargs = kwargs.copy()

    num_prompts = len(prompts)
    num_reward_fns = len(self.reward_fns)
    rewards = np.zeros((num_prompts, num_reward_fns))

    # Compute all rewards for each prompt-completion pair.
    for i, reward_fn in enumerate(self.reward_fns):
      # Update the kwargs with the algo_config parameters.
      signature = inspect.signature(reward_fn)
      reward_fn_config_params = {}
      # Iterate over the function's expected parameters
      for name, _ in signature.parameters.items():
        # Skip standard parameters that are always passed (self, prompts, completions, kwargs)
        if name in ["self", "prompts", "completions", "kwargs"]:
          continue

        # Check if the parameter name matches a key in the algo_config dict. If
        # so, set the value to the algo_config parameter value, otherwise respect the value in the base_kwargs.
        if name in algo_config_params and name not in base_kwargs:
          reward_fn_config_params[name] = algo_config_params[name]

      call_kwargs = base_kwargs.copy()
      call_kwargs.update(reward_fn_config_params)

      r = reward_fn(prompts=prompts, completions=completions, **call_kwargs)

      if r is None:
        raise RuntimeError(
            f"Failed to obtain result from {reward_fn.__name__}. Result is"
            " None."
        )
      if isinstance(r, list) and len(r) != len(prompts):
        raise RuntimeError(
            f"Length mismatch after {reward_fn.__name__}: "
            f"len(r)={len(r)}, len(prompts)={num_prompts}. "
            f"Content of r: {r}"
        )

      rewards[:, i] = np.array(r)

    # Prepare metrics for logging.
    log_metrics = self._prepare_log_metrics(
        prompts,
        completions,
        rewards,
    )
    sum_rewards = np.nansum(rewards, axis=1)
    rewards_info = {
        "rewards": sum_rewards,
        "log_metrics": log_metrics,
    }

    def _log_one_example(log_metrics: Dict[str, Any]):
      # [FUNCTION DOC]
      # HELPER: Logs a header, then for each metric the first element of its first
      # value as a string snippet, then a footer, via absl info logging.
      logging.info("======= example rewards =======")

      # add a snippet of the prompt, completion, and reward
      def snippet(s: str, k: int = 50):
        # [FUNCTION DOC]
        # HELPER: Returns s unchanged if at most 2k chars (k default 50), else the
        # first k chars, an ellipsis and the last k chars.
        if len(s) <= 2 * k:
          return s
        return s[:k] + "..." + s[-k:]

      for k, v in log_metrics.items():
        logging.info("%s:\t%s", k, snippet(str(v[0][0])))
      logging.info("=======================")

    if os.getenv("TUNIX_DEBUG_REWARDS"):
      _log_one_example(log_metrics)

    return rewards_info

  def _prepare_log_metrics(
      self,
      prompts: List[str],
      completions: List[str],
      rewards: np.ndarray,  # (num_prompts, num_reward_fns)
  ) -> Dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: Builds the logging dict: "prompts" and "completions" (raw lists,
    # aggregation None), the rewards sum/mean/min/max across reward functions
    # per example (axis 1), and for each reward function "rewards/" plus its
    # __name__ mapping to its column with mean aggregation. Returns the dict.
    # KNOWN ISSUE: callables without a __name__ attribute (e.g. functools.partial
    # objects) raise AttributeError; the comment mentions an rl_cluster
    # attribute that does not exist.
    """Logs individual and summed rewards, along with prompts/completions, for each trajectory."""
    # Assuming self.reward_fns and self.rl_cluster are accessible instance attributes
    metrics_to_log = {}

    # Log prompts and completions.
    metrics_to_log["prompts"] = (prompts, None)
    metrics_to_log["completions"] = (completions, None)

    # Log the sum/mean rewards for each prompt-completion pair.
    metrics_to_log.update(
        _calculate_scalar_reward_log_metrics(rewards, prefix="rewards")
    )

    # Log individual rewards for this trajectory
    for i, reward_fn in enumerate(self.reward_fns):
      metric_name = f"rewards/{reward_fn.__name__}"
      metrics_to_log[metric_name] = (rewards[:, i], np.mean)

    return metrics_to_log


@function_registry.register_reward_manager("agentic-sequence-level")
class AgenticSequenceRewardManager(SequenceRewardManager):  # pytype: disable=base-class-error
  # [CLASS DOC]
  # Registered as "agentic-sequence-level". Final reward per trajectory is the
  # environment-provided trajectory reward plus, if any reward functions are
  # configured, their NaN-aware summed rewards.
  """Reward manager for agentic settings.

  Supports two reward sources:
  - Pluggable reward_fns evaluated post-rollout (e.g. deepscaler).
  - Trajectory rewards from the environment, passed via `trajectory_rewards`
    kwarg (e.g. deepswe).

  reward_fns is optional. When not provided, only trajectory_rewards are used
  and the fn computation step is skipped entirely.
  """

  def __init__(
      self,
      reward_fns: RewardFn | List[RewardFn] | None,
      algo_config: algo_config_lib.AlgorithmConfig,
      **kwargs,
  ):
    # [FUNCTION DOC]
    # If reward_fns is None, sets an empty reward list and stores algo_config
    # without validation; otherwise uses the parent initializer (which rejects
    # empty lists). Extra kwargs are ignored.
    if reward_fns is None:
      self.reward_fns = []
      self.algo_config = algo_config
    else:
      super().__init__(reward_fns, algo_config)  # pytype: disable=attribute-error

  def __call__(
      self,
      prompts: List[str],
      completions: List[str],
      **kwargs,
  ) -> Dict[str, Any]:
    # [FUNCTION DOC]
    # Inputs: prompts, completions, kwargs that must contain
    # "trajectory_rewards" (KeyError otherwise; it is removed from kwargs).
    # Converts trajectory rewards to an array, logs their sum/mean/min/max over
    # axis 0 under prefix "trajectory_rewards" (scalars). If reward functions
    # exist, computes their rewards with the remaining kwargs, adds them to the
    # trajectory rewards and merges their log metrics. Returns a dict with
    # "rewards" and "log_metrics".
    # KNOWN ISSUE: the addition is in place, so when trajectory_rewards is
    # already a float NumPy array the caller's array is modified; an integer
    # array would raise a casting error when adding float rewards.

    log_metrics = {}

    # Extract trajectory rewards from kwargs and log them. Even trajectory rewards will be all zero if not provided.
    trajectory_rewards = kwargs.pop("trajectory_rewards")
    trajectory_rewards_array = np.asarray(trajectory_rewards)
    # Log trajectory rewards separately
    log_metrics.update(
        _calculate_scalar_reward_log_metrics(
            trajectory_rewards_array, prefix="trajectory_rewards", axis=0
        )
    )
    final_rewards = trajectory_rewards_array

    if self.reward_fns:
      rewards_info = self._compute_rewards(prompts, completions, **kwargs)
      final_rewards += rewards_info["rewards"]
      log_metrics.update(rewards_info["log_metrics"])

    return {"rewards": final_rewards, "log_metrics": log_metrics}
