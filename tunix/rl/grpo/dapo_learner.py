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
# DAPO (Decoupled clip and Dynamic sAmpling Policy Optimization,
# arXiv 2503.14476) configuration, overlong reward shaping and learner.
# DAPOConfig extends GRPOConfig (asymmetric clipping via epsilon_high, no KL,
# token-mean loss, overlong-buffer settings). reward_shaping is an extra
# reward function penalizing completions that run into the overlong buffer.
# DAPOLearner is a GRPO learner that appends reward_shaping to the reward
# functions when the overlong buffer is enabled. Re-exports TrainingInputT,
# RewardFn and MetricFn.
# KNOWN ISSUE: the module docstring says "Helper functions for GRPO Trainer".
"""Helper functions for GRPO Trainer."""

import dataclasses
from typing import Any, Dict, List, Optional, Sequence
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import rl_learner
from tunix.rl.grpo import grpo_learner as grpo_learner_lib

TrainingInputT = rl_learner.TrainingInputT
RewardFn = rl_learner.RewardFn
MetricFn = rl_learner.MetricFn


@dataclasses.dataclass(kw_only=True)
class DAPOConfig(grpo_learner_lib.GRPOConfig):
  # [CLASS DOC]
  # Keyword-only dataclass. Defaults: algo_variant "dapo", advantage_estimator
  # "grpo", policy_loss_fn "grpo", loss_agg_mode "token-mean", reward_manager
  # "sequence-level", num_generations 2, num_iterations 1, beta None (no KL),
  # epsilon 0.2 (lower clip), epsilon_high 0.28 (upper clip), dynamic_sampling
  # True (not implemented), overlong_buffer dict with enable True,
  # overlong_buffer_length 4096, overlong_buffer_penalty 1.0,
  # max_response_length 20480. The overlong_buffer dict is forwarded by the
  # reward manager as a keyword argument to reward functions whose signature
  # names it.
  """Configuration for DAPO.

  Attributes:
   algo_variant: The algorithm variant to use.
   advantage_estimator: The advantage estimator to use.
   policy_loss_fn: The policy loss function to use.
   loss_agg_mode: The aggregation mode for the loss function.
   loss_algo: The loss algorithm to use. To be deprecated.
   num_generations: The number of times the policy generates multiple responses
     for a given prompt within a single training step. This corresponds to 'G'
     in Algorithm 1 in the paper. A higher value means more samples are used to
     compute relative advantages.
   num_iterations: The number of iterations per batch (𝜇 in GRPO algo 1).
   beta: The coefficient for the KL divergence penalty (𝛽) in the GRPO loss
     function. This term prevents policy updates from deviating too far from the
     reference model. A value of 0.0 means no KL penalty is applied. Always None
     for DAPO.
   epsilon: Epsilon value for clipping (𝜀 in GRPO loss in paper). Similar to
     PPO, it ensures stable updates.
   epsilon_high: Epsilon value for upper bound clipping.
   dynamic_sampling: Whether to use dynamic sampling.
   overlong_buffer: The overlong buffer to use for overlong reward shaping.
   References: - DAPO:
     https://arxiv.org/pdf/2503.14476
  """

  algo_variant: str = "dapo"
  advantage_estimator: str = "grpo"
  policy_loss_fn: str = "grpo"
  loss_agg_mode: str = "token-mean"
  reward_manager: str = "sequence-level"
  num_generations: int = 2
  num_iterations: int = 1
  beta: None = None  # No KL term.
  epsilon: float = 0.2
  epsilon_high: float = 0.28  # Clip higher
  dynamic_sampling: bool = True  # TODO(sizhi): Add dynamic sampling.
  overlong_buffer: Optional[Dict[str, Any]] = dataclasses.field(
      default_factory=lambda: {
          "enable": True,
          "overlong_buffer_length": 4096,  # Threshold before penalties apply.
          "overlong_buffer_penalty": 1.0,
          "max_response_length": 20480,  # Hard maximum generation length.
      }
  )

  def __post_init__(self):
    # [FUNCTION DOC]
    # Validates: beta must be None, epsilon_high must be at least epsilon; if the
    # overlong buffer is present and enabled, all of overlong_buffer_length,
    # overlong_buffer_penalty and max_response_length must be present and not
    # None, penalty must be non-negative, buffer length and max response length
    # positive. Raises ValueError on the first violation. Returns None.
    # KNOWN ISSUE: does not call GRPOConfig/AlgorithmConfig __post_init__, so
    # num_generations and loss_algo checks and config logging are skipped. Also
    # GRPOLearner tests beta against 0.0, and None differs from 0.0, so the
    # reference model log-probs are still computed for DAPO (and a KL term may
    # be attempted) despite "no KL".
    if self.beta is not None:
      raise ValueError(
          "DAPO does not support KL penalty, so beta must be None."
      )
    if self.epsilon_high < self.epsilon:
      raise ValueError("epsilon_high must be greater than or equal to epsilon.")

    if self.overlong_buffer is not None and self.overlong_buffer.get("enable"):
      buffer = self.overlong_buffer
      required = [
          "overlong_buffer_length",
          "overlong_buffer_penalty",
          "max_response_length",
      ]

      missing = [k for k in required if buffer.get(k) is None]
      if missing:
        raise ValueError(f"overlong_buffer is enabled but missing: {missing}")

      if buffer["overlong_buffer_penalty"] < 0:
        raise ValueError("overlong_buffer_penalty must be non-negative.")

      if buffer["overlong_buffer_length"] <= 0:
        raise ValueError("overlong_buffer_length must be positive.")

      if buffer["max_response_length"] <= 0:
        raise ValueError("max_response_length must be positive.")


def reward_shaping(
    prompts: List[str],
    completions: List[str],
    mode: rl_cluster_lib.Mode,
    overlong_buffer: Dict[str, Any] | None = None,
    **kwargs,
) -> List[float]:
  # [FUNCTION DOC]
  # Reward function for overlong shaping. Inputs: prompts, completions (strings),
  # mode, overlong_buffer dict, ignored kwargs. Raises ValueError if
  # overlong_buffer is None (KeyError if keys missing). expected length =
  # max_response_length minus overlong_buffer_length. For each completion,
  # exceed = its length minus expected length; reward is the minimum of zero and
  # negative exceed divided by buffer length times penalty. Returns the list of
  # floats (0 for short completions, increasingly negative inside the buffer).
  # KNOWN ISSUE: length is measured in characters of the completion string, not
  # tokens, while the thresholds describe generation (token) lengths; the
  # penalty is not capped beyond the buffer.
  """Reward shaping function for DAPO."""
  del prompts, mode, kwargs
  if overlong_buffer is None:
    raise ValueError("reward_shaping is called but with empty overlong_buffer.")

  overlong_buffer_length = overlong_buffer["overlong_buffer_length"]
  overlong_buffer_penalty = overlong_buffer["overlong_buffer_penalty"]
  max_response_length = overlong_buffer["max_response_length"]

  expected_response_length = max_response_length - overlong_buffer_length
  scores = []
  for completion in completions:
    output_length = len(completion)
    exceed_length = output_length - expected_response_length
    overlong_reward = min(
        -exceed_length / overlong_buffer_length * overlong_buffer_penalty, 0
    )
    scores.append(overlong_reward)
  return scores


class DAPOLearner(grpo_learner_lib.GrpoLearner[DAPOConfig]):
  """DAPO learner."""
  # [CLASS DOC]
  # GRPO learner specialized with DAPOConfig; only constructor behavior differs.

  def __init__(
      self,
      rl_cluster: rl_cluster_lib.RLCluster,
      algo_config: DAPOConfig,
      reward_fns: RewardFn | List[RewardFn],
      metric_fns: Sequence[MetricFn] | None = None,
      data_shuffle_seed: int | None = None,
  ):
    # [FUNCTION DOC]
    # Inputs as GRPOLearner. Wraps a single reward function into a list
    # (non-sequence inputs). If overlong_buffer is set and its "enable" entry is
    # truthy, appends reward_shaping to the reward list. Then calls GRPOLearner
    # init with the (possibly extended) list.
    # KNOWN ISSUE: when the caller passes a list, it is mutated in place by the
    # append; passing a tuple raises AttributeError on append; a missing
    # "enable" key raises KeyError.
    """Initializes the `DAPOLearner`."""
    reward_fns = (
        [reward_fns] if not isinstance(reward_fns, Sequence) else reward_fns
    )
    if algo_config.overlong_buffer and algo_config.overlong_buffer["enable"]:
      reward_fns.append(reward_shaping)
    super().__init__(
        rl_cluster=rl_cluster,
        algo_config=algo_config,
        reward_fns=reward_fns,
        metric_fns=metric_fns,
        data_shuffle_seed=data_shuffle_seed,
    )
