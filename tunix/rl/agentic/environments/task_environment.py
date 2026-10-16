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
# Defines TaskEnvironment, a single-turn BaseTaskEnv: reset returns the task (one
# dataset example) and the first action always ends the episode with a reward
# computed by the optional reward_fn. Imports reward (unused) and absl logging.
"""RL Environment for single-turn task-based agent interactions."""

from typing import Any, Dict

from absl import logging
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.environments import base_environment
from tunix.rl.agentic.rewards import reward


class TaskEnvironment(base_environment.BaseTaskEnv):
  # [CLASS DOC]
  # Single-step environment with max_steps fixed at 1. Attributes inherited from
  # BaseTaskEnv; task holds the example dict.
  """Reinforcement learning environment for single-turn agent interactions.

  This environment is designed for tasks where the agent receives an
  initial observation (the task) and provides a single response, after
  which the episode terminates. It does not involve multi-step interactions
  or tool use.

  It is implemented as a `BaseTaskEnv` subclass with `max_steps=1`, and
  a `_step_impl` that always finishes the episode on the first action.
  """

  def __init__(
      self,
      single_example: Dict[str, Any],
      *,
      reward_fn=None,
      **kwargs,
  ):
    # [FUNCTION DOC]
    # Initialize. Inputs: single_example (dict used as the task); keyword-only
    # reward_fn (callable with task= and action= keywords, or None); extra kwargs
    # (e.g. group_id) forwarded to BaseTaskEnv and stored in extra_kwargs.
    # Behavior: logs a warning once per process when reward_fn is None; calls the
    # base constructor with max_steps=1.
    # Exceptions: TypeError if kwargs contains max_steps (duplicate keyword).
    """Initialize the task environment.

    Args:
      single_example: A single prompt.
      reward_fn: RewardFunction to be used in environment. Optional.
      **kwargs: Extra arguments ignored by this environment but accepted for
        compatibility with a common environment config interface.
    """
    if reward_fn is None:
      logging.log_first_n(
          logging.WARNING,
          "No reward_fn provided.",
          1,
      )

    super().__init__(
        task=single_example, reward_fn=reward_fn, max_steps=1, **kwargs
    )

  def _initial_observation(self) -> Dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: Return the task dict itself as the initial observation.
    """Reset the environment and return the task as the initial observation."""
    return self.task

  def _step_impl(self, action: Any) -> base_environment.EnvStepResult:
    # [FUNCTION DOC]
    # HELPER: Unwrap an Action to its payload, compute reward_fn(task=task,
    # action=payload) if a reward_fn is set (else 0.0), and return
    # EnvStepResult(observation {}, reward value, done True, info {"response":
    # payload}). The reward is whatever reward_fn returns (may be a RewardOutput).
    # KNOWN ISSUE: the docstring says info includes reward metadata, but only the
    # response is included.
    """Process the agent's action, calculate reward, and terminate.

    In a single-turn environment, any action terminates the episode. We assume
    `action` is the agent's final response string (or other suitable type).

    Args:
      action: The action taken by the agent.

    Returns:
      An `EnvStepResult` containing an empty observation, the calculated reward,
      done=True, and info including the agent's response and reward metadata.
    """
    if isinstance(action, agent_types.Action):
      action = action.action
    if self.reward_fn is not None:
      reward_val = self.reward_fn(task=self.task, action=action)
    else:
      reward_val = 0.0
    return base_environment.EnvStepResult(
        observation={},
        reward=reward_val,
        done=True,
        info={"response": action},
    )

  @classmethod
  def from_dict(cls, env_args: Dict[str, Any]) -> "TaskEnvironment":
    # [FUNCTION DOC]
    # Classmethod factory. Pops "reward_fn" and "group_id" (both default None) from
    # env_args; the remaining dict becomes the task. Returns cls(single_example=...,
    # reward_fn=..., group_id=...), so group_id ends up in extra_kwargs.
    # Side effects: mutates the caller's env_args dict.
    """Create TaskEnvironment instance from configuration dictionary.

    This preserves the original behavior:

      * `reward_fn` is popped from `env_args` (if present).
      * The remaining entries are treated as the `task` specification.

    This allows configs like:

      env_args = {
        "question": "...",
        "answer": "...",
        "reward_fn": my_reward_fn,
      }

    without requiring a nested "task" key.

    Args:
      env_args: A dictionary containing environment configuration.
    """
    reward_fn = env_args.pop("reward_fn", None)
    group_id = env_args.pop("group_id", None)
    return cls(single_example=env_args, reward_fn=reward_fn, group_id=group_id)
