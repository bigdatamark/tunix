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
# Defines ToolEnvironment, a multi-step BaseTaskEnv in which the agent's actions
# are lists of OpenAI-style tool-call dicts ({"id", "type", "function": {"name",
# "arguments" as JSON string}}). Non-finish calls are executed through a
# ToolManager and their string outputs returned as the next observation; a string
# action or any call named "finish" ends the episode and triggers reward
# computation. Imports reward (unused) and uses absl logging.
"""RL Environment for tool-based agent interactions.

This module defines the `ToolEnvironment`, a reinforcement learning environment
designed to interface with agents that interact with a set of tools. It handles
tool execution, reward calculation, and episode management within an RL
framework.
"""

import json
from typing import Any, Dict, List
import uuid

from absl import logging
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.environments import base_environment
from tunix.rl.agentic.rewards import reward
from tunix.rl.agentic.tools import base_tool
from tunix.rl.agentic.tools import tool_manager


class ToolEnvironment(base_environment.BaseTaskEnv):
  # [CLASS DOC]
  # Tool-execution environment. Attributes: those of BaseTaskEnv (task, reward_fn,
  # max_steps default 10, step_count, extra_kwargs) plus tool_manager (ToolManager
  # built from tool_map). Observations: the task dict on reset, {"tool_outputs":
  # {call_id: output_string}} on intermediate steps, {} on the final step.
  """Reinforcement learning environment for tool-based agent interactions.

  This environment enables agents to execute external tools and receive
  structured feedback compatible with standard RL training pipelines.
  It manages tool execution, reward computation, episode termination,
  and maintains compatibility with the BaseEnv/BaseTaskEnv interface.

  The environment supports both single and multi-step tool interactions,
  automatic episode termination through 'finish' tool calls or reaching
  `max_steps`, and flexible reward function integration for various
  evaluation strategies.
  """

  def __init__(
      self,
      task: Dict[str, Any] | None = None,
      *,
      tool_map: Dict[str, type[base_tool.BaseTool]],
      reward_fn=None,
      max_steps: int = 10,
      **kwargs,
  ):
    # [FUNCTION DOC]
    # Initialize. Inputs: task (dict or None); keyword-only tool_map (required,
    # name-to-BaseTool-class), reward_fn (callable taking task= and action= keyword
    # args, or None), max_steps (default 10), extra kwargs forwarded to BaseTaskEnv.
    # Behavior: if reward_fn is None, logs a warning once per process; calls the base
    # constructor; builds the ToolManager.
    # Exceptions: TypeError from ToolManager for abstract tool classes; AttributeError
    # if tool_map is None.
    # KNOWN ISSUE: docstring says a missing reward_fn defaults to dummy_reward, but
    # the code just yields a reward of 0.0 at episode end.
    """Initialize the tool environment with task configuration and constraints.

    Args:
      task (Dict[str, Any] | None): Task specification containing problem
        description, ground truth, or other task-specific parameters. If None,
        defaults to empty dict.
      tool_map (Dict[str, type[BaseTool]]): Mapping of tool names to their
        implementation classes for tool discovery and execution.
      reward_fn: Reward function that takes (task, action) and returns float. If
        None, defaults to `dummy_reward` with a warning.
      max_steps (int): Maximum number of interaction steps before forced
        termination. Prevents infinite loops and controls episode length.
      **kwargs: Additional arguments reserved for future extensions.
    """
    if reward_fn is None:
      logging.log_first_n(
          logging.WARNING,
          "No reward_fn provided, skipping trajectory reward computation.",
          1,
      )

    # Let BaseTaskEnv handle task, reward_fn, step_count, and max_steps.
    super().__init__(
        task=task,
        reward_fn=reward_fn,
        max_steps=max_steps,
        **kwargs,
    )

    # Tool execution system for managing available tools and their invocation.
    self.tool_manager = tool_manager.ToolManager(tool_map)

  def _initial_observation(self) -> Dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: Return the task dict itself (not a copy) as the initial observation.
    """Reset the environment to initial state for a new episode.

    Clears step counter (in BaseTaskEnv) and returns the task specification as
    the initial observation. The task dictionary serves as the starting context
    for the agent to begin tool-based problem solving.

    Returns:
      Dict[str, Any]: The task specification as the initial observation.
    """
    return self.task

  def _step_impl(self, action: Any) -> base_environment.EnvStepResult:
    # [FUNCTION DOC]
    # HELPER: One logical step.
    # Inputs: action, which may be an Action (unwrapped to its payload), None
    # (treated as empty list), a single dict (wrapped in a list), a string, or a list
    # of call dicts.
    # Behavior: done is True if the action is a string, or if it is a list containing
    # a call whose function name is "finish". When done: extracts the answer with
    # _extract_llm_answer, computes reward_fn(task=task, action=answer) if a reward_fn
    # exists else 0.0, and returns EnvStepResult(observation {}, that reward, done
    # True, info {"response": action}). Otherwise executes all calls via
    # _execute_tool_calls and returns EnvStepResult(observation {"tool_outputs":
    # outputs}, reward 0.0, done False, info {"response": action}).
    # Note the reward value is whatever reward_fn returns (may be a RewardOutput, not
    # a float). An empty list action is executed as zero calls (not done).
    # Exceptions: errors from reward_fn, JSON decoding or malformed call dicts.
    """Execute one logical step of tool interaction based on agent's action.

    Processes the agent's action which can be a string response (indicating
    completion) or a list of tool calls to execute. Manages episode termination
    logic based on action type and 'finish' tool calls. The `max_steps` limit
    is enforced by the outer `BaseTaskEnv.step()`.

    Args:
      action (Any): Agent's action - can be string (final answer), list of tool
        call dicts, or None (treated as empty list).

    Returns:
      EnvStepResult: The outcome of this logical step.
    """
    if isinstance(action, agent_types.Action):
      action = action.action

    # Handle None action as empty action list.
    if action is None:
      action = []

    # Normalize single tool call dict to list format.
    if isinstance(action, dict):
      action = [action]

    is_string = isinstance(action, str)
    done = is_string

    # Check for explicit finish tool call when action is a list.
    if isinstance(action, list):
      if any(
          call.get("function", {}).get("name") == "finish" for call in action
      ):
        done = True

    # Handle episode termination: compute final reward.
    if done:
      llm_answer = self._extract_llm_answer(action)
      if self.reward_fn is not None:
        reward_val = self.reward_fn(task=self.task, action=llm_answer)
      else:
        reward_val = 0.0
      return base_environment.EnvStepResult(
          observation={},
          reward=reward_val,
          done=True,
          info={"response": action},
      )

    # Handle continuing episode: execute tools and return intermediate results.
    tool_outputs = self._execute_tool_calls(action)
    obs = {"tool_outputs": tool_outputs}
    return base_environment.EnvStepResult(
        observation=obs,
        reward=0.0,
        done=False,
        info={"response": action},
    )

  @staticmethod
  def _extract_llm_answer(action: Any) -> str:
    # [FUNCTION DOC]
    # HELPER: Static. Get the final answer text from an action. A string is returned
    # as is. For a list, the first call named "finish" returns its function
    # "arguments" mapping's "response" value (default ""). Otherwise returns
    # str(action).
    # KNOWN ISSUE: if the finish call's arguments are a JSON string (as produced by
    # ToolAgent for parsed tool calls), calling get on a string raises
    # AttributeError; only the dict form produced by ToolAgent's fallback works.
    """Extract the final answer string from various action formats.

    Handles multiple action types including direct string responses,
    finish tool calls with response arguments, and fallback string
    conversion for other action types.

    Args:
      action (Any): Agent's action in various possible formats.

    Returns:
      str: Extracted answer string for reward computation.
    """
    if isinstance(action, str):
      return action
    if isinstance(action, list):
      # Look for finish tool call with response argument.
      for call in action:
        if call.get("function", {}).get("name") == "finish":
          args = call["function"].get("arguments", {})
          return args.get("response", "")
    # Fallback: convert action to string representation.
    return str(action)

  def _execute_tool_calls(
      self, tool_calls: List[Dict[str, Any]]
  ) -> Dict[str, str]:
    # [FUNCTION DOC]
    # HELPER: Execute non-finish tool calls.
    # Inputs: list of call dicts. Behavior: for each, reads function name, decodes
    # function arguments from a JSON string, uses the dict's "id" or a new uuid4,
    # builds a ToolCall and attaches the id as an extra attribute "id"; then runs all
    # via tool_manager.execute_calls with parallel True.
    # Outputs: dict of call id to output string.
    # Exceptions: KeyError for missing keys; JSONDecodeError / TypeError if
    # arguments are not a valid JSON string.
    """Execute a list of tool calls and return their outputs.

    Converts raw tool call dictionaries to ToolCall objects, assigns
    unique IDs if missing, and delegates execution to the ToolManager
    with parallel execution enabled for performance.

    Args:
      tool_calls (List[Dict[str, Any]]): List of tool call specifications
        containing function name, arguments, and optional call ID.

    Returns:
      Dict[str, str]: Mapping from tool call IDs to their output strings.
    """
    call_objs = []
    for tc in tool_calls:
      name = tc["function"]["name"]
      args = json.loads(tc["function"]["arguments"])
      call_id = tc.get("id") or str(uuid.uuid4())

      # Create ToolCall object and attach ID for result tracking.
      call_obj = base_tool.ToolCall(name=name, arguments=args)
      setattr(call_obj, "id", call_id)
      call_objs.append(call_obj)

    return self.tool_manager.execute_calls(call_objs, parallel=True)

  @classmethod
  def from_dict(cls, env_args: Dict[str, Any]) -> "ToolEnvironment":
    # [FUNCTION DOC]
    # Classmethod factory. Pops "tool_map" (default None), "reward_fn" (default None)
    # and "max_steps" (default 10) from env_args; the remaining dict is used as the
    # task. Returns cls(task=..., tool_map=..., reward_fn=..., max_steps=...).
    # Side effects: mutates the caller's env_args dict (keys popped, and the same
    # dict object becomes the task).
    """Create ToolEnvironment instance from configuration dictionary.

    Factory method that extracts environment-specific parameters from
    the configuration dict and uses remaining entries as the task
    specification. This preserves the original behavior:

      * `tool_map` (required) specifies the available tools.
      * `reward_fn` (optional) overrides the default reward function.
      * `max_steps` (optional, defaults to 10) controls episode length.
      * Any other entries become part of the `task` specification.

    Args:
      env_args (Dict[str, Any]): Configuration dictionary.

    Returns:
      ToolEnvironment: Configured environment instance ready for use.
    """
    # Extract environment configuration parameters.
    tool_map = env_args.pop("tool_map", None)
    reward_fn = env_args.pop("reward_fn", None)
    max_steps = env_args.pop("max_steps", 10)

    # Remaining entries form the task specification.
    task = env_args

    return cls(
        task=task,
        tool_map=tool_map,
        reward_fn=reward_fn,
        max_steps=max_steps,
    )
