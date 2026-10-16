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
# Agent base classes. LLMBaseAgent is the abstract interface every agent exposes
# to the trajectory collection engine: read-only chat_completions (messages to
# send to the LLM) and trajectory, update_from_env / update_from_model (sync, with
# async wrappers that run the sync method in the default thread-pool executor),
# reset, and get_current_step. ConversationAgentBase implements the common
# chat-agent state: a message list seeded with a system message, a Trajectory,
# recording of environment feedback into the last Step, and default conversion of
# observations into user messages.
"""Base classes for Large Language Model powered agents.

This module defines:

* `LLMBaseAgent`: the minimal abstract base class that provides a standard
  interface for agents interacting with LLMs and environments.

* `ConversationAgentBase`: a higher-level base class for chat-style agents
  that maintain conversation history and trajectories. Most concrete agents
  (single-turn, tool-using, gaming, etc.) should subclass this instead of
  `LLMBaseAgent` directly.
"""

import abc
import asyncio
import copy
from typing import Any, Dict

from tunix.rl.agentic.agents import agent_types


class LLMBaseAgent(abc.ABC):
  # [CLASS DOC]
  # Abstract agent interface (abc.ABC). Abstract members: chat_completions and
  # trajectory properties, update_from_env, update_from_model, reset. Concrete
  # helpers: async wrappers update_from_env_async / update_from_model_async and
  # get_current_step.
  """Abstract base class for Large Language Model powered agents."""

  # ──────────────────────────────────────────────────────────────
  # State Access Properties
  # ──────────────────────────────────────────────────────────────

  @property
  @abc.abstractmethod
  def chat_completions(self) -> list[dict[str, str]]:
    # [FUNCTION DOC]
    # Abstract property: the list of chat message dicts forming the current LLM
    # context. Base raises NotImplementedError.
    """Get the current conversation context for the LLM."""
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def trajectory(self) -> agent_types.Trajectory:
    # [FUNCTION DOC]
    # Abstract property: the Trajectory of the current episode. Base raises
    # NotImplementedError.
    """Get the complete trajectory for the current task/episode."""
    raise NotImplementedError

  # ──────────────────────────────────────────────────────────────
  # Environment Interaction Interface
  # ──────────────────────────────────────────────────────────────

  @abc.abstractmethod
  def update_from_env(
      self,
      observation: Any,
      reward: float,
      done: bool,
      info: Dict[str, Any] | None = None,
      **kwargs,
  ) -> None:
    # [FUNCTION DOC]
    # Abstract. Consume environment feedback (observation, reward, done flag,
    # optional info dict, extra kwargs) after an action. Returns None.
    # Base raises NotImplementedError.
    """Process feedback from environment after action execution."""
    raise NotImplementedError("update_from_env is not implemented.")

  async def update_from_env_async(self, *args, **kwargs) -> None:
    # [FUNCTION DOC]
    # Async wrapper: obtains the current event loop and runs update_from_env in the
    # loop's default executor (a thread) with the given positional arguments.
    # Inputs: positional args forwarded. Outputs: awaitable resolving to the sync
    # method's return value.
    # KNOWN ISSUE: loop.run_in_executor does not accept keyword arguments, so passing
    # any keyword argument raises TypeError (they would need functools.partial).
    # Also asyncio.get_event_loop is deprecated inside coroutines in favor of
    # get_running_loop. Side effects: those of update_from_env, executed in another
    # thread.
    """Asynchronous version of update_from_env."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, self.update_from_env, *args, **kwargs
    )

  # ──────────────────────────────────────────────────────────────
  # Model Interaction Interface
  # ──────────────────────────────────────────────────────────────

  @abc.abstractmethod
  def update_from_model(self, response: str, **kwargs) -> agent_types.Action:
    # [FUNCTION DOC]
    # Abstract. Consume an LLM response string (plus kwargs) and return the parsed
    # Action. Base raises NotImplementedError.
    """Process LLM response and extract structured action."""
    raise NotImplementedError("update_from_model is not implemented.")

  async def update_from_model_async(
      self, *args, **kwargs
  ) -> agent_types.Action:
    # [FUNCTION DOC]
    # Async wrapper running update_from_model in the default executor thread and
    # returning its Action.
    # KNOWN ISSUE: same as update_from_env_async; keyword arguments cause a TypeError
    # because run_in_executor only forwards positional arguments.
    """Asynchronous version of update_from_model."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, self.update_from_model, *args, **kwargs
    )

  # ──────────────────────────────────────────────────────────────
  # Lifecycle Management
  # ──────────────────────────────────────────────────────────────

  @abc.abstractmethod
  def reset(self) -> None:
    # [FUNCTION DOC]
    # Abstract. Reset agent state for a new episode. Base body does nothing.
    """Reset agent state for a new episode."""
    ...

  # ──────────────────────────────────────────────────────────────
  # Debugging and Introspection
  # ──────────────────────────────────────────────────────────────

  def get_current_step(self) -> agent_types.Step | None:
    # [FUNCTION DOC]
    # Return the last Step of self.trajectory, or None when there are no steps.
    # No side effects.
    """Get the most recent step for debugging and introspection."""
    if not self.trajectory.steps:
      return None
    return self.trajectory.steps[-1]


class ConversationAgentBase(LLMBaseAgent):
  # [CLASS DOC]
  # Concrete base for chat agents. State: system_prompt (str), _trajectory (a fresh
  # Trajectory), _messages (list of message dicts, starts with one system message),
  # step (int counter, set to 0; not otherwise updated here). Subclasses implement
  # update_from_model and may override _init_messages and _observation_to_messages.
  # KNOWN ISSUE: the docstring mentions caching the last observation and requires
  # implementing _observation_to_messages, but no cache exists and a default
  # implementation is provided.
  """Base class for chat-style LLM agents with trajectory support.

  This class implements common functionality for agents that:
  * Maintain a list of chat messages (`_messages`) to send to the LLM.
  * Maintain a `Trajectory` of `Step` objects for RL training.
  * Cache the last environment observation for step recording.

  Subclasses are expected to:
  * Provide a system prompt via constructor.
  * Implement `_observation_to_messages()` to convert environment observations
    into chat messages.
  * Implement `update_from_model()` to parse LLM responses into `Action`s and
    append new `Step`s to the trajectory.
  """

  def __init__(self, system_prompt: str):
    # [FUNCTION DOC]
    # Store system_prompt, create an empty Trajectory and message list, initialize
    # messages via _init_messages(system_prompt) and set step to 0.
    # Inputs: system_prompt string. Side effects: sets the attributes above.
    self.system_prompt = system_prompt
    self._trajectory = agent_types.Trajectory()
    self._messages: list[dict[str, Any]] = []
    self._init_messages(system_prompt)
    self.step = 0

  # ---------- Internal helpers ----------

  def _init_messages(self, system_prompt: str) -> None:
    # [FUNCTION DOC]
    # HELPER: Replace the message list with a single system message whose content is
    # system_prompt, or the empty string when system_prompt is falsy. Subclasses may
    # override to inject e.g. tool documentation.
    """Initialize conversation history with a system prompt.

    Subclasses may override this to inject additional content (e.g., tool
    documentation) into the initial system message.

    Args:
      system_prompt: The system prompt to use.
    """
    self._messages = [{"role": "system", "content": system_prompt or ""}]

  def _observation_to_messages(
      self, observation: Any, reward: float, done: bool, info: Dict[str, Any]
  ) -> None:
    # [FUNCTION DOC]
    # HELPER: Append environment observations as user messages.
    # Inputs: observation, reward, done, info (the last three ignored).
    # Behavior: if observation is a dict with key "prompts", append a user message
    # with that value (empty string if falsy); else if a dict with key "question",
    # append a user message with that value (empty string if falsy); else if a
    # string, append it as a user message; otherwise do nothing.
    # Side effects: mutates self._messages. Outputs: None.
    # KNOWN ISSUE: the docstring does not mention the "prompts" key, which takes
    # priority over "question".
    """Convert environment observation into chat messages.

    Default behavior:
    * If observation is a dict containing "question", use it as user content.
    * If observation is a string, append as a user message.
    * Otherwise, do nothing.

    Subclasses can override this to handle richer observation formats.

    Args:
      observation: The observation from the environment.
      reward: The reward from the environment.
      done: Whether the episode is done.
      info: Additional information from the environment.
    """
    del reward, done, info  # Unused in default implementation.
    # prompts should not be applied with template beforehand to avoid double
    # templating.
    if isinstance(observation, dict) and "prompts" in observation:
      self._messages.append(
          {"role": "user", "content": observation["prompts"] or ""}
      )
    elif isinstance(observation, dict) and "question" in observation:
      self._messages.append(
          {"role": "user", "content": observation["question"] or ""}
      )
    elif isinstance(observation, str):
      self._messages.append({"role": "user", "content": observation})

  # ---------- Properties ----------

  @property
  def chat_completions(self) -> list[dict[str, str]]:
    # [FUNCTION DOC]
    # Property returning the live message list (not a copy).
    return self._messages

  @property
  def trajectory(self) -> agent_types.Trajectory:
    # [FUNCTION DOC]
    # Property returning the live Trajectory object.
    return self._trajectory

  # ---------- Public interface implementations ----------

  def update_from_env(
      self,
      observation: Any,
      reward: float,
      done: bool,
      info: Dict[str, Any] | None = None,
      **kwargs,
  ) -> None:
    # [FUNCTION DOC]
    # Apply environment feedback.
    # Inputs: observation, reward, done, optional info dict, extra kwargs (ignored).
    # Behavior: if the trajectory has no task yet, sets it from this first
    # observation: a string becomes a dict with key "prompts" mapped to a one-element
    # list containing the string; anything else is deep-copied. Then, if a current
    # (last) step exists, sets its observation, reward, done and info (empty dict if
    # info is falsy). Finally, if observation is not None, calls
    # _observation_to_messages to extend the conversation.
    # Outputs: None. Side effects: mutates trajectory and messages.
    """Update current step with environment feedback and extend conversation."""
    # First observation from env is the task specification.
    if self._trajectory.task is None:
      if isinstance(observation, str):
        self._trajectory.task = {"prompts": [observation]}
      else:
        self._trajectory.task = copy.deepcopy(observation)

    step = self.get_current_step()
    if step:
      step.observation = observation
      step.reward = reward
      step.done = done
      step.info = info or {}

    # Let subclass / default handler convert observation into messages.
    if observation is not None:
      self._observation_to_messages(observation, reward, done, info)

  def reset(self) -> None:
    # [FUNCTION DOC]
    # Start a new episode: replace the trajectory with a fresh one, re-initialize
    # messages with the stored system prompt and set step to 0.
    """Reset trajectory, cache, and conversation history."""
    self._trajectory = agent_types.Trajectory()
    self._init_messages(self.system_prompt)
    self.step = 0
