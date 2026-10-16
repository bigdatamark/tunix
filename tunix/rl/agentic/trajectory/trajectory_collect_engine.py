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
# Per-episode rollout engine. TrajectoryCollectEngine drives one agent through
# one environment episode: reset env and agent, then loop model call -> agent
# parses action -> env.step (in a worker thread with a deadline) -> agent records
# feedback, while tracking response-token budget, wall-clock timeout and step
# limit; afterwards it optionally adds a final reward, computes discounted
# Monte-Carlo returns and the total reward, closes the env, and returns the
# result in one of several formats ("Trajectory", "Steps", "Token",
# "Conversation"). Also offers collect_multiple to run many pairs concurrently.
# Uses numpy, absl logging, the perf tracer (spans around env steps), the agentic
# tokenization utilities, and base_rollout.RolloutOutput as the model-call result
# type (fields text, tokens, logprobs, left_padded_prompt_tokens, each indexed
# by sample, of which sample 0 is used).
# Defines unused ParamSpec P and module aliases BaseTaskEnv and
# ConversationAgentBase.
"""Engine for collecting trajectories from agent-environment interactions.

This module defines the `TrajectoryCollectEngine`, which facilitates the
asynchronous collection of rollouts by managing the interaction loop between
an LLM-based agent and an environment. It supports single and concurrent
multi-pair trajectory collection.
"""
import asyncio
import json
import time
from typing import Any, AsyncGenerator, Callable, Concatenate, Dict, List, Optional, ParamSpec, Set, Tuple

from absl import logging
import numpy as np
from tunix.perf.experimental import constants as perf_constants
from tunix.perf.experimental import tracer as perf_tracer_v2
from tunix.rl.agentic import utils
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.agents import base_agent
from tunix.rl.agentic.environments import base_environment
from tunix.rl.rollout import base_rollout

P = ParamSpec("P")

BaseTaskEnv = base_environment.BaseTaskEnv
ConversationAgentBase = base_agent.ConversationAgentBase


class TrajectoryCollectEngine:
  # [CLASS DOC]
  # Engine for one agent/env pair (create a new engine per episode).
  # Key state: agent, env, model_call (+ model_call_kwargs), final_reward_fn
  # (read from env at reset), perf_v2 tracer (no-op by default), max_steps (from
  # env.max_steps, default 1), gamma, max_response_length, a running
  # _response_token_count, timeout seconds, tokenizer and chat_parser (both needed
  # for token tracking), _start_ts, filter_statuses, overlong_filter, and timing
  # accumulators env_time (reset/step latency and cpu time) and reward_time
  # (reward latency and cpu time).
  """Asynchronous trajectory collection engine for agent-env interactions.

  This engine orchestrates complete rollout episodes by managing the interaction
  loop between LLM-based agents and environments. It handles model inference,
  environment stepping, reward computation, and trajectory storage with support
  for concurrent multi-pair execution and streaming results.

  The engine implements the standard RL rollout pattern: reset → step* → final
  reward computation → return calculation, while providing flexible callback
  integration for custom model calls and reward functions.
  """

  def __init__(
      self,
      agent: ConversationAgentBase,
      env: BaseTaskEnv,
      *,
      model_call: Callable[..., base_rollout.RolloutOutput],
      model_call_kwargs: Optional[Dict[str, Any]] = None,
      gamma: float = 1.0,
      max_response_length: Optional[int] = None,
      timeout: float = 600.0,
      tokenizer=None,
      chat_parser=None,
      filter_statuses: Optional[Set[agent_types.TrajectoryStatus]] = None,
      overlong_filter: bool = False,
      perf_v2: Optional[perf_tracer_v2.Tracer] = None,
  ):
    # [FUNCTION DOC]
    # Store configuration.
    # Inputs: agent, env; keyword-only model_call (callable invoked as
    # model_call(chat_completions, env, max_generation_steps=..., **model_call_kwargs)
    # returning a RolloutOutput), model_call_kwargs (default empty), gamma (1.0),
    # max_response_length (None = unlimited), timeout (600 s), tokenizer,
    # chat_parser, filter_statuses (default set: MAX_STEPS_REACHED,
    # MAX_CONTEXT_LIMIT_REACHED, TIMEOUT, ENV_TIMEOUT; an empty set also falls back to
    # the default), overlong_filter (False), perf_v2 tracer (None -> NoopTracer).
    # Initializes counters and timing dicts to 0.0. Logs a warning if
    # max_response_length is set without both tokenizer and chat_parser.
    # KNOWN ISSUE: docstring documents a final_reward_fn argument that does not exist
    # (it is taken from the env at reset) and describes model_call as taking only chat
    # completions and returning a string. perf_v2 is assigned twice (harmless).
    # Note that the token budget is still enforced from model output tokens even
    # without tokenizer/chat_parser, contrary to the warning text.
    """Initialize the trajectory collection engine.

    Args:
        agent (ConversationAgentBase): The agent that will interact with the
          environment
        env (BaseTaskEnv): The environment providing tasks and feedback
        model_call (Callable): Function that takes chat completions as first
          argument with optional kwargs and returns model response string.
          Handles the actual LLM inference.
        model_call_kwargs (Optional[Dict[str, Any]]): Optional kwargs to pass to
          model_call.
        final_reward_fn (Optional[Callable]): Optional function to compute
          additional reward at episode end. Takes (task, response) and returns
          float. Defaults to zero if not provided.
        gamma (float): Discount factor for MC reward calculation (1.0 = no
          discounting).
        max_response_length (Optional[int]): Maximum number of context tokens to
          use before forced termination.
        timeout (float): Maximum episode duration in seconds before timeout
          termination
        tokenizer: Optional tokenizer for converting messages to token IDs. This
          is required if we want to track down token counts.
        chat_parser: Optional chat parser for formatting messages
        filter_statuses (Set[TrajectoryStatus]): A set of statuses that are
          masked out for overlong filtering.
        overlong_filter: Whether to filter overlong trajectories.
        perf_v2 (Optional[perf_tracer_v2.Tracer]): Optional performance tracer
          to use for performance measurements. Defaults to a no-op tracer.
    """
    self.agent = agent
    self.env = env
    self.model_call = model_call
    self.final_reward_fn = None
    self.model_call_kwargs = model_call_kwargs or {}
    self.perf_v2 = (
        perf_v2 if perf_v2 is not None else perf_tracer_v2.NoopTracer()
    )
    self.max_steps = getattr(self.env, "max_steps", 1)
    self.gamma = gamma
    self.max_response_length = max_response_length
    self._response_token_count = 0
    self.timeout = timeout

    # Tokenizer utilities for stepwise tokenization
    self.tokenizer = tokenizer
    self.chat_parser = chat_parser
    self._start_ts: float = 0.0
    self.filter_statuses = filter_statuses or {
        agent_types.TrajectoryStatus.MAX_STEPS_REACHED,
        agent_types.TrajectoryStatus.MAX_CONTEXT_LIMIT_REACHED,
        agent_types.TrajectoryStatus.TIMEOUT,
        agent_types.TrajectoryStatus.ENV_TIMEOUT,
    }

    self.overlong_filter = overlong_filter
    self.perf_v2 = perf_v2 or perf_tracer_v2.NoopTracer()
    self.env_time = {
        "reset_latency": 0.0,  # Wall-clock time (Total real-world time elapsed)
        "reset_cpu_time": (
            0.0
        ),  # Thread/CPU time (Actual processing time on the worker thread)
        "step_latency": 0.0,  # Wall-clock time (Total real-world time elapsed)
        "step_cpu_time": (
            0.0
        ),  # Thread/CPU time (Actual processing time on the worker thread)
    }
    self.reward_time = {
        "reward_latency": (
            0.0
        ),  # Wall-clock time (Total real-world time elapsed)
        "reward_cpu_time": (
            0.0
        ),  # Thread/CPU time (Actual processing time on the worker thread)
    }

    if self.max_response_length is not None and not (
        self.tokenizer and self.chat_parser
    ):
      logging.warning(
          "max_response_length is set to %d, but no tokenizer or chat_parser is"
          " provided. response length limits will not be enforced.",
          self.max_response_length,
      )

  async def _run_with_timing(
      self, func: Callable[..., Any], *args, timeout: Optional[float] = None
  ) -> Tuple[Any, float, float]:
    # [FUNCTION DOC]
    # HELPER: Coroutine. Run a synchronous func(*args) in the default thread executor
    # and measure it.
    # Inputs: func, positional args, optional keyword-only timeout in seconds.
    # Behavior: wraps func to measure thread CPU time on the worker thread; measures
    # wall time around the await; when timeout is given, awaits with
    # asyncio.wait_for (cancelling the await, though the worker thread keeps running).
    # Outputs: tuple (result, wall seconds, cpu seconds).
    # Exceptions: asyncio.TimeoutError on timeout; anything func raises.
    """Runs a sync function in an executor and returns (result, wall_time, cpu_time).

    Args:
      func: Synchronous callable to run in the executor.
      *args: Positional arguments forwarded to func.
      timeout: Optional deadline in seconds. If provided and the executor future
        does not finish in time, asyncio.TimeoutError is re-raised to the
        caller.

    Raises:
      asyncio.TimeoutError: When timeout is not None and is exceeded.
    """

    def _clocked_wrapper():
      # [FUNCTION DOC]
      # HELPER: Runs func(*args) on the worker thread and returns (result, thread CPU
      # time delta).
      t_start = time.thread_time()
      res = func(*args)
      t_delta = time.thread_time() - t_start
      return res, t_delta

    loop = asyncio.get_running_loop()
    wall_start = time.perf_counter()

    fut = loop.run_in_executor(None, _clocked_wrapper)
    if timeout is not None:
      result, cpu_delta = await asyncio.wait_for(fut, timeout=timeout)
    else:
      result, cpu_delta = await fut

    wall_delta = time.perf_counter() - wall_start
    return result, wall_delta, cpu_delta

  async def collect(self, mode: str = "Conversation") -> Any:
    # [FUNCTION DOC]
    # Coroutine. Run a full episode and return output in the requested format.
    # Inputs: mode, one of "Trajectory", "Steps", "Token", "Conversation" (default
    # "Conversation").
    # Behavior: awaits _reset, sets status RUNNING, then loops: if the number of
    # recorded steps reached max_steps, set MAX_STEPS_REACHED and stop; otherwise run
    # _one_step and stop when it reports done, setting SUCCEEDED if the status is
    # still RUNNING. masked_out is true when overlong_filter is on and the final status
    # is in filter_statuses. Unless masked_out, awaits _append_final_reward. Then
    # computes MC returns and the trajectory reward; always closes the env afterwards.
    # Mode validation happens after the episode runs (ValueError for unknown modes).
    # Outputs per mode:
    # "Trajectory": the agent's Trajectory with env_time and reward_time attached.
    # "Steps": list of dicts per step with assistant_text (model_response), env_text
    # (observation), done, assistant_tokens, assistant_masks, env_tokens, env_masks,
    # reward, mc_return, env_time, reward_time.
    # "Token": dict with conversation_text (chat messages), prompt_tokens (the
    # trajectory's ad hoc prompt_tokens attribute or empty list),
    # conversation_tokens (concatenation, per step, of assistant tokens then env
    # tokens, empty segments dropped, int32 empty array if none),
    # conversation_masks (concatenated masks, all zeros when masked_out),
    # status name, trajectory_reward, env_time, reward_time, old_logprobs
    # (concatenation of per-step logprobs aligned with tokens: model logprobs for
    # assistant tokens, or zeros if missing, and zeros for env tokens; None if empty),
    # policy_version (env.task "policy_version"), original_input (trajectory task),
    # group_id (env.extra_kwargs "group_id").
    # "Conversation": the agent's live chat message list.
    # Exceptions: AssertionError if a step's logprobs length differs from its
    # assistant tokens; errors from env/model propagate (after env close when raised
    # in the reward phase).
    # KNOWN ISSUE: in "Token" mode masks and tokens are filtered independently by
    # emptiness, so a step with tokens but empty/None masks would misalign; also
    # env tokens may be None-checked but env_masks used without check.
    """Execute a complete rollout episode and return the resulting trajectory.

    Orchestrates the full interaction sequence: environment reset, iterative
    agent-environment steps, final reward computation, Monte Carlo return
    calculation, and resource cleanup.

    Args:
        mode (str): Output format. Options: 
          - "Trajectory": return full Trajectory object.
          - "Token": return flattened tokenized dict for training.
          - "Steps": return stepwise tokenized data only.
          - "Conversation": return raw conversation messages (default).

    Returns:
        Trajectory | dict | list: Depending on mode.
    """  # fmt: skip
    await self._reset()

    self.agent.trajectory.status = agent_types.TrajectoryStatus.RUNNING

    while True:
      if len(self.agent.trajectory.steps) >= self.max_steps:
        self.agent.trajectory.status = (
            agent_types.TrajectoryStatus.MAX_STEPS_REACHED
        )
        break

      done = await self._one_step()

      if done:
        if self.agent.trajectory.status == agent_types.TrajectoryStatus.RUNNING:
          self.agent.trajectory.status = agent_types.TrajectoryStatus.SUCCEEDED
        break

    masked_out = (
        self.overlong_filter
        and self.agent.trajectory.status in self.filter_statuses
    )
    try:
      if not masked_out:
        await self._append_final_reward()
      else:
        logging.debug(
            "%s mask out trajectory due to status %s",
            self._debug_prefix,
            self.agent.trajectory.status.name,
        )
      self.compute_mc_reward()
      self.compute_trajectory_reward()
    finally:
      await self._close()

    if mode not in ["Trajectory", "Steps", "Token", "Conversation"]:
      raise ValueError(
          f"Unsupported mode: {mode}, currently supported modes: "
          f" {['Trajectory', 'Steps', 'Token', 'Conversation']}",
      )

    if mode == "Trajectory":
      self.agent.trajectory.env_time = self.env_time
      self.agent.trajectory.reward_time = self.reward_time
      return self.agent.trajectory
    elif mode == "Steps":
      return [
          {
              "assistant_text": getattr(step, "model_response", ""),
              "env_text": getattr(step, "observation", ""),
              "done": getattr(step, "done", False),
              "assistant_tokens": getattr(step, "assistant_tokens", []),
              "assistant_masks": getattr(step, "assistant_masks", []),
              "env_tokens": getattr(step, "env_tokens", []),
              "env_masks": getattr(step, "env_masks", []),
              "reward": step.reward,
              "mc_return": step.mc_return,
              "env_time": self.env_time,
              "reward_time": self.reward_time,
          }
          for step in self.agent.trajectory.steps
      ]
    elif mode == "Token":
      # flatten all steps into single batch dict
      conversation_tokens, conversation_masks, logprobs = [], [], []
      prompt_tokens = getattr(self.agent.trajectory, "prompt_tokens", [])

      for step in self.agent.trajectory.steps:
        # Keep tokens/masks/logprobs appended in lockstep — a step with env_tokens
        # but no vllm logprobs (initial observation, empty completion) would
        # otherwise leave the logprobs array short by `len(env_tokens)` and offset
        # every subsequent step.
        assistant_tokens = getattr(step, "assistant_tokens", None)
        env_tokens = getattr(step, "env_tokens", None)
        step_logprobs = getattr(step, "logprobs", None)
        if assistant_tokens is not None:
          conversation_tokens.append(assistant_tokens)
          conversation_masks.append(step.assistant_masks)
          if step_logprobs is not None:
            assert len(step_logprobs) == len(assistant_tokens), (
                f"Logprobs length {len(step_logprobs)} does not match assistant"
                f" tokens length {len(assistant_tokens)}"
            )
            logprobs.append(step_logprobs)
          else:
            logprobs.append(np.zeros(len(assistant_tokens)))
        if env_tokens is not None:
          conversation_tokens.append(env_tokens)
          conversation_masks.append(step.env_masks)
          logprobs.append(np.zeros(len(env_tokens)))

      conversation_tokens = [
          np.asarray(tokens)
          for tokens in conversation_tokens
          if len(tokens) > 0
      ]
      conversation_masks = [
          np.asarray(masks) for masks in conversation_masks if len(masks) > 0
      ]
      logprobs = [
          np.asarray(step_logprobs)
          for step_logprobs in logprobs
          if len(step_logprobs) > 0
      ]
      conversation_masks = (
          np.concatenate(conversation_masks, axis=0)
          if conversation_masks
          else np.array([], dtype=np.int32)
      )
      conversation_tokens = (
          np.concatenate(conversation_tokens, axis=0)
          if conversation_tokens
          else np.array([], dtype=np.int32)
      )
      final_masks = (
          np.zeros_like(conversation_masks)
          if masked_out
          else conversation_masks
      )

      return {
          "conversation_text": self.agent.chat_completions,
          "prompt_tokens": prompt_tokens,
          "conversation_tokens": conversation_tokens,
          "conversation_masks": final_masks,
          "status": self.agent.trajectory.status.name,
          "trajectory_reward": self.agent.trajectory.reward,
          "env_time": self.env_time,
          "reward_time": self.reward_time,
          "old_logprobs": (
              np.concatenate(logprobs, axis=0) if logprobs else None
          ),
          "policy_version": self.env.task.get("policy_version"),
          "original_input": self.agent.trajectory.task,
          "group_id": self.env.extra_kwargs.get("group_id"),
      }
    elif mode == "Conversation":
      # return raw conversation history
      return self.agent.chat_completions

  @staticmethod
  async def collect_multiple(
      pairs: List[Tuple[ConversationAgentBase, BaseTaskEnv]],
      *,
      model_call: Callable[..., base_rollout.RolloutOutput],
      gamma: float = 1.0,
      timeout: float = 30.0,
      max_response_length: Optional[int] = None,
      mode: str = "Trajectory",
      filter_statuses: Optional[Set[agent_types.TrajectoryStatus]] = None,
      overlong_filter: bool = True,
      perf_v2: Optional[perf_tracer_v2.Tracer] = None,
  ) -> AsyncGenerator[Tuple[int, Any], None]:
    # [FUNCTION DOC]
    # Static async generator. Run many (agent, env) pairs concurrently and yield
    # (pair_index, result) tuples in completion order.
    # Inputs: pairs list; keyword-only model_call, gamma (1.0), timeout (30 s),
    # max_response_length, mode (default "Trajectory"), filter_statuses,
    # overlong_filter (default True, unlike the constructor), perf_v2.
    # Behavior: creates one engine per pair (without tokenizer, chat_parser or
    # model_call_kwargs) and collects with the given mode, using asyncio.as_completed.
    # Exceptions: the first failing episode's exception propagates to the consumer.
    # KNOWN ISSUE: since tokenizer/chat_parser are never passed, Token/Steps modes
    # from this method carry no env tokens and no assistant tokens.
    """Execute multiple agent-environment pairs concurrently.

    Runs multiple rollouts in parallel and yields completed trajectories
    as they finish, enabling efficient batch processing with streaming
    results. Useful for distributed training or large-scale evaluation.

    Args:
        pairs (List[Tuple[ConversationAgentBase, BaseTaskEnv]]): List of (agent,
          environment) pairs
        model_call (Callable): Shared model inference function for all pairs
        gamma (float): Discount factor for return calculation
        max_response_length (Optional[int]): Maximum context limit per episode
        timeout (float): Per-episode timeout in seconds
        mode (str): Output format. See `collect` method for options.
        filter_statuses (Optional[Set[TrajectoryStatus]]): A set of statuses
          that are masked out for filtering.
        overlong_filter (bool): Whether to filter overlong trajectories.
        perf_v2 (Optional[perf_tracer_v2.Tracer]): Optional performance tracer
          to use for performance measurements.

    Yields:
        Tuple[int, Any]: `(pair_index, result)`. The type of `result`
          depends on the `mode` argument. See the `collect` method for details.
    """

    async def _run_one(i: int, agent: ConversationAgentBase, env: BaseTaskEnv):
      # [FUNCTION DOC]
      # HELPER: Coroutine building an engine for pair i and returning (i, collected
      # result).
      """Execute a single agent-env pair with the given configuration."""
      engine = TrajectoryCollectEngine(
          agent,
          env,
          model_call=model_call,
          gamma=gamma,
          max_response_length=max_response_length,
          timeout=timeout,
          filter_statuses=filter_statuses,
          overlong_filter=overlong_filter,
          perf_v2=perf_v2,
      )
      traj = await engine.collect(mode=mode)
      return i, traj

    # Launch all pairs concurrently and yield results as they complete
    tasks = [_run_one(i, agent, env) for i, (agent, env) in enumerate(pairs)]
    for coro in asyncio.as_completed(tasks):
      yield await coro

  async def _reset(self):
    # [FUNCTION DOC]
    # HELPER: Coroutine. Start an episode.
    # Behavior: runs env.reset in the executor with timing (adds to reset latency/cpu
    # counters), takes the observation; sets final_reward_fn to env.final_reward_fn if
    # the env has that attribute, else None; resets the agent and feeds the
    # observation via update_from_env with reward 0.0, done False, empty info. If both
    # tokenizer and chat_parser are set, tokenizes the current chat messages as the
    # first messages with a generation prompt and stores the token list on the
    # trajectory as a new attribute prompt_tokens. Records the start timestamp and
    # zeroes the response token count.
    # Exceptions: propagates reset/agent/tokenization errors.
    """Resets the environment and agent at the beginning of a new episode.

    This involves calling the environment's reset method, updating the agent's
    state, and optionally tokenizing the initial prompt messages.
    """
    logging.debug("%s env.reset starting", self._debug_prefix)
    (obs, _), wall_time, cpu_time = await self._run_with_timing(self.env.reset)
    logging.debug(
        "%s env.reset done in %.1fs",
        self._debug_prefix,
        wall_time,
    )

    self.env_time["reset_latency"] += wall_time
    self.env_time["reset_cpu_time"] += cpu_time
    self.final_reward_fn = (
        self.env.final_reward_fn
        if hasattr(self.env, "final_reward_fn")
        else None
    )
    self.agent.reset()
    self.agent.update_from_env(observation=obs, reward=0.0, done=False, info={})

    if self.tokenizer is not None and self.chat_parser is not None:
      # Get the current messages (usually System + User)
      init_messages = self.agent.chat_completions
      prompt_tokens, _ = utils.tokenize_and_generate_masks(
          init_messages,
          tokenizer=self.tokenizer,
          parser=self.chat_parser,
          contains_first_msg=True,
          contains_generation_msg=True,
      )
      self.agent.trajectory.prompt_tokens = prompt_tokens

    self._start_ts = time.perf_counter()
    self._response_token_count = 0

  @property
  def _debug_prefix(self) -> str:
    # [FUNCTION DOC]
    # HELPER: Property returning a log prefix
    # "[step_idx=<n>, pair_index=<p>, group_id=<g>]" from the number of recorded
    # steps and the env's extra_kwargs (None when missing).
    """Returns a consistent log prefix with step_idx, pair_index, and group_id."""
    extra = getattr(self.env, "extra_kwargs", {}) or {}
    step_idx = len(self.agent.trajectory.steps)
    pair_index = extra.get("pair_index")
    group_id = extra.get("group_id")
    return (
        f"[step_idx={step_idx}, pair_index={pair_index}, group_id={group_id}]"
    )

  def _get_perf_tags(self) -> Dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: Build perf-tracing tags: group id and pair index from env.extra_kwargs
    # (when present and not None) under the perf constants GROUP_ID and PAIR_INDEX,
    # and env.task "policy_version" under STEP when not None.
    # Exceptions: AttributeError if env.task exists but is not a mapping.
    """Extracts performance tracing tags from the environment."""
    tags = {}
    if hasattr(self.env, "extra_kwargs"):
      group_id = self.env.extra_kwargs.get("group_id")
      if group_id is not None:
        tags[perf_constants.GROUP_ID] = group_id
      pair_index = self.env.extra_kwargs.get("pair_index")
      if pair_index is not None:
        tags[perf_constants.PAIR_INDEX] = pair_index
    if hasattr(self.env, "task"):
      policy_version = self.env.task.get("policy_version")
      if policy_version is not None:
        tags[perf_constants.STEP] = policy_version
    return tags

  def _check_and_set_context_limit_reached(self) -> bool:
    # [FUNCTION DOC]
    # HELPER: If max_response_length is set and the response token count has reached
    # it, set status MAX_CONTEXT_LIMIT_REACHED, log, and return True; else False.
    """Returns True and updates trajectory status if response budget is exhausted."""
    if (
        self.max_response_length is not None
        and self._response_token_count >= self.max_response_length
    ):
      self.agent.trajectory.status = (
          agent_types.TrajectoryStatus.MAX_CONTEXT_LIMIT_REACHED
      )
      logging.debug("%s MAX_CONTEXT_LIMIT_REACHED", self._debug_prefix)
      return True
    return False

  async def _one_step(self) -> bool:
    # [FUNCTION DOC]
    # HELPER: Coroutine. Execute one agent/env step; return True if the episode ends.
    # Behavior:
    # 1. Return True if the token budget is already exhausted.
    # 2. Compute max_generation_steps as the remaining budget (None if unlimited) and
    # call model_call in the default executor (exceptions are logged and re-raised).
    # 3. On the first step, if the output has left-padded prompt tokens, replace the
    # trajectory's prompt_tokens with sample 0 of them.
    # 4. Add the length of sample 0's tokens to the count; if the budget is now
    # exhausted return True (the response is discarded, not recorded as a step).
    # 5. Pass sample 0's text to agent.update_from_model and take its action payload
    # (None becomes an empty list with a warning).
    # 6. Step the env in the executor inside a perf span, with timeout equal to the
    # remaining episode time. On timeout: status ENV_TIMEOUT, log an error, mark the
    # current step done and return True.
    # 7. Accumulate step timings, log observation/info, call agent.update_from_env.
    # 8. Attach sample 0 logprobs to the current step when available.
    # 9. If tokenizer and chat_parser are set: when an assistant message exists, set
    # the step's assistant tokens to sample 0 tokens and masks to ones; when there are
    # env messages after it and the episode neither ended nor timed out, tokenize
    # them (not first, with generation prompt), store env tokens and masks as arrays
    # and add their count to the budget.
    # 10. If the overall timeout has passed: status TIMEOUT, mark step done, return
    # True. Otherwise return the env's done flag.
    # KNOWN ISSUE: if remaining time is already negative, wait_for times out
    # immediately and the status becomes ENV_TIMEOUT instead of TIMEOUT. The TIMEOUT
    # branch dereferences get_current_step() without a None check. Rewards returned
    # by env.step are stored as is; reward functions returning RewardOutput objects
    # would break the later numeric sums.
    """Executes a single step and returns the Step object and Done status.

    This involves calling the model, updating the agent with the response,
    stepping the environment with the agent's action, and updating the agent
    with the environment's feedback.

    Returns:
        bool: True if the episode is done (either by environment or timeout),
          False otherwise.
    """
    if self._check_and_set_context_limit_reached():
      return True
    max_generation_steps = (
        self.max_response_length - self._response_token_count
        if self.max_response_length is not None
        else None
    )
    logging.debug("%s model_call starting", self._debug_prefix)

    def _safe_model_call():
      # [FUNCTION DOC]
      # HELPER: Calls model_call(chat_completions, env, max_generation_steps=...,
      # **model_call_kwargs); logs with traceback and re-raises any exception.
      try:
        return self.model_call(
            self.agent.chat_completions,
            self.env,
            max_generation_steps=max_generation_steps,
            **self.model_call_kwargs,
        )
      except Exception as e:
        logging.exception("Caught exception inside model_call: %s", e)
        raise

    rollout_output = await asyncio.get_event_loop().run_in_executor(
        None,
        _safe_model_call,
    )
    logging.debug("%s model_call done", self._debug_prefix)

    # Align trajectory prompt tokens with the rollout worker's actual
    # tokenization on the first turn to prevent prompt token desync.
    if (
        not self.agent.trajectory.steps
        and rollout_output.left_padded_prompt_tokens is not None
    ):
      self.agent.trajectory.prompt_tokens = (
          rollout_output.left_padded_prompt_tokens[0]
      )

    if rollout_output.tokens:
      self._response_token_count += len(rollout_output.tokens[0])
    if self._check_and_set_context_limit_reached():
      return True

    action = self.agent.update_from_model(rollout_output.text[0]).action
    logging.debug(
        "%s Agent Action:\n%s",
        self._debug_prefix,
        json.dumps(action, default=str, indent=2),
    )
    if action is None:
      logging.warning(
          "Agent returned None action, using empty action list as fallback"
      )
      action = []

    step_idx = len(self.agent.trajectory.steps)
    remaining_time = self.timeout - (time.perf_counter() - self._start_ts)

    tags = self._get_perf_tags()
    try:
      with self.perf_v2.span(
          perf_constants.ENVIRONMENT,
          tags=tags,
      ):
        (obs, rew, done, info), wall_time, cpu_time = (
            await self._run_with_timing(
                self.env.step, action, timeout=remaining_time
            )
        )
    except asyncio.TimeoutError:
      self.agent.trajectory.status = agent_types.TrajectoryStatus.ENV_TIMEOUT
      if step_idx == 0:
        logging.error(
            "%s env.step hung at step 0 (first action) and was killed after"
            " %.1f s remaining timeout. This trajectory produced no usable"
            " data. Consider investigating the environment.",
            self._debug_prefix,
            remaining_time,
        )
      else:
        logging.error(
            "%s env.step hung at step %d and was killed after %.1f s"
            " remaining timeout.",
            self._debug_prefix,
            step_idx,
            remaining_time,
        )
      cur_step = self.agent.get_current_step()
      if cur_step is not None:
        cur_step.done = True
      return True

    self.env_time["step_latency"] += wall_time
    self.env_time["step_cpu_time"] += cpu_time

    logging.debug(
        "%s Env Observation (Rew: %s, Done: %s):\n%s",
        self._debug_prefix,
        rew,
        done,
        json.dumps(obs, default=str, indent=2),
    )
    logging.debug(
        "%s Env Info:\n%s",
        self._debug_prefix,
        json.dumps(info, default=str, indent=2),
    )
    self.agent.update_from_env(obs, rew, done, info)

    cur_step = self.agent.get_current_step()

    if cur_step is not None and rollout_output.logprobs is not None:
      cur_step.logprobs = rollout_output.logprobs[0]

    step_timed_out = time.perf_counter() - self._start_ts > self.timeout
    if cur_step is not None and self.tokenizer and self.chat_parser:
      assistant_message, env_messages = (
          utils.get_recent_assistant_user_messages(self.agent.chat_completions)
      )

      # Assistant tokens/masks
      if assistant_message:
        cur_step.assistant_tokens = rollout_output.tokens[0]
        cur_step.assistant_masks = np.ones_like(rollout_output.tokens[0])

      # Environment tokens/masks
      # Terminal-step environment messages are not appended to the response
      # token stream when the step ends the trajectory.
      if env_messages and not done and not step_timed_out:
        e_tokens, e_masks = utils.tokenize_and_generate_masks(
            env_messages,
            tokenizer=self.tokenizer,
            parser=self.chat_parser,
            contains_first_msg=False,
            contains_generation_msg=True,
        )
        cur_step.env_tokens = np.array(e_tokens)
        cur_step.env_masks = np.array(e_masks)
        self._response_token_count += len(e_tokens)

    if step_timed_out:
      self.agent.trajectory.status = agent_types.TrajectoryStatus.TIMEOUT
      logging.warning("Episode timed out after %d seconds.", self.timeout)
      self.agent.get_current_step().done = True
      return True

    return done

  async def _append_final_reward(self):
    # [FUNCTION DOC]
    # HELPER: Coroutine. If there is a last step and final_reward_fn is callable, run
    # final_reward_fn() with no arguments in the executor with timing, add timings to
    # reward_time and add the result to the last step's reward; otherwise log and
    # skip.
    # KNOWN ISSUE: the docstring and constructor docs say the function takes (task,
    # response), but it is called with no arguments.
    """Compute and add final reward to the last step of the episode.

    Applies the final reward function (if provided) to the episode's
    final response and adds it to the last step's reward. This enables
    additional reward signals based on overall episode performance.
    """
    last_step = self.agent.get_current_step()
    if (
        last_step is None
        or self.final_reward_fn is None
        or not callable(self.final_reward_fn)
    ):
      # Skip reward computation in trajectory collection if no reward function
      # is provided or no step is taken.
      logging.debug("%s Final reward function is skipped", self._debug_prefix)
      return
    final_reward, wall_time, cpu_time = await self._run_with_timing(
        self.final_reward_fn
    )

    self.reward_time["reward_latency"] += wall_time
    self.reward_time["reward_cpu_time"] += cpu_time
    last_step.reward += final_reward
    logging.debug(
        "%s Final reward computed: %s", self._debug_prefix, final_reward
    )

  def compute_trajectory_reward(self):
    # [FUNCTION DOC]
    # Set trajectory.reward to the float sum of all step rewards (0.0 for no steps)
    # and return the trajectory.
    # Note: the falsy-trajectory guard never triggers for a dataclass instance.
    """Computes and stores the total reward for the trajectory.

    The trajectory reward is the undiscounted sum of rewards from all steps and
    is stored in `trajectory.reward`.

    Returns:
        The updated trajectory with the `reward` attribute populated.
    """
    trajectory = self.agent.trajectory
    if not trajectory:
      return None
    trajectory.reward = float(
        np.sum(np.array([s.reward for s in trajectory.steps]))
    )
    return trajectory

  def compute_mc_reward(self):
    # [FUNCTION DOC]
    # Compute discounted returns backwards: starting from 0, for each step from last
    # to first, return = step reward + gamma * next return; stored in step.mc_return.
    """Compute Monte Carlo rewards for all steps in the trajectory.

    Calculates discounted rewards working backwards from the final step.
    Each step's Monte Carlo reward (return) is its immediate reward plus the
    discounted reward of subsequent steps. The result is stored in
    `step.mc_return`.
    """
    trajectory = self.agent.trajectory
    g = 0.0
    for step in reversed(trajectory.steps):
      g = step.reward + self.gamma * g
      step.mc_return = g

  async def _close(self):
    # [FUNCTION DOC]
    # HELPER: Coroutine. Log timing counters at debug level, then run env.close in
    # the default executor with a 150 s timeout; on timeout log an error (the worker
    # thread may leak). Other exceptions from close propagate.
    """Clean up resources by closing the environment.

    Ensures proper cleanup of environment resources such as network
    connections, file handles, or external processes.
    """
    logging.debug("%s Closing environment.", self._debug_prefix)
    for k, v in self.env_time.items():
      logging.debug("%s k=%s v=%s", self._debug_prefix, k, v)
    for k, v in self.reward_time.items():
      logging.debug("%s k=%s v=%s", self._debug_prefix, k, v)

    try:
      await asyncio.wait_for(
          asyncio.get_event_loop().run_in_executor(
              None, self.env.close
          ),
          timeout=150.0,
      )
    except asyncio.TimeoutError:
      logging.error(
          "%s env.close() timed out after 150s — executor thread may be"
          " leaked. This will starve the thread pool over time.",
          self._debug_prefix,
      )
    logging.debug("%s Environment closed.", self._debug_prefix)
