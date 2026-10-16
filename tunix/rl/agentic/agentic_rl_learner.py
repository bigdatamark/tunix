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
# Abstract base for agentic RL learners (e.g. GRPO) that train a policy from
# multi-turn agent/environment rollouts collected asynchronously.
# Architecture: the learner owns a dedicated asyncio event loop running forever on
# a daemon thread (default executor sized max_concurrency + 1). Training
# (train) runs on the caller's thread: it feeds batches of prompts into a
# thread-safe prompt queue; a producer coroutine on the loop turns each prompt
# into num_generations (agent, env) pairs, runs them through a RolloutOrchestrator
# / TrajectoryCollectEngine in "Token" mode, groups finished trajectories by
# group_id, converts groups into TrainExamples via the subclass's
# _process_results (rewards + advantages), and pushes them into a data queue
# consumed by the training loop, which updates the actor (and critic if present),
# periodically evaluates, and at each full-batch boundary syncs weights to the
# rollout model under a RolloutSyncLock and enqueues the next prompt batch.
# off_policy_steps + 1 batches are prefetched, allowing rollouts to run ahead.
# Also defines: type aliases ArrayLike, TrainingInputT (dict of str to list of str
# or array), RewardFn, MetricFn; TrainExample (common.TrainExample plus an
# optional policy_version array); AgenticRLConfig; type var TConfig.
# Imports reward_manager and the agentic rewards module only for their
# registration side effects.
"""Base class for Agentic RL Learners."""

from __future__ import annotations
import abc
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import contextlib
import copy
import dataclasses
import itertools
import queue
import threading
from typing import Any, AsyncIterator, Callable, Dict, Generic, Iterable, Iterator, List, Sequence, Type, TypeVar, Optional, Set

from absl import logging
import flax
import jax
from jax import typing
import jax.numpy as jnp
import numpy as np
from tunix.rl import algorithm_config as algo_config_lib
from tunix.rl import common
from tunix.perf.experimental import constants as perf_constants
from tunix.rl import function_registry
from tunix.rl import reward_manager  # pylint: disable=unused-import
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout
from tunix.rl import utils as rl_utils
from tunix.rl.agentic import utils as agentic_utils
from tunix.rl.agentic.agents import base_agent
from tunix.rl.agentic.agents import model_agent
from tunix.rl.agentic.environments import base_environment
from tunix.rl.agentic.environments import task_environment
from tunix.rl.agentic.pipeline import rollout_orchestrator
from tunix.rl.agentic.rewards import reward  # pylint: disable=unused-import
from tunix.rl.agentic.trajectory import trajectory_collect_engine
from tunix.rl.queue import data_queue as queue_lib
from tunix.sft import utils as sft_utils

ArrayLike = typing.ArrayLike
TrainingInputT = Dict[str, List[str] | ArrayLike]
RewardFn = Callable[..., List[float]]
MetricFn = Callable[..., rl_cluster_lib.MetricsT]


@flax.struct.dataclass(frozen=True)
class TrainExample(common.TrainExample):
  # [CLASS DOC]
  # Frozen flax struct dataclass extending common.TrainExample with policy_version
  # (numpy array or None), the policy version(s) that generated the sample, used for
  # off-policy filtering.
  policy_version: np.ndarray | None = None


@dataclasses.dataclass(slots=True, kw_only=True)
class AgenticRLConfig(algo_config_lib.AlgorithmConfig):
  # [CLASS DOC]
  # Slotted keyword-only dataclass extending AlgorithmConfig. Fields (defaults):
  # system_prompt (""), reward_manager name ("agentic-sequence-level", looked up in
  # the function registry), max_response_length (1024 tokens per episode; must
  # equal the rollout config max_tokens_to_generate), max_concurrency (32
  # concurrent episodes), off_policy_steps (0), num_generations (1 samples per
  # prompt, also the group size), num_iterations (1 optimization passes per
  # batch), episode_timeout (1800.0 s), filter_statuses (set of TrajectoryStatus or
  # None for the engine default), overlong_filter (False), use_rollout_logps (True:
  # use rollout logprobs as old logprobs; requires return_logprobs in rollout
  # config).
  """Base configuration for Agentic RL algorithms.

  Parameters:
    system_prompt: System prompt for the agent.
    max_response_length: Maximum number of tokens for each episode.
    max_concurrency: Maximum number of concurrent requests to the rollout
      engines.
    off_policy_steps: Number of off-policy steps can be accepted before a
      policy update.
    num_generations: Number of samples per prompt.
    num_iterations: Number of iterations per batch.
    episode_timeout: Timeout for each episode in seconds.
  """

  system_prompt: str = ""
  # TODO(tsbao): we need to update the scripts that uses max_tokens_to_generate
  # once this new agentic_rl_learner is used.
  reward_manager: str = "agentic-sequence-level"
  max_response_length: int = 1024
  max_concurrency: int = 32
  off_policy_steps: int = 0
  num_generations: int = 1
  num_iterations: int = 1
  episode_timeout: float = 1800.0
  filter_statuses: Optional[Set] = None
  overlong_filter: bool = False
  use_rollout_logps: bool = True


TConfig = TypeVar("TConfig", bound=AgenticRLConfig)


class AgenticRLLearner(abc.ABC, Generic[TConfig]):
  # [CLASS DOC]
  # Abstract generic learner parameterized by its config type. Subclasses must
  # implement _process_results. Holds the RL cluster, reward manager, agent/env
  # classes and kwargs, a background event loop, rollout sync lock, batch-size
  # bookkeeping, policy_version counter, eval bookkeeping, and reward windows used
  # for per-step log summaries (guarded by a threading lock).
  """Base class for Agentic RL Learners using asynchronous rollouts."""

  class _AsyncQueueIterator:
    # [CLASS DOC]
    # HELPER: Async iterator adapting a blocking queue.Queue: each __anext__ runs the
    # blocking get in the loop's default executor and stops iteration on a None
    # sentinel.
    """Async iterator that yields items from a sync queue."""

    def __init__(
        self,
        q: queue.Queue[TrainingInputT | None],
        loop: asyncio.AbstractEventLoop,
    ):
      # [FUNCTION DOC]
      # HELPER: Store the queue and event loop.
      self.q = q
      self.loop = loop

    def __aiter__(self):
      # [FUNCTION DOC]
      # HELPER: Return self (the object is its own async iterator).
      return self

    async def __anext__(self):
      # [FUNCTION DOC]
      # HELPER: Coroutine. Await queue.get in the executor; raise StopAsyncIteration if
      # the item is None, otherwise return it. Occupies one executor thread while
      # blocked.
      item = await self.loop.run_in_executor(None, self.q.get)
      if item is None:
        raise StopAsyncIteration
      return item

  def __init__(
      self,
      rl_cluster: rl_cluster_lib.RLCluster,
      algo_config: TConfig,
      reward_fns: RewardFn | List[RewardFn] | None = None,
      chat_parser: Any | None = None,
      metric_fns: Sequence[MetricFn] | None = None,
      agent_class: Type[
          base_agent.ConversationAgentBase
      ] = model_agent.ModelAgent,
      agent_kwargs: Dict[str, Any] | None = None,
      env_class: Type[
          base_environment.BaseTaskEnv
      ] = task_environment.TaskEnvironment,
      env_kwargs: Dict[str, Any] | None = None,
  ):
    # [FUNCTION DOC]
    # Construct the learner.
    # Inputs: rl_cluster (RLCluster with actor trainer, rollout, tokenizer, configs),
    # algo_config, reward_fns (single function, list or None, handed to the reward
    # manager), chat_parser (optional BaseChatTemplateParser; when None the cluster
    # applies the tokenizer's chat template), metric_fns (stored, default empty),
    # agent_class (default ModelAgent) and agent_kwargs, env_class (default
    # TaskEnvironment) and env_kwargs.
    # Behavior: validates rollout configs; builds the reward manager from the
    # registry by algo_config.reward_manager; marks the actor trainer (and critic
    # trainer if the attribute exists) as externally managed; sets
    # rl_cluster.global_steps from the actor trainer's restored global step; copies
    # iter_steps; initializes eval counters (_last_eval_train_step -1 so eval runs at
    # most once per optimizer step under gradient accumulation); decides
    # should_sync_weights (actor and rollout models not sharing weights) and
    # can_enable_async_rollout (actor and rollout meshes differ); reads rollout and
    # compute-logps micro batch sizes (logps defaulting to 1); logs HBM usage; sets
    # chat_parser, tokenizer, policy_version (= global steps), a RolloutSyncLock,
    # _full_batch_size 0, _process_in_consumer False; starts a daemon thread that
    # creates/gets an event loop, installs a ThreadPoolExecutor of max_concurrency + 1
    # workers as its default executor, hands the loop back through a queue and runs it
    # forever; records the global-step start time; creates reward windows and lock.
    # Exceptions: ValueError from config validation; registry lookup errors.
    # Side effects: mutates rl_cluster state; starts a thread that is never stopped.
    """Initializes the `AgenticRLLearner`.

    Args:
      rl_cluster: RL cluster containing actor, reference and reward models.
      algo_config: Configuration object.
      reward_fns: Reward functions.
      chat_parser: A parser to handle chat message formatting.
      metric_fns: A sequence of callables that compute metrics for the
        completions. Each callable should accept ``prompts``, ``completions``,
        ``rewards``, ``advantages`` and optional keyword arguments, and return
        a dictionary of metric names to tuples of
        ``(metric_value, aggregation_fn)``:

           >>> def metric_fn(
           ...     prompts, completions, rewards, advantages, **kargs
           ... ):
           ...     return {
           ...       # ...
           ...       "prompt_min_len": (min(len(p) for p in prompts), np.min),
           ...       # ... }
      agent_class: User defined agent class.
      agent_kwargs: Keyword arguments for the agent class.
      env_class: User defined environment class.
      env_kwargs: Keyword arguments for the environment class.
    """
    self.rl_cluster = rl_cluster
    self.algo_config = algo_config
    self._validate_rollout_config()
    reward_manager_fn = function_registry.get_reward_manager(
        algo_config.reward_manager
    )
    self.reward_manager = reward_manager_fn(
        reward_fns=reward_fns,
        algo_config=algo_config,
    )
    self.metric_fns = metric_fns or []
    self.rl_cluster.actor_trainer.is_managed_externally = True
    if hasattr(self.rl_cluster, "critic_trainer"):
      self.rl_cluster.critic_trainer.is_managed_externally = True

    self.agent_class = agent_class
    self.agent_kwargs = agent_kwargs or {}
    self.env_class = env_class
    self.env_kwargs = env_kwargs or {}

    self._training_config = self.rl_cluster.cluster_config.training_config

    self.rl_cluster.global_steps = (
        self.rl_cluster.actor_trainer.restored_global_step()
    )
    # Current iter steps for micro-batch based training.
    self._iter_steps = self.rl_cluster.actor_trainer.iter_steps
    self._eval_iter_steps = 0
    # Tracks the last train_step value at which evaluation was run. The
    # optimizer is wrapped in ``optax.MultiSteps(grad_accum_steps)``, which
    # keeps ``actor_trainer.train_steps`` constant for ``grad_accum_steps``
    # consecutive micro-iterations. Without this guard, the
    # ``train_steps % eval_every_n_steps == 0`` check would fire at every
    # micro-iteration during an eval boundary, causing the full evaluation
    # rollout to be replayed ``grad_accum_steps`` times for the same step.
    self._last_eval_train_step = -1

    # Sync weights if the actor model and rollout model are not sharing weights.
    self.should_sync_weights = not (
        rl_utils.is_sharing_weights(
            self.rl_cluster.actor_trainer.model,
            self.rl_cluster.rollout.model(),
        )
    )

    # Enable async rollout if trainer and rollout are not on the same mesh.
    # If they do, then doesn't make sense for the interleave because they will
    # have resource contention.
    self.can_enable_async_rollout = (
        self.rl_cluster.cluster_config.role_to_mesh[rl_cluster_lib.Role.ACTOR]
        != self.rl_cluster.cluster_config.role_to_mesh[
            rl_cluster_lib.Role.ROLLOUT
        ]
    )

    self._rollout_micro_batch_size = (
        self._training_config.rollout_micro_batch_size
    )
    self._compute_logps_micro_batch_size = (
        self._training_config.compute_logps_micro_batch_size or 1
    )
    sft_utils.show_hbm_usage(title="AgenticRLLearner init")

    self.chat_parser = chat_parser
    self.tokenizer = rl_cluster.tokenizer
    self.policy_version = self.rl_cluster.global_steps
    self._rollout_sync_lock = agentic_utils.RolloutSyncLock()
    self._full_batch_size = 0
    self._process_in_consumer: bool = False

    loop_queue = queue.Queue()

    def run_loop_forever():
      # [FUNCTION DOC]
      # HELPER: Thread target. Gets or creates this thread's event loop, sets a
      # ThreadPoolExecutor with max_concurrency + 1 workers as its default executor,
      # hands the loop to the constructor via a queue, then runs the loop forever
      # (never returns; the daemon thread dies with the process).
      loop = agentic_utils.get_or_create_loop()
      loop.set_default_executor(
          ThreadPoolExecutor(max_workers=algo_config.max_concurrency + 1)
      )
      loop_queue.put(loop)
      loop.run_forever()

    loop_thread = threading.Thread(target=run_loop_forever, daemon=True)
    loop_thread.start()
    self.loop = loop_queue.get()
    self._global_step_start_time = time.time()

    # Per-step reward accumulators populated inside ``_compute_rewards``.
    # Drained at the global-step boundary to emit a one-line per-step
    # summary that mirrors what an external metric logger would show.
    # Each bin keeps at most ``full_batch_size``-worth of recent values
    # so a producer that races one batch ahead of the consumer does not
    # double-count.
    self._train_rewards_window: List[float] = []
    self._eval_rewards_window: List[float] = []
    self._rewards_window_lock = threading.Lock()

  def _validate_rollout_config(self):
    # [FUNCTION DOC]
    # HELPER: Check every rollout config (a single config treated as "train", or each
    # value of a mode-keyed dict) against the algo config.
    # Exceptions: ValueError if max_tokens_to_generate differs from
    # max_response_length; if use_rollout_logps is set but return_logprobs is false;
    # or if the cluster's rollout engine is "vllm" without rollout_vllm_server_mode.
    """Validates that the rollout config is properly aligned with the algo config."""
    rollout_config = self.rl_cluster.cluster_config.rollout_config
    if not isinstance(rollout_config, dict):
      configs_to_check = {"train": rollout_config}
    else:
      configs_to_check = rollout_config

    for mode, config in configs_to_check.items():
      if config.max_tokens_to_generate != self.algo_config.max_response_length:
        raise ValueError(
            f"RolloutConfig ({mode}) max_tokens_to_generate "
            f"({config.max_tokens_to_generate}) must match AgenticRLConfig "
            f"max_response_length ({self.algo_config.max_response_length}). "
            "Please align these configurations before initializing RLCluster."
        )
      if self.algo_config.use_rollout_logps and not config.return_logprobs:
        raise ValueError(
            f"RolloutConfig ({mode}) must have return_logprobs=True for "
            "AgenticRLLearner when use_rollout_logps=True. Please set this "
            "before initializing RLCluster."
        )
      if (
          self.rl_cluster.cluster_config.rollout_engine == "vllm"
          and not config.rollout_vllm_server_mode
      ):
        raise ValueError(
            f"RolloutConfig ({mode}) must have rollout_vllm_server_mode set to "
            "True for AgenticRLLearner if using vLLM engine. Please set this "
            "before initializing RLCluster."
        )
  def _compute_rewards(
      self,
      prompts: List[str],
      completions: List[str],
      mode: rl_cluster_lib.Mode,
      expected_step: int | None = None,
      **kwargs,
  ) -> np.ndarray:
    # [FUNCTION DOC]
    # HELPER: Compute rewards via the reward manager and record them.
    # Inputs: prompts, completions, mode (TRAIN/EVAL), expected_step (default 0),
    # extra kwargs forwarded to the reward manager.
    # Behavior: adds kwargs["mode"] = str(mode) (ValueError if "mode" was already
    # present); calls the reward manager with prompts, completions and kwargs; buffers
    # its "log_metrics" asynchronously at expected_step; appends the rewards to the
    # train or eval window under the lock, trimming the train window to the most
    # recent full_batch_size * num_generations values when full batch size is known.
    # Outputs: the reward manager's "rewards" value (as returned, not converted).
    # Exceptions: ValueError as above; KeyError if the manager result lacks keys.
    # KNOWN ISSUE: docstring claims a JAX array return and RuntimeError cases that the
    # code does not implement.
    """Computes the rewards for completions using the provided reward functions.

    Args:
      prompts: A list of input prompts.
      completions: A list of generated text completions.
      mode: The mode to use for logging metrics.
      expected_step: The expected training step.
      **kwargs: Additional keyword arguments passed to the reward functions.

    Returns:
      A JAX array (shape `[num_prompts]`) of scalar rewards for each
      prompt-completion pair. The rewards are the sum across all the provided
      reward functions.

    Raises:
        RuntimeError: If 'r' reward is None, indicating a failure to obtain the
        result, or if the length of 'r' reward does not match the length of
        'prompts'.
    """
    if "mode" in kwargs:
      raise ValueError(f"kwargs already contains mode as a key: {kwargs}")
    kwargs["mode"] = str(mode)

    rewards_info = self.reward_manager(
        prompts=prompts,
        completions=completions,
        **kwargs,
    )

    # Pass the expected_step explicitly because it is calculated based on
    # the batch index (predicted step) to align metrics with the correct
    # training step in the asynchronous execution.
    expected_step = 0 if expected_step is None else expected_step
    self.rl_cluster.buffer_metrics_async(
        rewards_info["log_metrics"], mode=mode, step=expected_step
    )

    rewards_array = np.asarray(rewards_info["rewards"])
    with self._rewards_window_lock:
      target = (
          self._train_rewards_window
          if mode == rl_cluster_lib.Mode.TRAIN
          else self._eval_rewards_window
      )
      target.extend(rewards_array.tolist())
      # Cap train window at full_batch_size * num_generations (one full step's
      # worth of per-sequence rewards) to bound the producer-vs-consumer
      # race: the producer can race up to ``off_policy_steps + 1`` batches
      # ahead, so without a cap the window would over-count next-step rewards
      # at the current step's boundary.
      if mode == rl_cluster_lib.Mode.TRAIN and self._full_batch_size > 0:
        cap = self._full_batch_size * self.algo_config.num_generations
        excess = len(target) - cap
        if excess > 0:
          del target[:excess]

    return rewards_info["rewards"]

  def _create_micro_batch_iterator(
      self,
      full_batch_iterator: Iterator[TrainingInputT],
      micro_batch_size: int,
  ) -> Iterator[TrainingInputT]:
    # [FUNCTION DOC]
    # HELPER: Generator re-batching dict batches into micro-batches.
    # Inputs: iterator of dict batches; micro_batch_size.
    # Behavior: keeps a per-key list buffer. Each incoming batch's values are added:
    # numpy/JAX arrays are flattened element-wise, lists/tuples are extended,
    # scalars/other values appended as one item. While the buffer length (measured
    # by the first key) is at least micro_batch_size, yields a dict whose values are
    # numpy arrays of the first micro_batch_size items per key, removing them.
    # Leftover items smaller than one micro-batch are dropped at the end.
    # KNOWN ISSUE: multi-dimensional arrays are flattened, so per-example rows of 2D
    # fields are split into scalars and misaligned with other keys; keys of unequal
    # lengths are not detected.
    """Re-batches large inputs into an iterator of micro-batches.

    Args:
      full_batch_iterator: Iterator yielding large `TrainingInputT` batches.
      micro_batch_size: The desired size of the micro-batches.

    Yields:
      `TrainingInputT` dicts, each with `micro_batch_size` samples.
    """
    buffer = {}

    def get_buffer_len(buf: dict[str, list[Any]]) -> int:
      # [FUNCTION DOC]
      # HELPER: Length of the first value list in the buffer, or 0 if empty.
      if not buf:
        return 0
      return len(next(iter(buf.values())))

    for large_batch in full_batch_iterator:
      for key, values in large_batch.items():
        if key not in buffer:
          buffer[key] = []

        if isinstance(values, (np.ndarray, jax.Array)):
          buffer[key].extend(list(values.flatten()))
        elif isinstance(values, (list, tuple)):
          buffer[key].extend(values)
        else:
          buffer[key].append(values)

      while get_buffer_len(buffer) >= micro_batch_size:
        micro_batch = {}
        for key in buffer:
          micro_batch_list_slice = buffer[key][:micro_batch_size]
          micro_batch[key] = np.array(micro_batch_list_slice)
          buffer[key] = buffer[key][micro_batch_size:]

        yield micro_batch

  def _create_agent_env_pair(
      self, single_example: TrainingInputT, group_id: int, pair_index: int
  ) -> tuple[base_agent.ConversationAgentBase, base_environment.BaseTaskEnv]:
    # [FUNCTION DOC]
    # HELPER: Build one (agent, env) pair for an example.
    # Inputs: single_example (dict for one prompt), group_id, pair_index.
    # Behavior: instantiates agent_class with system_prompt from the config, overridden
    # by any agent_kwargs entries; asserts env_kwargs contains neither "group_id" nor
    # "pair_index"; instantiates env_class with the example as first positional
    # argument plus group_id, pair_index and env_kwargs as keywords (ending up in the
    # env's extra_kwargs for BaseTaskEnv subclasses).
    # Outputs: (agent, env). Exceptions: AssertionError; constructor errors.
    """Constructs an (agent, environment) pair for a single input sample.

    This is used to set up a rollout for one generation within a group.

    Args:
      single_example: A training input containing a single prompt.
      group_id: An identifier for group generations from the same original
        prompt.
      pair_index: The index of the pair within the group.

    Returns:
      A tuple of agent and environment.
    """

    agent = self.agent_class(
        **{"system_prompt": self.algo_config.system_prompt, **self.agent_kwargs}
    )  # if agent_kwargs contains "system_prompt", it will be honored.

    assert "group_id" not in self.env_kwargs
    assert "pair_index" not in self.env_kwargs
    env = self.env_class(
        single_example,
        **{"group_id": group_id, "pair_index": pair_index, **self.env_kwargs},
    )

    return agent, env

  def _model_call(
      self,
      chat_lists: List[Dict[str, str]],
      env: Any = None,
      max_generation_steps: int | None = None,
  ) -> base_rollout.RolloutOutput:
    # [FUNCTION DOC]
    # HELPER: Model-call function given to the trajectory engine.
    # Inputs: chat_lists (message list), env (optional), max_generation_steps.
    # Behavior: if env is given, writes the current policy_version into env.task
    # (mutating the task). If a chat_parser exists, renders the messages into a single
    # prompt string as a first message with generation prompt. Builds perf tags from
    # env.extra_kwargs: group id, step (group_id // full batch size, when known) and
    # pair index. Calls rl_cluster.generate with the prompts, apply_chat_template
    # true only when no chat parser is set, mode TRAIN (also used for eval
    # rollouts), the tags and max_generation_steps.
    # Outputs: RolloutOutput. Exceptions: propagates generation errors.
    # KNOWN ISSUE: when a chat_parser is used, a single string (not a list) is passed
    # as prompts; correctness relies on rl_cluster.generate accepting it.
    """Calls model generation."""
    if env:
      env.task["policy_version"] = self.policy_version

    if self.chat_parser:
      chat_lists = self.chat_parser.parse(
          messages=chat_lists,
          add_generation_prompt=True,
          is_first_msg=True,  # no op if system msg is populated in reset
      )
    tags = {}
    if env and hasattr(env, "extra_kwargs"):
      if "group_id" in env.extra_kwargs:
        tags[perf_constants.GROUP_ID] = env.extra_kwargs["group_id"]
        if self._full_batch_size > 0:
          tags[perf_constants.STEP] = (
              env.extra_kwargs["group_id"] // self._full_batch_size
          )
      if "pair_index" in env.extra_kwargs:
        tags[perf_constants.PAIR_INDEX] = env.extra_kwargs["pair_index"]

    result = self.rl_cluster.generate(
        prompts=chat_lists,
        apply_chat_template=False if self.chat_parser else True,
        mode=rl_cluster_lib.Mode.TRAIN,
        trace_tags=tags,
        max_generation_steps=max_generation_steps,
    )

    return result

  def _build_orchestrator(self) -> rollout_orchestrator.RolloutOrchestrator:
    # [FUNCTION DOC]
    # HELPER: Create a RolloutOrchestrator using TrajectoryCollectEngine with
    # engine kwargs: model_call = _model_call, tokenizer, chat_parser, timeout =
    # episode_timeout, max_response_length, overlong_filter, filter_statuses,
    # perf_v2 = cluster tracer; max_concurrency from config; shared rollout sync lock.
    """Builds and configures a RolloutOrchestrator for parallel rollouts."""
    engine_kwargs = dict(
        model_call=self._model_call,
        tokenizer=self.tokenizer,
        chat_parser=self.chat_parser,
        timeout=self.algo_config.episode_timeout,
        max_response_length=self.algo_config.max_response_length,
        overlong_filter=self.algo_config.overlong_filter,
        filter_statuses=self.algo_config.filter_statuses,
        perf_v2=self.rl_cluster.perf_v2,
    )
    return rollout_orchestrator.RolloutOrchestrator(
        engine_cls=trajectory_collect_engine.TrajectoryCollectEngine,
        engine_kwargs=engine_kwargs,
        max_concurrency=self.algo_config.max_concurrency,
        rollout_sync_lock=self._rollout_sync_lock,
    )

  async def _orchestrator_producer(
      self,
      orchestrator: rollout_orchestrator.RolloutOrchestrator,
      prompt_iterator: Iterable[TrainingInputT] | AsyncIterator[TrainingInputT],
      num_generations: int = 1,
      collect_mode: str = "Token",
  ):
    # [FUNCTION DOC]
    # HELPER: Async generator yielding groups of trajectory results.
    # Inputs: orchestrator, prompt_iterator (sync iterable or async iterator of
    # single-example dicts), num_generations (pairs per prompt), collect_mode
    # (default "Token").
    # Behavior: defines an async pair stream that starts group_id at
    # global_steps * full_batch_size and, for each example, creates num_generations
    # (agent, env) pairs concurrently in executor threads (each with a deep copy of
    # the example), yields them, then increments group_id. Starts
    # run_producers_from_stream as a background task with group size
    # config.num_generations and grouping by env.extra_kwargs["group_id"], yields
    # control once, then iterates orchestrator.yield_batches with batch size
    # num_generations (from the config), yielding each non-empty group (a list of
    # TrajectoryItems whose traj is the "Token"-mode dict).
    # On exit (normal, GeneratorExit or cancellation) cancels the producer task if
    # still running and schedules a fire-and-forget task awaiting its cancellation.
    # KNOWN ISSUE: the docstring calls num_generations "episodes per pair";
    # group_size and batch size use config.num_generations while pairs per prompt use
    # the parameter, so differing values would break grouping.
    """Generates trajectory groups using the orchestrator pattern.

    Args:
      orchestrator: The RolloutOrchestrator instance to use.
      prompt_iterator: An iterable yielding single `TrainingInputT` examples.
      num_generations: The number of episodes to run per agent-environment pair.
      collect_mode: The mode for trajectory collection (e.g., "Token").

    Yields:
      A list of trajectories for a group.
    """
    is_async_iterator = hasattr(prompt_iterator, "__aiter__")

    async def pairs_stream_generator():
      # [FUNCTION DOC]
      # HELPER: Async generator of (agent, env) pairs with per-prompt group ids; see
      # the enclosing function.
      """Yield (agent, env) pairs with unique group_id per original prompt."""
      # TODO (tsbao): fix the group id when we can resume from mid global step
      # with mini-batch.
      group_id = self.rl_cluster.global_steps * self._full_batch_size
      if is_async_iterator:
        async for single_example in prompt_iterator:
          # Create agent-env pairs in parallel for a group to handle potential
          # cold start latency on env creation.
          agent_env_pairs = await asyncio.gather(*[
              self.loop.run_in_executor(
                  None,
                  self._create_agent_env_pair,
                  copy.deepcopy(single_example),
                  group_id,
                  pair_index,
              )
              for pair_index in range(num_generations)
          ])
          for agent, env in agent_env_pairs:
            yield agent, env
          group_id += 1
      else:
        for single_example in prompt_iterator:
          agent_env_pairs = await asyncio.gather(*[
              self.loop.run_in_executor(
                  None,
                  self._create_agent_env_pair,
                  copy.deepcopy(single_example),
                  group_id,
                  pair_index,
              )
              for pair_index in range(num_generations)
          ])
          for agent, env in agent_env_pairs:
            yield agent, env
          group_id += 1

    # Start producers in the background.
    producer_task = asyncio.create_task(
        orchestrator.run_producers_from_stream(
            pairs_stream=pairs_stream_generator(),
            group_size=self.algo_config.num_generations,
            group_key_fn=lambda i, env, traj: env.extra_kwargs["group_id"],
            collect_mode=collect_mode,
        )
    )

    # Let the producer start and initialize its manager before consuming.
    await asyncio.sleep(0)

    # Consume full groups and yield them with their original input.
    async_generator = orchestrator.yield_batches(
        batch_size=self.algo_config.num_generations
    )
    try:
      async with contextlib.aclosing(async_generator) as stream:
        async for group in stream:
          if group:
            # Retrieve the original input embedded in the task.
            yield group
    except (GeneratorExit, asyncio.CancelledError):
      # This is the normal shutdown path for a generator.
      return
    finally:
      # Ensure the background producer task is cancelled and cleaned up.
      if not producer_task.done():
        producer_task.cancel()

        async def await_cancellation():
          # [FUNCTION DOC]
          # HELPER: Await the cancelled producer task, suppressing CancelledError.
          with contextlib.suppress(asyncio.CancelledError):
            await producer_task

        cancellation_task = asyncio.create_task(await_cancellation())
        del cancellation_task

  def _batch_to_train_example(
      self,
      batch_results: list[Any],
      mode: rl_cluster_lib.Mode,
  ) -> List[TrainExample]:
    # [FUNCTION DOC]
    # HELPER: Convert a list of TrajectoryItems into TrainExamples.
    # Behavior: expected step is first item's group_id // full batch size in TRAIN
    # mode, or the cluster's global_steps otherwise; delegates to _process_results.
    # Exceptions: IndexError on an empty list; ZeroDivisionError if full batch size
    # is 0 in TRAIN mode.
    """Converts a group of trajectories into a list of `TrainExample`s.

    Args:
      batch_results: A list of trajectories from the same generation group.
      mode: The current mode (TRAIN or EVAL).

    Returns:
      A list of `TrainExample` instances, ready for training.
    """
    # Create a merged training_input where each field from the original input
    # is repeated G times to align with the G completions.
    if mode == rl_cluster_lib.Mode.TRAIN:
      expected_step = batch_results[0].group_id // self._full_batch_size
    else:
      expected_step = self.rl_cluster.global_steps

    return self._process_results(
        trajectories=batch_results,
        mode=mode,
        expected_step=expected_step,
    )

  @abc.abstractmethod
  def _process_results(
      self,
      trajectories: List[Any],
      mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
      expected_step: int | None = None,
  ) -> List[TrainExample]:
    # [FUNCTION DOC]
    # Abstract. Turn trajectories of a group (or several groups) into TrainExamples:
    # compute rewards, advantages and model inputs. Inputs: trajectories, mode
    # (default TRAIN), expected_step. Base returns None.
    """Processes generation results, computes rewards and advantages."""
    pass

  def _generate_and_compute_advantage(
      self,
      training_input: TrainingInputT,
      mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
  ) -> TrainExample:
    # [FUNCTION DOC]
    # Not used by agentic learners; always raises NotImplementedError.
    """Unused in AgenticRLLearner."""
    raise NotImplementedError(
        "_generate_and_compute_advantage is not used in AgenticRLLearner"
    )

  def _num_iterations(self) -> int:
    # [FUNCTION DOC]
    # HELPER: Return config.num_iterations.
    """Returns the number of iterations per batch."""
    return self.algo_config.num_iterations

  def _num_generations(self) -> int:
    # [FUNCTION DOC]
    # HELPER: Return config.num_generations.
    """Returns the number of generations per prompt."""
    return self.algo_config.num_generations

  async def _producer(
      self,
      orchestrator,
      prompt_queue: queue.Queue[TrainingInputT | None],
      train_data_queue,
  ):
    # [FUNCTION DOC]
    # HELPER: Coroutine running on the background loop; produces training data.
    # Inputs: orchestrator, prompt_queue (blocking queue of full prompt batches,
    # None = end), train_data_queue (data queue for the consumer).
    # Behavior: wraps prompt_queue as an async iterator and splits each batch into
    # single-example micro-batches (size 1). Iterates _orchestrator_producer in
    # "Token" mode with config num_generations. For each group: if
    # _process_in_consumer, puts the raw group into the data queue; otherwise converts
    # it to TrainExamples and puts every example num_iterations times (repeated passes
    # in order). Non-RuntimeError exceptions are logged with traceback and re-raised.
    # Finally always puts None into train_data_queue (end signal) and None into
    # prompt_queue (to unblock waiters).
    """Produces training examples from prompts in the dataset_iterator."""
    loop = asyncio.get_running_loop()
    async_queue_iter = self._AsyncQueueIterator(prompt_queue, loop)

    async def _iterate_micro_batches():
      # [FUNCTION DOC]
      # HELPER: Async generator yielding single-example dicts from queued batches.
      async for item in async_queue_iter:
        for prompt in self._create_micro_batch_iterator(iter([item]), 1):
          yield prompt

    prompt_iterator = _iterate_micro_batches()
    try:
      async for batch in self._orchestrator_producer(
          orchestrator=orchestrator,
          prompt_iterator=prompt_iterator,
          num_generations=self.algo_config.num_generations,
          collect_mode="Token",
      ):
        try:
          if self._process_in_consumer:
            # Put raw batch (list of trajectories) into queue.
            # We put it once, and consumer will handle iterations.
            train_data_queue.put(batch)
          else:
            train_examples = self._batch_to_train_example(
                batch_results=batch,
                mode=rl_cluster_lib.Mode.TRAIN,
            )
            for _ in range(self._num_iterations()):
              for train_example in train_examples:
                train_data_queue.put(train_example)
        except Exception as e:
          if not isinstance(e, RuntimeError):
            logging.exception(
                "Exception in _producer while processing batch: %s", e
            )
          raise
    finally:
      # Signal production is complete for this batch, even if errors occurred.
      train_data_queue.put(None)
      # Ensure that any background threads waiting on the prompt queue are
      # unblocked.
      prompt_queue.put(None)

  def _data_consumer_batch_generator(
      self, queue: queue_lib.AbstractDataQueue, batch_size: int
  ):
    # [FUNCTION DOC]
    # HELPER: Generator grouping queue items into lists of up to batch_size, reading
    # with blocking get until a None sentinel; the final list may be shorter; returns
    # when no items remain. The parameter name shadows the queue module locally.
    """Yields micro-batches from a queue until a None is received."""
    item_iterator = iter(lambda: queue.get(block=True), None)
    while True:
      batch = list(itertools.islice(item_iterator, batch_size))
      if not batch:
        return  # The iterator is exhausted.
      yield batch

  def train(
      self,
      train_dataset: Iterable[TrainingInputT],
      eval_dataset: Iterable[TrainingInputT] | None = None,
      skip_jit: bool = False,
  ) -> None:
    # [FUNCTION DOC]
    # Main training loop (blocking; runs on the caller's thread).
    # Inputs: train_dataset (iterable of full batches, dicts of equal-length
    # columns), optional eval_dataset, skip_jit (forwarded to trainer updates).
    # Behavior:
    # 1. If resuming (global_steps > 0), skip that many batches; if the dataset runs
    # out, log, close the cluster and return. If the dataset is empty, likewise.
    # 2. full_batch_size = length of the first column of the first batch; derive
    # mini_batch_size (config or full) and train_micro_batch_size (config or mini).
    # Rollout micro batch size is forced to 1. If compute_logps micro batch size > 1
    # it must equal train_micro_batch_size (else ValueError) and switches to
    # "process in consumer" mode (raw groups are converted in the training loop).
    # Checks the rollout, logps and mini batch sizes divide full_batch_size.
    # 3. Builds eval prompts as single-example dicts; creates the data queue,
    # orchestrator and prompt queue; prefetches off_policy_steps + 1 batches (None if
    # the data runs out); starts _producer on the background loop.
    # 4. Consumes items in groups of train_micro_batch_size (optionally sequence
    # packed when max_seq_token_per_tpu is set). For each micro-batch: stop (putting
    # None in the prompt queue) once max_steps is reached; increment iter steps;
    # build the merged TrainExample (convert groups in consumer mode, else
    # concatenate the TrainExamples along axis 0). If eval prompts exist and the
    # actor's train_steps is a multiple of eval_every_n_steps and differs from the
    # last eval step, run a full eval rollout on the loop (blocking) and pass the eval
    # examples to the update. Split the merged example along the batch axis into
    # chunks of train_micro_batch_size * num_generations sequences (only arrays whose
    # leading dim equals the total are sliced). Call update_actor (and update_critic
    # if present) num_iterations times in consumer mode, otherwise once.
    # 5. After full_batch_size / train_micro_batch_size micro-batches (one global
    # step): log a one-line summary (train reward mean, solve rate with threshold
    # 0.1, count, mean abs advantage, mean completion length, time, best-effort
    # trainer metrics such as loss, grad_norm, pg_loss, entropy, kl, log-ratio,
    # clipfrac, and eval reward/solve when eval ran), buffer the step time metric;
    # then either (weights not shared) acquire the weight-sync lock, sync weights
    # (which increments global_steps inside the cluster), increment policy_version,
    # enqueue the next prompt batch, and release the lock; or (shared weights)
    # increment global_steps and enqueue the next batch. Export perf metrics and
    # reset per-step counters.
    # 6. Waits for the producer future (re-raising its exception) and closes the
    # cluster.
    # KNOWN ISSUE: the micro-batch count per global step uses full_batch_size //
    # train_micro_batch_size, which assumes divisibility; in process-in-consumer mode
    # the micro-batch summary uses only the last micro-batch's advantages and masks.
    """Main training loop for the AgenticRLLearner."""
    full_batch_iterator = iter(train_dataset)

    if self.rl_cluster.global_steps > 0:
      logging.info(
          "Skipping %d batches from train_dataset to fast-forward to step %d",
          self.rl_cluster.global_steps,
          self.rl_cluster.global_steps,
      )
      # TODO(b/483779605): Current implementation of fast-forwarding does not
      # take into account the mini-batch size. Follow-up CL will address this.
      for _ in range(self.rl_cluster.global_steps):
        try:
          next(full_batch_iterator)
        except StopIteration:
          logging.warning("Train dataset exhausted while skipping batches.")
          self.rl_cluster.close()
          return

    try:
      first_item = next(full_batch_iterator)
    except StopIteration:
      logging.warning("Training dataset is empty.")
      self.rl_cluster.close()
      return

    full_batch_size = len(next(iter(first_item.values())))
    self._full_batch_size = full_batch_size
    # Initialize batch sizes.
    mini_batch_size = self._training_config.mini_batch_size or full_batch_size
    train_micro_batch_size = (
        self._training_config.train_micro_batch_size or mini_batch_size
    )
    # Rollout micro batch size has to be 1 since we only process individual
    # prompts.
    self._rollout_micro_batch_size = 1
    self._process_in_consumer = False

    if self._compute_logps_micro_batch_size > 1:
      if self._compute_logps_micro_batch_size != train_micro_batch_size:
        raise ValueError(
            "compute_logps_micro_batch_size"
            f" ({self._compute_logps_micro_batch_size}) must be equal to"
            f" train_micro_batch_size ({train_micro_batch_size})"
        )
      self._process_in_consumer = True

    for v, n in [
        (self._rollout_micro_batch_size, f"{self._rollout_micro_batch_size=}"),
        (
            self._compute_logps_micro_batch_size,
            f"{self._compute_logps_micro_batch_size=}",
        ),
        (mini_batch_size, f"{mini_batch_size=}"),
    ]:
      rl_utils.check_divisibility(v, full_batch_size, n, f"{full_batch_size=}")
    grad_acc_steps = self._training_config.get_with_default(
        "gradient_accumulation_steps", 1
    )

    logging.info(  # pylint: disable=logging-fstring-interpolation
        f"Training with {full_batch_size=}, {mini_batch_size=},"
        f" {train_micro_batch_size=}, {self._rollout_micro_batch_size=},"
        f" {self._compute_logps_micro_batch_size=}, {grad_acc_steps=}"
    )

    logging.info("Starting AgenticRLLearner training loop.")
    full_dataset_iterator = itertools.chain([first_item], full_batch_iterator)

    all_eval_prompts = (
        list(self._create_micro_batch_iterator(iter(eval_dataset), 1))
        if eval_dataset
        else []
    )

    training_config = self.rl_cluster.cluster_config.training_config

    train_data_queue = queue_lib.SimpleDataQueue(maxsize=0)

    # 1. Start producer thread to generate rollouts and training examples.
    orchestrator = self._build_orchestrator()

    prompt_queue = queue.Queue()
    initial_buffer_size = self.algo_config.off_policy_steps + 1
    logging.info(
        "Prefilling prompt queue with %d batches.", initial_buffer_size
    )
    for _ in range(initial_buffer_size):
      try:
        self._put_prompts_to_queue(prompt_queue, next(full_dataset_iterator))
      except StopIteration:
        prompt_queue.put(None)
        break

    producer_future = asyncio.run_coroutine_threadsafe(
        self._producer(orchestrator, prompt_queue, train_data_queue),
        self.loop,
    )

    # 2. Consume training examples and train.
    train_data_gen = self._data_consumer_batch_generator(
        train_data_queue, train_micro_batch_size
    )
    if self._training_config.max_seq_token_per_tpu is not None:
      logging.info(
          "Using sequence packing with max_seq_token_per_tpu: %d",
          self._training_config.max_seq_token_per_tpu,
      )
      train_data_gen = rl_utils.pack_sequences(
          train_data_gen, self._training_config.max_seq_token_per_tpu
      )
    micro_batches_since_last_sync = 0
    micro_batches_per_full_batch = full_batch_size // train_micro_batch_size
    did_eval_this_global_step = False
    for train_micro_batch in train_data_gen:
      if (
          self._training_config.max_steps
          and self.rl_cluster.global_steps >= self._training_config.max_steps
      ):
        logging.info(
            "Reached max_steps: %d >= %d",
            self.rl_cluster.global_steps,
            self._training_config.max_steps,
        )
        prompt_queue.put(None)
        break
      self._iter_steps += 1

      # TODO(tsbao): Re-enable this once off-policy filtering is needed.
      # Filter out examples that are too old (off-policy).
      # filtered_train_micro_batch = self._filter_outdated_offpolicy_examples(
      #     train_micro_batch
      # )
      # if not filtered_train_micro_batch:
      #   continue
      # train_micro_batch = filtered_train_micro_batch

      if self._process_in_consumer:
        # train_micro_batch is a list of lists of trajectories.
        all_trajectories = [t for group in train_micro_batch for t in group]
        train_examples = self._batch_to_train_example(
            batch_results=all_trajectories,
            mode=rl_cluster_lib.Mode.TRAIN,
        )
        # GRPO returns a list with a single TrainExample.
        merged_train_micro_batch = train_examples[0]
      else:
        merged_train_micro_batch = jax.tree.map(
            lambda *xs: jnp.concatenate(xs, axis=0), *train_micro_batch
        )

      # --- Evaluation Logic ---
      current_eval_dataset = None
      current_train_step = self.rl_cluster.actor_trainer.train_steps
      if (
          all_eval_prompts
          and current_train_step % training_config.eval_every_n_steps == 0
          and current_train_step != self._last_eval_train_step
      ):
        self._last_eval_train_step = current_train_step
        self._eval_iter_steps = 0
        eval_orchestrator = self._build_orchestrator()

        async def _eval_runner_async(current_eval_orchestrator):
          # [FUNCTION DOC]
          # HELPER: Coroutine collecting eval TrainExamples by running _orchestrator_producer
          # over all eval prompts (EVAL mode conversion) and returning the concatenated
          # list.
          eval_examples = []
          async for batch in self._orchestrator_producer(
              current_eval_orchestrator,
              all_eval_prompts,
              num_generations=self._num_generations(),
          ):
            eval_example = self._batch_to_train_example(
                batch,
                rl_cluster_lib.Mode.EVAL,
            )
            eval_examples.extend(eval_example)
          return eval_examples

        eval_future = asyncio.run_coroutine_threadsafe(
            _eval_runner_async(eval_orchestrator), self.loop
        )
        eval_examples = eval_future.result()
        self._eval_iter_steps += 1
        current_eval_dataset = eval_examples
        did_eval_this_global_step = True

      # --- Training Step ---
      iterations = self._num_iterations() if self._process_in_consumer else 1

      # When ``train_micro_batch_size < mini_batch_size`` we want the trainer
      # to invoke ``train_step`` multiple times per outer iteration so the
      # optimizer (which fires every ``gradient_accumulation_steps`` micro-
      # steps) sees ``mini_batch_size``-shaped gradients while peak HBM is
      # only ``train_micro_batch_size``-shaped. Slice the merged train
      # example along its batch axis into chunks sized to one micro-step,
      # and pass the list to ``update_actor``; ``peft_trainer.train``
      # iterates the list and calls ``train_step`` once per chunk.
      seqs_per_chunk = (
          train_micro_batch_size * self.algo_config.num_generations
      )
      n_total = merged_train_micro_batch.completion_ids.shape[0]
      if n_total > seqs_per_chunk:
        chunked_train_micro_batch = [
            jax.tree_util.tree_map(
                lambda x: (
                    x[i : i + seqs_per_chunk]
                    if hasattr(x, "shape") and x.shape and x.shape[0] == n_total
                    else x
                ),
                merged_train_micro_batch,
            )
            for i in range(0, n_total, seqs_per_chunk)
        ]
      else:
        chunked_train_micro_batch = [merged_train_micro_batch]

      for i in range(iterations):
        if self._process_in_consumer and i > 0:
          # TODO(b/483779605) Sub-step checkpointing.
          self._iter_steps += 1

        self.rl_cluster.update_actor(
            chunked_train_micro_batch, current_eval_dataset, skip_jit
        )
        if hasattr(self.rl_cluster, "critic_trainer"):
          self.rl_cluster.update_critic(
              chunked_train_micro_batch, current_eval_dataset, skip_jit
          )

      # --- Weight Sync Logic ---
      micro_batches_since_last_sync += 1
      if micro_batches_since_last_sync == micro_batches_per_full_batch:
        global_step_time = time.time() - self._global_step_start_time
        logging.info(
            f"Global step {self.rl_cluster.global_steps} completed in"
            f" {global_step_time:.2f} seconds."
        )
        # One-line per-step diagnostic: raw rewards, solve rate, completion
        # length, advantage scale, and eval (when an eval just fired this
        # step). Mirrors the per-iter view a wandb dashboard would show
        # without depending on the async metric logger pipeline.
        with self._rewards_window_lock:
          train_rewards = np.asarray(self._train_rewards_window, dtype=np.float32)
          eval_rewards = np.asarray(self._eval_rewards_window, dtype=np.float32)
          self._train_rewards_window.clear()
          if did_eval_this_global_step:
            self._eval_rewards_window.clear()
        adv = np.asarray(merged_train_micro_batch.advantages, dtype=np.float32)
        cmask = np.asarray(
            merged_train_micro_batch.completion_mask, dtype=np.float32
        )
        compl_len = cmask.sum(axis=-1).mean() if cmask.size else 0.0
        adv_abs_mean = float(np.abs(adv).mean()) if adv.size else float("nan")
        train_r_mean = (
            float(train_rewards.mean()) if train_rewards.size else float("nan")
        )
        train_solve = (
            float((train_rewards > 0.1).mean())
            if train_rewards.size
            else float("nan")
        )
        if eval_rewards.size and did_eval_this_global_step:
          eval_r_mean = float(eval_rewards.mean())
          eval_solve = float((eval_rewards > 0.1).mean())
          eval_str = (
              f" eval_reward={eval_r_mean:.3f}"
              f" eval_solve={eval_solve:.3f}"
              f" eval_n={eval_rewards.size}"
          )
        else:
          eval_str = ""
        # Best-effort read of trainer-side per-step metrics (grad_norm,
        # pg_loss, entropy, kl) directly from the actor trainer's metric
        # buffer so they appear in the per-step absl log alongside the
        # rollout metrics, independently of any external metric logger.
        trainer_str = ""
        try:
          actor_trainer = self.rl_cluster.actor_trainer
          trainer_buf = (
              getattr(actor_trainer, "_prev_buffered_train_metrics", None)
              or getattr(actor_trainer, "_buffered_train_metrics", None)
          )
          if trainer_buf is not None:
            extras = []
            if trainer_buf.losses:
              extras.append(f"loss={float(trainer_buf.loss):.4f}")
            am = trainer_buf.additional_metrics
            for key, label in (
                ("grad_norm", "grad_norm"),
                ("pg_loss", "pg_loss"),
                ("entropy", "entropy"),
                ("kl", "kl"),
                ("log_ratio/abs_mean", "log_ratio_abs"),
                ("pg_clipfrac", "clipfrac"),
            ):
              if key in am:
                vals, _ = am[key]
                if vals:
                  v = float(np.mean([np.asarray(x) for x in vals]))
                  extras.append(f"{label}={v:.4f}")
            if extras:
              trainer_str = " " + " ".join(extras)
        except Exception as e:  # pylint: disable=broad-except
          logging.debug("Failed to read trainer buffered metrics: %s", e)
        logging.info(
            "[step %d] train_reward=%.3f train_solve=%.3f n=%d"
            " adv_abs_mean=%.3f compl_len=%.1f time=%.1fs%s%s",
            self.rl_cluster.global_steps,
            train_r_mean,
            train_solve,
            int(train_rewards.size),
            adv_abs_mean,
            float(compl_len),
            global_step_time,
            trainer_str,
            eval_str,
        )
        self.rl_cluster.buffer_metrics_async(
            {"perf/global_step_time": (global_step_time, np.mean)},
            mode=rl_cluster_lib.Mode.TRAIN,
            step=self.rl_cluster.global_steps,
        )
        if self.should_sync_weights:
          logging.info("Requesting sync lock to sync weights...")
          self._rollout_sync_lock.acquire_weight_sync()
          try:
            logging.info("Sync lock acquired. Syncing weights.")
            with self.rl_cluster.perf_v2.span(
                perf_constants.WEIGHT_SYNC,
                self.rl_cluster.perf_v2.all_devices,
                tags={
                    perf_constants.STEP: self.rl_cluster.global_steps,
                },
            ):
              self.rl_cluster.sync_weights()
            self.policy_version += 1
            logging.info(
                "Weights synced. Policy version incremented to %d.",
                self.policy_version,
            )
            try:
              with self.rl_cluster.perf_v2.span(
                  perf_constants.DATA_LOADING,
                  tags={
                      perf_constants.STEP: self.rl_cluster.global_steps,
                  },
              ):
                batch = next(full_dataset_iterator)
              self._put_prompts_to_queue(prompt_queue, batch)
            except StopIteration:
              prompt_queue.put(None)
          finally:
            self._rollout_sync_lock.release_weight_sync()
            logging.info("Sync lock released.")
        else:
          self.rl_cluster.global_steps += 1
          try:
            with self.rl_cluster.perf_v2.span(
                perf_constants.DATA_LOADING,
                tags={
                    perf_constants.STEP: self.rl_cluster.global_steps,
                },
            ):
              batch = next(full_dataset_iterator)
            self._put_prompts_to_queue(prompt_queue, batch)
          except StopIteration:
            prompt_queue.put(None)

        self.rl_cluster.buffer_metrics(
            self.rl_cluster.perf_v2.export(),
            mode=rl_cluster_lib.Mode.TRAIN,
        )
        micro_batches_since_last_sync = 0
        did_eval_this_global_step = False
        self._global_step_start_time = time.time()

    _ = producer_future.result()
    self.rl_cluster.close()

  def _put_prompts_to_queue(
      self,
      prompt_queue: queue.Queue[TrainingInputT | None],
      batch,
  ):
    # [FUNCTION DOC]
    # HELPER: Enqueue a prompt batch or an end-of-data marker.
    # Behavior: if max_steps is set and reached, puts None; else if the batch's size
    # (length of its first column) differs from full_batch_size, logs a warning and
    # puts None; else puts the batch.
    # KNOWN ISSUE: the docstring says StopIteration is raised; it is not.
    """Puts a batch of prompts into the queue.

    If the batch size does not match the expected full batch size, a warning is
    logged, and a StopIteration is raised to signal the end of the dataset.
    A None is put into the queue upon StopIteration to signal completion.

    Args:
      prompt_queue: The queue to put the batch into.
      batch: The batch of prompts (TrainingInputT).
    """
    current_batch_size = len(next(iter(batch.values())))
    if (
        self._training_config.max_steps
        and self.rl_cluster.global_steps >= self._training_config.max_steps
    ):
      logging.info(
          "Reached max_steps: %d >= %d",
          self.rl_cluster.global_steps,
          self._training_config.max_steps,
      )
      prompt_queue.put(None)
    elif current_batch_size != self._full_batch_size:
      logging.warning(
          "partial batch %d vs %d detected. The rest of the batch will be"
          " skipped.",
          current_batch_size,
          self._full_batch_size,
      )
      prompt_queue.put(None)
    else:
      prompt_queue.put(batch)

  def _filter_outdated_offpolicy_examples(
      self,
      train_micro_batch: List[TrainExample],
  ) -> List[TrainExample]:
    # [FUNCTION DOC]
    # HELPER: Keep examples whose policy_version is present and either -1 (unknown)
    # or at most off_policy_steps behind the current policy_version (first element
    # used). Logs a warning when all are dropped. Returns the kept list. Currently
    # unused (call site commented out).
    """Filters out outdated off-policy examples."""
    filtered_train_micro_batch = []
    for train_example in train_micro_batch:
      if train_example.policy_version is not None and (
          train_example.policy_version[0] == -1
          or (
              self.policy_version - train_example.policy_version[0]
              <= self.algo_config.off_policy_steps
          )
      ):
        filtered_train_micro_batch.append(train_example)
    if not filtered_train_micro_batch:
      logging.warning(
          "Skipping microbatch: all %d examples are too old."
          " Current policy version: %d, data versions: %s,"
          " off_policy_steps: %d",
          len(train_micro_batch),
          self.policy_version,
          str([
              train_example.policy_version[0]
              for train_example in train_micro_batch
          ]),
          self.algo_config.off_policy_steps,
      )
    return filtered_train_micro_batch
