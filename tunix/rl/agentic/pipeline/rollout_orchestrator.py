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
# Producer/consumer orchestration of concurrent agent-environment rollouts.
# RolloutOrchestrator pulls (agent, env) pairs from a sync or async iterable,
# runs up to max_concurrency TrajectoryCollectEngine episodes at once as asyncio
# tasks (each guarded by the shared RolloutSyncLock so weight syncs can pause new
# rollouts), wraps each finished Trajectory in a TrajectoryItem and pushes it into
# a GroupQueueManager; a consumer async generator yields fixed-size batches of
# grouped items. Defines module aliases for Trajectory, ConversationAgentBase,
# BaseTaskEnv, TrajectoryCollectEngine, TrajectoryItem and GroupQueueManager.
# Requires Python 3.11+ (ExceptionGroup, aiter/anext builtins).
"""Orchestrates parallel rollouts of LLM agents in environments.

This module defines the `RolloutOrchestrator` class, which manages the
concurrent collection of trajectories from multiple agent-environment pairs and
groups them into batches for further processing.
"""

from __future__ import annotations

import asyncio
from collections.abc import Hashable
import copy
import traceback
from typing import Any, AsyncIterable, Callable, Dict, Iterable, List, Optional, Tuple, Type

from absl import logging
from tunix.rl.agentic import utils
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.agents import base_agent
from tunix.rl.agentic.environments import base_environment
from tunix.rl.agentic.queue_manager import group_queue_manager
from tunix.rl.agentic.trajectory import trajectory_collect_engine


Trajectory = agent_types.Trajectory
ConversationAgentBase = base_agent.ConversationAgentBase
BaseTaskEnv = base_environment.BaseTaskEnv
TrajectoryCollectEngine = trajectory_collect_engine.TrajectoryCollectEngine
TrajectoryItem = agent_types.TrajectoryItem
GroupQueueManager = group_queue_manager.GroupQueueManager


class RolloutOrchestrator:
  # [CLASS DOC]
  # Rollout orchestrator. State: engine_cls and engine_kwargs (used to build one
  # engine per episode), max_concurrency, _tasks (list of runner tasks still being
  # tracked), _stop (asyncio.Event signaling consumers/producers to stop),
  # _group_queue_manager (created by run_producers_from_stream, never reset, so the
  # orchestrator is single-use), _rollout_sync_lock (RolloutSyncLock).
  # Typical use: start run_producers_from_stream as a background task, then iterate
  # yield_batches.
  """Orchestrates parallel rollouts of LLM agents in environments.

  This class manages the concurrent collection of trajectories from multiple
  agent-environment pairs using `TrajectoryCollectEngine` instances. It groups
  the collected trajectories into batches via a `GroupQueueManager` and yields
  these batches for further processing.
  """

  def __init__(
      self,
      *,
      rollout_sync_lock: utils.RolloutSyncLock,
      engine_cls: Type[TrajectoryCollectEngine] = TrajectoryCollectEngine,
      engine_kwargs: Optional[Dict[str, Any]] = None,
      max_concurrency: Optional[int] = None,
  ):
    # [FUNCTION DOC]
    # Store configuration. Keyword-only inputs: rollout_sync_lock (required),
    # engine_cls (default TrajectoryCollectEngine), engine_kwargs (dict copied by
    # reference, default empty), max_concurrency (int or None). Creates an empty task
    # list, a stop Event and no queue manager yet.
    # KNOWN ISSUE: the docstring describes the lock as staggering rollout start times,
    # but it actually is a reader/writer lock that blocks rollouts during weight
    # syncs.
    """Initializes the RolloutOrchestrator.

    The orchestrator manages a pool of trajectory collection engines, each
    running an agent-environment interaction to collect a trajectory.
    Each output trajectory is considered an "episode".

    Args:
      rollout_sync_lock: A lock to synchronize the very start of multiple
        parallel rollout operations, ensuring they don't all start at the exact
        same moment, potentially overwhelming resources.
      engine_cls: The class used to instantiate trajectory collection engines.
        Each engine is responsible for running a single episode of interaction
        between an agent and an environment.
      engine_kwargs: A dictionary of default keyword arguments to be passed to
        the `engine_cls` constructor when creating new engine instances.
      max_concurrency: The maximum number of agent-environment interaction
        episodes to run in parallel. This limits the number of concurrent calls
        to the underlying language model.
    """
    self.engine_cls = engine_cls
    self.engine_kwargs = engine_kwargs or {}
    self.max_concurrency = max_concurrency
    self._tasks: List[asyncio.Task] = []
    self._stop = asyncio.Event()
    self._group_queue_manager: Optional[GroupQueueManager] = None
    self._rollout_sync_lock = rollout_sync_lock

  async def _collect_trajectory(
      self,
      agent: ConversationAgentBase,
      env: BaseTaskEnv,
      mode: Optional[str] = None,
      model_call_kwargs: Optional[Dict[str, Any]] = None,
  ) -> Trajectory:
    # [FUNCTION DOC]
    # HELPER: Coroutine. Build an engine and collect one trajectory.
    # Behavior: shallow-copies engine_kwargs, adds model_call_kwargs if provided,
    # constructs engine_cls(agent, env, **kwargs) and awaits engine.collect(mode) if
    # mode is truthy else engine.collect().
    # Outputs: the Trajectory returned by the engine. Exceptions propagate.
    """Helper method to collect a single trajectory."""
    engine_kwargs = self.engine_kwargs.copy()
    if model_call_kwargs:
      engine_kwargs["model_call_kwargs"] = model_call_kwargs
    engine = self.engine_cls(agent, env, **engine_kwargs)
    if mode:
      return await engine.collect(mode)
    return await engine.collect()

  async def _run_and_queue_one_episode(
      self,
      agent: ConversationAgentBase,
      env: BaseTaskEnv,
      manager: GroupQueueManager,
      group_key_fn: Callable[[int, BaseTaskEnv, Trajectory], Hashable],
      start_step_fn: Optional[Callable[[], int]],
      collect_mode: Optional[str],
  ):
    # [FUNCTION DOC]
    # HELPER: Coroutine. Collect one episode and enqueue it.
    # Behavior: reads pair_idx from env.extra_kwargs["pair_index"], collects a
    # trajectory with the given mode, computes group id via group_key_fn(pair_idx,
    # env, traj), start_step via start_step_fn() or 0, builds a TrajectoryItem with
    # metadata {"generation_id": pair_idx} and awaits manager.put.
    # Outputs: 1 (episode count). Exceptions: KeyError if env lacks pair_index;
    # anything from collection or put.
    """Collects one trajectory and queues it."""
    pair_idx = env.extra_kwargs["pair_index"]
    traj = await self._collect_trajectory(agent, env, mode=collect_mode)
    gid = group_key_fn(pair_idx, env, traj)
    start_step = start_step_fn() if start_step_fn else 0
    item = TrajectoryItem(
        pair_index=pair_idx,
        group_id=gid,
        start_step=start_step,
        traj=traj,
        metadata={"generation_id": pair_idx},
    )
    await manager.put(item)
    return 1

  async def _runner(
      self,
      agent: ConversationAgentBase,
      env: BaseTaskEnv,
      manager: GroupQueueManager,
      group_key_fn: Callable[[int, BaseTaskEnv, Trajectory], Hashable],
      start_step_fn: Optional[Callable[[], int]] = None,
      collect_mode: Optional[str] = None,
  ):
    # [FUNCTION DOC]
    # HELPER: Coroutine. Run exactly one episode for a pair under the rollout lock.
    # Behavior: logs start at debug level; acquires the rollout side of the
    # RolloutSyncLock, runs _run_and_queue_one_episode, and always releases the lock.
    # If an ExceptionGroup escapes, logs each sub-exception, prints the traceback and
    # re-raises the first sub-exception; other exceptions propagate unchanged. Always
    # logs completion with the episode count (0 or 1).
    # KNOWN ISSUE: acquire_rollout is a blocking threading call made on the event loop
    # thread; while a weight sync is pending it blocks the entire event loop, which
    # can stall or deadlock other coroutines. The docstring's "continuously collects"
    # and num_episodes do not match: only one episode runs.
    """Runs the trajectory collection loop for a single agent-environment pair.

    This method continuously collects trajectories using `_collect_trajectory`
    and puts them into the `GroupQueueManager`. It handles potential exceptions
    during trajectory collection and respects the `_stop` event and
    `num_episodes` limit.

    Args:
      agent: The ConversationAgentBase instance.
      env: The BaseTaskEnv instance.
      manager: The GroupQueueManager to put collected trajectories into.
      group_key_fn: A callable to determine the group ID for a trajectory.
      start_step_fn: An optional callable to get the starting step for each
        trajectory item.
      collect_mode: An optional string to select the collection mode.
    """
    episode_count = 0
    logging.debug(
        "Starting generating trajectories(_runner) for pair %d",
        env.extra_kwargs["pair_index"],
    )

    try:
      # Parallel execution for the group
      self._rollout_sync_lock.acquire_rollout()
      try:
        episode_count = await self._run_and_queue_one_episode(
            agent=agent,
            env=env,
            manager=manager,
            group_key_fn=group_key_fn,
            start_step_fn=start_step_fn,
            collect_mode=collect_mode,
        )
      finally:
        self._rollout_sync_lock.release_rollout()
    except ExceptionGroup as eg:
      for e in eg.exceptions:
        logging.error(
            "Fatal error in runner for pair %d: %s",
            env.extra_kwargs["pair_index"],
            e,
        )
      traceback.print_exc()
      raise eg.exceptions[0]
    finally:
      logging.debug(
          "Runner for pair %d completed with %d episodes",
          env.extra_kwargs["pair_index"],
          episode_count,
      )

  async def run_producers_from_stream(
      self,
      pairs_stream: (
          Iterable[Tuple[ConversationAgentBase, BaseTaskEnv]]
          | AsyncIterable[Tuple[ConversationAgentBase, BaseTaskEnv]]
      ),
      *,
      group_size: int,
      group_key_fn: Callable[
          [int, BaseTaskEnv, Trajectory], Hashable
      ] = lambda i, _, __: i,
      collect_mode: Optional[str] = None,
      start_step_fn: Optional[Callable[[], int]] = None,
  ):
    # [FUNCTION DOC]
    # Coroutine. Producer loop over a stream of (agent, env) pairs.
    # Inputs: pairs_stream (iterable or async iterable of pairs); keyword-only
    # group_size, group_key_fn (default returns pair index), collect_mode,
    # start_step_fn.
    # Behavior: logs start; raises ValueError if max_concurrency is falsy and
    # RuntimeError if a queue manager already exists. Creates a GroupQueueManager,
    # clears _stop and _tasks. Loop until stopped: fill the pool by drawing pairs
    # (using anext for async streams) and creating a _runner task for each while fewer
    # than max_concurrency are active; stream exhaustion is recorded; any other error
    # drawing a pair is logged and re-raised. Exit when no active tasks remain.
    # Otherwise wait for the first task(s) to finish, call result() on each (re-raising
    # a runner failure) and drop them from _tasks. After the loop, gathers any
    # remaining tracked tasks with exceptions suppressed.
    # Error handling: CancelledError is logged and re-raised; other exceptions are
    # logged, stored in the queue manager via put_exception (waking consumers with the
    # error) and re-raised. Finally (shielded from cancellation) calls
    # prepare_clear on the queue manager to tell consumers production has ended.
    # Side effects: creates asyncio tasks; logging.
    # KNOWN ISSUE: the initial info log formats max_concurrency with %d before the
    # None check, producing a logging error when it is None. More importantly,
    # prepare_clear makes the queue return empty groups immediately even if complete
    # groups are still queued, so groups not yet consumed when production ends are
    # dropped and the consumer stops early.
    """Dynamically runs collectors from a stream of agent-env pairs.

    This coroutine manages a pool of producer tasks. It draws pairs from
    `pairs_stream` and starts a `_runner` for each. It maintains up to
    `self.max_concurrency` active runners, starting new ones as they
    finish, until the `pairs_stream` is exhausted. This method is intended to
    be run as a background task. It sets up a shared queue that can be
    consumed from using `yield_batches`.

    Args:
      pairs_stream: An iterable of tuples, where each tuple contains an
        ConversationAgentBase and a BaseTaskEnv instance.
      group_size: The number of trajectories to collect before forming a group.
      group_key_fn: A callable that takes `(pair_index, env, trajectory)` and
        returns a hashable group identifier. Using a callable allows for
        flexible grouping strategies. For example, trajectories can be grouped
        by task properties from the environment (`env`) or by outcomes within
        the collected trajectory (`trajectory`). The default is to group by the
        agent-environment pair index.
      collect_mode: An optional string to select the collection mode for
        `TrajectoryCollectEngine`.
      start_step_fn: An optional callable to get the starting step for each
        trajectory item.

    Raises:
      ValueError: If `max_concurrency` is not set.
      RuntimeError: If the orchestrator is already running.
    """
    logging.info(
        "Starting run_producers_from_stream with %d concurrency",
        self.max_concurrency,
    )

    if not self.max_concurrency:
      raise ValueError("max_concurrency must be set to use start_producers.")
    if self._group_queue_manager:
      raise RuntimeError("Orchestrator is already running.")

    self._group_queue_manager = GroupQueueManager(group_size=group_size)
    self._stop.clear()
    self._tasks.clear()

    is_async_stream = hasattr(pairs_stream, "__aiter__")
    if is_async_stream:
      pairs_iterator = aiter(pairs_stream)  # pytype: disable=wrong-arg-types
    else:
      pairs_iterator = iter(pairs_stream)
    active_tasks: set[asyncio.Task] = set()
    stream_exhausted = False

    try:
      logging.debug(
          "Orchestrator producer loop starting with %d concurrency",
          self.max_concurrency,
      )
      while not self._stop.is_set():
        # Phase 1: Fill worker pool
        # As long as we have concurrency slots available and the input stream
        # is not exhausted, start new runner tasks.
        while (
            not stream_exhausted
            and len(active_tasks) < self.max_concurrency
            and not self._stop.is_set()
        ):
          try:
            if is_async_stream:
              agent, env = await anext(pairs_iterator)  # pytype: disable=name-error
            else:
              agent, env = next(pairs_iterator)
            task = asyncio.create_task(
                self._runner(
                    agent=agent,
                    env=env,
                    manager=self._group_queue_manager,
                    group_key_fn=group_key_fn,
                    start_step_fn=start_step_fn,
                    collect_mode=collect_mode,
                )
            )
            active_tasks.add(task)
            self._tasks.append(task)
          except (StopIteration, StopAsyncIteration):
            logging.debug("Pairs stream exhausted.")
            stream_exhausted = True
            break
          except Exception as e:
            logging.error(
                "Error getting next trajectory: %s",
                e,
            )
            raise e
        # If no tasks are running and stream is exhausted, done.
        if not active_tasks:
          break  # All done

        # Phase 2: Wait for any task to complete
        # This frees up a slot for a new task if the stream is not exhausted.
        done, pending = await asyncio.wait(
            active_tasks, return_when=asyncio.FIRST_COMPLETED
        )
        # Eagerly check for exceptions in completed tasks. If a runner fails,
        # it could cause a deadlock where the consumer waits for a group that
        # will never be completed. Propagating the exception ensures a clean
        # shutdown.
        for task in done:
          task.result()  # This will re-raise any exception in the task.
          # Remove the completed task from the _tasks list.
          if task in self._tasks:
            self._tasks.remove(task)
        active_tasks = pending

      # Wait for any stragglers if we were stopped prematurely
      if self._tasks:
        await asyncio.gather(*self._tasks, return_exceptions=True)
    except asyncio.CancelledError:
      logging.debug("Producer task was cancelled.")
      # The consumer's `finally` block will handle cleanup.
      raise
    except Exception as e:
      logging.error("Producer task failed: %s", e)
      if self._group_queue_manager:
        await self._group_queue_manager.put_exception(e)
      raise
    finally:
      # Shield the final cleanup step to ensure it runs even if the producer
      # task is being cancelled. This prevents leaving the manager in an
      # inconsistent state.
      if self._group_queue_manager:
        await asyncio.shield(self._group_queue_manager.prepare_clear())

  async def yield_batches(self, batch_size: int):
    # [FUNCTION DOC]
    # Async generator consumer. Yields lists of TrajectoryItem of size batch_size
    # (or smaller when production ends) from the queue manager until _stop is set or
    # an empty batch signals the end.
    # Exceptions: RuntimeError if producers were not started; queue errors (including
    # a producer exception stored via put_exception) are logged and re-raised.
    # GeneratorExit and CancelledError are swallowed as normal shutdown.
    # Side effects: in all cases on exit sets _stop and cancels every tracked runner
    # task that is not done (does not await them).
    """Yields batches of trajectories from the internal queue.

    This consumer method should be used in conjunction with
    `run_producers_from_stream`. It will yield batches until the producers have
    finished and the queue is empty. When the consumer is stopped (e.g., the
    async for loop is broken), it will trigger a cleanup of all background
    producer tasks.

    Args:
      batch_size: The maximum number of items to include in each yielded batch.

    Yields:
      A list of `TrajectoryItem` instances.

    Raises:
      RuntimeError: If `run_producers_from_stream` has not been called to start
        the producers.
    """
    if not self._group_queue_manager:
      raise RuntimeError("Producers have not been started.")
    try:
      while not self._stop.is_set():
        batch = await self._group_queue_manager.get_batch(batch_size)
        if not batch:
          # If batch is empty, it means producers are done and queue is empty.
          break
        yield batch
    except (GeneratorExit, asyncio.CancelledError):
      # This is the normal shutdown path when the consumer stops listening.
      pass
    except Exception as e:
      logging.error("Error yielding batches: %s", e)
      raise
    finally:
      # This block executes when the consumer (the 'async for' loop) stops.
      # The primary responsibility here is to signal all producers to stop.
      # We do not await task completion here as that's fragile in a generator's
      # finally block. Instead, we rely on the parent coroutine
      # (`run_producers_from_stream`) to handle the full cleanup, as it has
      # the correct context to await its child tasks.
      self._stop.set()
      logging.debug("Consumer stopped; signaling producers to stop.")
      for t in self._tasks:
        if not t.done():
          t.cancel()
