# [MODULE DOC]
# Async unit tests for RolloutOrchestrator streaming: producers consume a
# stream of (agent, env) pairs, collect trajectories concurrently (bounded by
# max_concurrency), group them via group_key_fn into groups of group_size,
# and yield_batches emits lists of TrajectoryItems of at most batch_size.
# The private trajectory collector is patched with an AsyncMock for all
# tests. Also checks that exceptions from collection and from the pair
# generator propagate to the consumer. Uses mock agent and environment
# classes defined here.
"""Tests for rollout_orchestrator."""

import asyncio
from collections.abc import Mapping
import math
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from tunix.rl.agentic import utils
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.agents import base_agent
from tunix.rl.agentic.environments import base_environment
from tunix.rl.agentic.pipeline import rollout_orchestrator


# Mock classes for dependencies
class MockAgent(base_agent.ConversationAgentBase):
  # [CLASS DOC]
  # HELPER: conversation agent with empty system prompt whose model update
  # always returns an empty Action.
  """A mock agent."""

  def __init__(self):
    # [FUNCTION DOC]
    # HELPER: initializes the base agent with an empty system prompt.
    super().__init__('')

  def update_from_model(self, response: str, **kwargs) -> agent_types.Action:
    # [FUNCTION DOC]
    # HELPER: ignores the response and returns a default Action.
    return agent_types.Action()


class MockEnv(base_environment.BaseTaskEnv):
  # [CLASS DOC]
  # HELPER: task environment tagged with env_id; extra kwargs (group_id,
  # pair_index) go to the base environment.
  """A mock environment."""

  def __init__(
      self, task: Mapping[str, Any] | None = None, env_id: int = 0, **kwargs
  ):
    # [FUNCTION DOC]
    # HELPER: forwards task and kwargs to the base init and stores env_id.
    super().__init__(task=task, **kwargs)
    self.env_id = env_id

  def _initial_observation(self):
    # [FUNCTION DOC]
    # HELPER: returns a dict with obs "initial_obs_<env_id>".
    return {'obs': f'initial_obs_{self.env_id}'}

  def _step_impl(self, action):
    # [FUNCTION DOC]
    # HELPER: returns a step result with obs "next_obs_<env_id>", reward
    # 1.0, done False and empty info, regardless of action.
    return base_environment.EnvStepResult(
        observation={'obs': f'next_obs_{self.env_id}'},
        reward=1.0,
        done=False,
        info={},
    )


class RolloutOrchestratorTest(parameterized.TestCase):
  # [CLASS DOC]
  # Tests for RolloutOrchestrator streaming production and batching.

  def setUp(self):
    # [FUNCTION DOC]
    # HELPER: patches RolloutOrchestrator._collect_trajectory with an
    # AsyncMock (self.mock_collect) and registers cleanup to stop the patch.
    super().setUp()
    self.collect_patcher = mock.patch.object(
        rollout_orchestrator.RolloutOrchestrator,
        '_collect_trajectory',
        new_callable=mock.AsyncMock,
    )
    self.mock_collect = self.collect_patcher.start()
    self.addCleanup(self.collect_patcher.stop)

  @parameterized.named_parameters(
      dict(
          testcase_name='group_size_1_batch_size_2',
          num_pairs=3,
          group_size=1,
          batch_size=2,
      ),
      dict(
          testcase_name='group_size_2_batch_size_2',
          num_pairs=4,
          group_size=2,
          batch_size=2,
      ),
      dict(
          testcase_name='group_size_1_batch_size_5',
          num_pairs=5,
          group_size=1,
          batch_size=5,
      ),
      dict(
          testcase_name='group_size_3_batch_size_3',
          num_pairs=6,
          group_size=3,
          batch_size=3,
      ),
      dict(
          testcase_name='group_size_2_batch_size_4',
          num_pairs=6,
          group_size=2,
          batch_size=4,
      ),
  )
  def test_streaming_successful_run(self, num_pairs, group_size, batch_size):
    # [FUNCTION DOC]
    # Runs the async successful-run scenario for the given parameters.
    asyncio.run(
        self._test_streaming_successful_run(num_pairs, group_size, batch_size)
    )

  async def _test_streaming_successful_run(
      self, num_pairs, group_size, batch_size
  ):
    # [FUNCTION DOC]
    # HELPER: streams num_pairs mock pairs into an orchestrator (max
    # concurrency 2) with group key pair_index integer-divided by group_size,
    # consumes all batches, then verifies: number of batches is ceil of
    # num_pairs / batch_size; each batch is at most batch_size; when
    # batch_size does not exceed a group size above 1, each batch holds one
    # group; every pair index appears exactly once; each item's traj is the
    # mocked trajectory for its env; group ids equal pair_index divided by
    # group_size; and each group contains exactly its contiguous indices.
    orchestrator = rollout_orchestrator.RolloutOrchestrator(
        max_concurrency=2,
        rollout_sync_lock=utils.RolloutSyncLock(),
    )

    def pair_generator():
      # [FUNCTION DOC]
      # HELPER: yields num_pairs (MockAgent, MockEnv) pairs with env_id and
      # pair_index equal to the loop index.
      for i in range(num_pairs):
        yield MockAgent(), MockEnv(env_id=i, group_id=0, pair_index=i)

    async def side_effect_fn(*args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: fake collector; reads the env from the second positional
      # argument and returns a trajectory dict naming its env_id.
      env = args[1]
      return {'trajectory': [f'traj_for_env_{env.env_id}']}

    self.mock_collect.side_effect = side_effect_fn

    producer_task = asyncio.create_task(
        orchestrator.run_producers_from_stream(
            pairs_stream=pair_generator(),
            group_size=group_size,
            group_key_fn=lambda i, env, traj: i // group_size,
        )
    )
    await asyncio.sleep(0)

    batches = []
    async for batch in orchestrator.yield_batches(batch_size=batch_size):
      batches.append(batch)
    await producer_task

    # Check if the orchestrator yields the correct number of batches.
    self.assertLen(batches, math.ceil(num_pairs / batch_size))
    for batch in batches:
      self.assertLessEqual(len(batch), batch_size)
      if group_size > 1 and batch_size <= group_size:
        # If group_size > 1 and batch_size <= group_size, items in a batch
        # are expected to come from the same group.
        group_ids = set(item.group_id for item in batch)
        self.assertLen(group_ids, 1)

    all_items = []
    for batch in batches:
      all_items.extend(batch)

    self.assertLen(all_items, num_pairs)

    pair_indices = sorted([item.pair_index for item in all_items])
    self.assertEqual(pair_indices, list(range(num_pairs)))

    items_by_group = {}
    for item in all_items:
      self.assertEqual(
          item.traj, {'trajectory': [f'traj_for_env_{item.pair_index}']}
      )
      self.assertEqual(item.group_id, item.pair_index // group_size)
      if item.group_id not in items_by_group:
        items_by_group[item.group_id] = []
      items_by_group[item.group_id].append(item)

    self.assertLen(items_by_group, num_pairs // group_size)
    for group_id in items_by_group:
      self.assertLen(items_by_group[group_id], group_size)
      pair_indices_in_group = sorted(
          [item.pair_index for item in items_by_group[group_id]]
      )
      expected_pair_indices = list(
          range(
              group_id * group_size,
              group_id * group_size + group_size,
          )
      )
      self.assertEqual(pair_indices_in_group, expected_pair_indices)

  def test_streaming_producer_runner_exception(self):
    # [FUNCTION DOC]
    # Runs the async collector-failure scenario.
    asyncio.run(self._test_streaming_producer_runner_exception())

  async def _test_streaming_producer_runner_exception(self):
    # [FUNCTION DOC]
    # HELPER: with 5 pairs where collecting env 2 raises ValueError, verifies
    # the error "Collection failed!" surfaces through yield_batches or the
    # awaited producer task.
    # KNOWN ISSUE: if the consumer loop raises, the producer task is never
    # awaited inside the test, which may leave it pending.
    orchestrator = rollout_orchestrator.RolloutOrchestrator(
        max_concurrency=2,
        rollout_sync_lock=utils.RolloutSyncLock(),
    )
    num_pairs = 5
    failing_pair_index = 2

    def pair_generator():
      # [FUNCTION DOC]
      # HELPER: yields num_pairs mock (agent, env) pairs.
      for i in range(num_pairs):
        yield MockAgent(), MockEnv(env_id=i, group_id=0, pair_index=i)

    async def failing_side_effect(*args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: fake collector raising ValueError for the failing env id,
      # otherwise returning a trajectory dict naming the env.
      env = args[1]
      if env.env_id == failing_pair_index:
        raise ValueError('Collection failed!')
      return {'trajectory': [f'traj_for_env_{env.env_id}']}

    self.mock_collect.side_effect = failing_side_effect

    producer_task = asyncio.create_task(
        orchestrator.run_producers_from_stream(
            pairs_stream=pair_generator(),
            group_size=1,
            group_key_fn=lambda i, *_: i,
        )
    )
    await asyncio.sleep(0)
    with self.assertRaisesRegex(ValueError, 'Collection failed!'):
      # Consumer loop.
      async for _ in orchestrator.yield_batches(batch_size=1):
        pass
      # Await producer to get the exception if not raised during consumption.
      await producer_task

  def test_streaming_generator_exception(self):
    # [FUNCTION DOC]
    # Runs the async pair-generator-failure scenario.
    asyncio.run(self._test_streaming_generator_exception())

  async def _test_streaming_generator_exception(self):
    # [FUNCTION DOC]
    # HELPER: with a pair generator that raises ValueError at index 2 and a
    # collector returning a fixed trajectory, verifies "Generator failed!"
    # propagates to the consumer or the awaited producer task.
    orchestrator = rollout_orchestrator.RolloutOrchestrator(
        max_concurrency=2,
        rollout_sync_lock=utils.RolloutSyncLock(),
    )
    failing_pair_index = 2

    def faulty_generator():
      # [FUNCTION DOC]
      # HELPER: yields mock pairs for indices 0 and 1, then raises
      # ValueError when reaching the failing index.
      for i in range(5):
        if i == failing_pair_index:
          raise ValueError('Generator failed!')
        yield MockAgent(), MockEnv(env_id=i, group_id=0, pair_index=i)

    self.mock_collect.side_effect = None
    self.mock_collect.return_value = {'trajectory': ['mock_traj']}

    producer_task = asyncio.create_task(
        orchestrator.run_producers_from_stream(
            pairs_stream=faulty_generator(),
            group_size=1,
            group_key_fn=lambda i, *_: i,
        )
    )
    await asyncio.sleep(0)
    with self.assertRaisesRegex(ValueError, 'Generator failed!'):
      async for _ in orchestrator.yield_batches(batch_size=1):
        pass
      await producer_task


if __name__ == '__main__':
  absltest.main()
