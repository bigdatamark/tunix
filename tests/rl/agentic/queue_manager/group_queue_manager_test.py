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
# Async unit tests for GroupQueueManager, which buffers TrajectoryItems by
# group_id until a group reaches group_size, then releases whole groups into
# batches via get_batch, keeping leftovers in an internal batch buffer.
# Also checks exception propagation via put_exception. Each test runs its
# coroutine with asyncio.run. No external side effects.

"""Tests for group_queue_manager."""

import asyncio

from absl.testing import absltest
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.queue_manager import group_queue_manager


def _create_item(
    group_id: str, pair_index: int = 0
) -> agent_types.TrajectoryItem:
  # [FUNCTION DOC]
  # HELPER: builds a TrajectoryItem with the given group_id and pair_index,
  # start_step 0 and traj None.
  """Helper to create a TrajectoryItem for testing."""
  return agent_types.TrajectoryItem(
      pair_index=pair_index,
      group_id=group_id,
      start_step=0,
      traj=None,
  )


class GroupQueueManagerTest(absltest.TestCase):
  # [CLASS DOC]
  # Tests for GroupQueueManager grouping, waiting, leftover and error paths.

  def test_put_and_get_simple_batch(self):
    # [FUNCTION DOC]
    # Verifies that with group_size 2 a group is not ready after one put,
    # becomes ready after the second, and get_batch(2) returns both items.
    """Tests basic put and get functionality."""

    async def _run_test():
      # [FUNCTION DOC]
      # HELPER: async body of the test.
      manager = group_queue_manager.GroupQueueManager(group_size=2)
      item1 = _create_item("g1", 0)
      item2 = _create_item("g1", 1)

      await manager.put(item1)
      self.assertEmpty(manager._ready_groups)

      await manager.put(item2)
      self.assertLen(manager._ready_groups, 1)

      batch = await manager.get_batch(2)
      self.assertLen(batch, 2)
      self.assertCountEqual([item1, item2], batch)

    asyncio.run(_run_test())

  def test_get_batch_waits_for_items(self):
    # [FUNCTION DOC]
    # Verifies get_batch(2) blocks until a concurrent producer task has put
    # both items of the group (with small sleeps between puts), then returns 2.
    """Tests that get_batch waits until a group is ready."""

    async def _run_test():
      # [FUNCTION DOC]
      # HELPER: async body of the test.
      manager = group_queue_manager.GroupQueueManager(group_size=2)
      item1 = _create_item("g1", 0)
      item2 = _create_item("g1", 1)

      async def producer():
        # [FUNCTION DOC]
        # HELPER: puts item1 then item2, sleeping 0.01s before each.
        await asyncio.sleep(0.01)
        await manager.put(item1)
        await asyncio.sleep(0.01)
        await manager.put(item2)

      producer_task = asyncio.create_task(producer())
      batch = await manager.get_batch(2)

      self.assertLen(batch, 2)
      await producer_task

    asyncio.run(_run_test())

  def test_batching_with_leftovers(self):
    # [FUNCTION DOC]
    # Verifies a ready group of 3 split by get_batch(2) yields the first two
    # items and leaves the third in _batch_buf, which get_batch(1) then
    # returns, leaving the buffer empty.
    """Tests batching where a group is split across two get_batch calls."""

    async def _run_test():
      # [FUNCTION DOC]
      # HELPER: async body of the test.
      manager = group_queue_manager.GroupQueueManager(group_size=3)
      items = [_create_item("g1", i) for i in range(3)]
      for item in items:
        await manager.put(item)

      self.assertEmpty(manager._batch_buf)
      batch1 = await manager.get_batch(2)
      self.assertLen(batch1, 2)
      self.assertCountEqual(items[:2], batch1)
      self.assertLen(manager._batch_buf, 1)
      self.assertEqual(manager._batch_buf[0], items[2])

      batch2 = await manager.get_batch(1)
      self.assertLen(batch2, 1)
      self.assertEqual(batch2[0], items[2])
      self.assertEmpty(manager._batch_buf)

    asyncio.run(_run_test())

  def test_put_exception(self):
    # [FUNCTION DOC]
    # Verifies that after put_exception(ValueError), subsequent put and
    # get_batch calls both raise that ValueError.
    """Tests that an exception is propagated to put and get calls."""

    async def _run_test():
      # [FUNCTION DOC]
      # HELPER: async body of the test.
      manager = group_queue_manager.GroupQueueManager(group_size=2)
      exc = ValueError("Test Exception")
      await manager.put_exception(exc)

      with self.assertRaises(ValueError):
        await manager.put(_create_item("g1", 0))

      with self.assertRaises(ValueError):
        await manager.get_batch(1)

    asyncio.run(_run_test())


if __name__ == "__main__":
  absltest.main()
