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
# Asyncio-based grouping queue for trajectory items. Producers put TrajectoryItem
# objects; items with the same group_id accumulate in a bucket until group_size
# items exist, at which point the bucket becomes a "ready group". Consumers pull
# fixed-size batches assembled from whole ready groups, with leftovers buffered.
# Supports propagating a producer exception to consumers and a two-phase clear.
# Defines aliases TrajectoryItem, field and dataclass (the latter two unused).
# KNOWN ISSUE: the module docstring mentions grouping by episode_id, but only
# group_id is used.
"""Manages queues of trajectory items, grouping them by group_id and episode_id."""

from __future__ import annotations

import asyncio
import collections
from collections.abc import Hashable
import dataclasses
from typing import Deque, Dict, List, Optional

from tunix.rl.agentic.agents import agent_types

TrajectoryItem = agent_types.TrajectoryItem
field = dataclasses.field
dataclass = dataclasses.dataclass


class GroupQueueManager:
  # [CLASS DOC]
  # Grouping queue. State: group_size (int); _buckets (dict group_id to list of
  # items, incomplete groups); _ready_groups (deque of complete groups, FIFO);
  # _clearing (bool flag); _exc (stored exception or None); _lock (asyncio.Lock
  # guarding bucket mutation in put and clear); _have_ready (asyncio.Event signaled
  # when a group becomes ready, an exception is set, or clearing begins);
  # _batch_buf (list of leftover items from a partially consumed group).
  # Must be used from a single event loop. Not thread-safe.
  """Manages queues of trajectory items, grouping them by group_id.

  This class collects `TrajectoryItem` instances into buckets based on their
  `group_id`. Once a bucket reaches `group_size`, it becomes a
  "ready group" and can be retrieved in batches. It also handles managing the
  number of open buckets and provides mechanisms for clearing and handling
  exceptions.
  """

  def __init__(
      self,
      *,
      group_size: int,
  ):
    # [FUNCTION DOC]
    # Initialize empty state with keyword-only group_size. Creates the asyncio Lock
    # and Event (bound to the running loop on first use).
    self.group_size = group_size
    self._buckets: Dict[Hashable, List[TrajectoryItem]] = {}
    self._ready_groups: Deque[List[TrajectoryItem]] = collections.deque()
    self._clearing = False
    self._exc: Optional[Exception] = None
    self._lock = asyncio.Lock()
    self._have_ready = asyncio.Event()
    self._batch_buf: List[TrajectoryItem] = []

  async def put_exception(self, exc: Exception):
    # [FUNCTION DOC]
    # Coroutine. Store exc so subsequent put and get calls raise it, and set the
    # ready event to wake any waiting consumer. Does not take the lock.
    self._exc = exc
    self._have_ready.set()

  async def prepare_clear(self):
    # [FUNCTION DOC]
    # Coroutine. Phase one of clearing: set _clearing so put becomes a no-op and
    # waiting consumers return an empty group, and set the ready event to wake them.
    self._clearing = True
    self._have_ready.set()

  async def clear(self):
    # [FUNCTION DOC]
    # Coroutine. Phase two of clearing, under the lock: empty buckets, ready groups and
    # batch buffer, reset the stored exception and _clearing flag, and replace the
    # ready event with a new Event.
    # KNOWN ISSUE: a consumer woken by prepare_clear that resumes only after clear()
    # has run sees _clearing False and no groups, then waits on the new event,
    # possibly forever if nothing else is produced.
    async with self._lock:
      self._buckets.clear()
      self._ready_groups.clear()
      self._batch_buf.clear()
      self._exc = None
      self._clearing = False
      self._have_ready = asyncio.Event()

  async def put(self, item: TrajectoryItem):
    # [FUNCTION DOC]
    # Coroutine. Add an item to its group's bucket.
    # Behavior: if clearing, silently drop the item; if an exception is stored,
    # raise it. Otherwise, under the lock (re-checking both conditions), append to the
    # bucket for item.group_id (creating it if needed). When the bucket length
    # equals group_size, append a copy of it to the ready deque, delete the bucket and
    # set the ready event.
    # Exceptions: re-raises the stored exception.
    # Note: a bucket only becomes ready on exact equality, so it never exceeds
    # group_size via this path.
    """Adds an item, grouping by `group_id`.

    Items are grouped in buckets. When a bucket reaches `self.group_size`, it's
    moved to `_ready_groups`.

    Args:
      item: The TrajectoryItem to add.

    Raises:
      Exception: If an exception has been set via `put_exception`.
    """
    if self._clearing:
      return
    if self._exc:
      raise self._exc
    key = item.group_id
    async with self._lock:
      if self._clearing:
        return
      if self._exc:
        raise self._exc
      bucket = self._buckets.setdefault(key, [])
      bucket.append(item)
      if len(bucket) == self.group_size:
        self._ready_groups.append(bucket.copy())
        del self._buckets[key]
        self._have_ready.set()

  async def _get_one_ready_group(self) -> List[TrajectoryItem]:
    # [FUNCTION DOC]
    # HELPER: Coroutine. Wait for and pop the oldest ready group. Loops: raise the
    # stored exception if any; return an empty list if clearing; pop and return the
    # leftmost ready group if any; otherwise wait on the ready event and then clear it.
    while True:
      if self._exc:
        raise self._exc
      if self._clearing:
        return []
      if self._ready_groups:
        return self._ready_groups.popleft()
      await self._have_ready.wait()
      self._have_ready.clear()

  async def get_batch(self, batch_size: int) -> List[TrajectoryItem]:
    # [FUNCTION DOC]
    # Coroutine. Return a batch of up to batch_size items.
    # Behavior: first takes as many items as possible (up to batch_size) from the
    # leftover buffer, returning immediately if that fills the batch. Then repeatedly
    # awaits a ready group: an empty group (clearing) ends the loop early; if the
    # group fits in the remaining room it is appended whole, otherwise the first part
    # fills the batch and the rest is appended to the leftover buffer.
    # Outputs: list of items; exactly batch_size unless clearing interrupted it.
    # Exceptions: re-raises a stored exception from the group wait (items already
    # taken from the buffer in this call are then lost to the caller).
    # Side effects: mutates leftover buffer and ready deque.
    """Retrieves a batch of TrajectoryItems, waiting until enough are ready.

    Items are taken from `_batch_buf` and then from `_ready_groups`. Excess
    items from groups are buffered in `_batch_buf`.

    Args:
      batch_size: The desired number of TrajectoryItems.

    Returns:
      A list of `TrajectoryItem` instances, up to `batch_size`.
    """
    out = []
    if self._batch_buf:
      take = min(batch_size, len(self._batch_buf))
      out.extend(self._batch_buf[:take])
      self._batch_buf = self._batch_buf[take:]
      if len(out) == batch_size:
        return out
    while len(out) < batch_size:
      group = await self._get_one_ready_group()
      if not group:
        break
      room = batch_size - len(out)
      if len(group) <= room:
        out.extend(group)
      else:
        out.extend(group[:room])
        self._batch_buf.extend(group[room:])
    return out
