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
# HELPER: small utility module providing InflightThrottler, which bounds
#   how many asynchronously dispatched device computations may be pending
#   at once (to cap device memory used by queued input batches).
# No side effects at import.
"""Inflight computation throttler."""

import queue

import jax


class InflightThrottler:
  # [CLASS DOC]
  # Bounded FIFO of pending jax.Array results. Attribute _inflight_queue is
  # a queue.Queue with maxsize=max_inflight, or None when unlimited. Usage:
  # call wait_for_next before dispatching a step, add_computation with any
  # output array of that step after dispatch, wait_for_all at the end.
  # Not designed for multi-threaded producers (full-check then get).
  """Rate limits the number of inflight computations on TPU.

  This is to control the HBM usage and avoid too much allocation for input
  batches.

  Example usage:

  ```
  throttler = InflightThrottler(max_inflight=2)

  for _ in range(max_train_step):
    ...
    input_batch = get_next_batch()
    throttler.wait_for_next()
    output = train_step(input_batch, ...)  # Schedules TPU computation.
    throttler.add_computation(output) # output is some jax.Array. e.g. loss
    ...
  ```
  """

  def __init__(self, max_inflight: int):
    # [FUNCTION DOC]
    # Input: max_inflight (int). If > 0 create a bounded queue of that
    #   size; otherwise leave the queue None (no throttling).
    # No output, exceptions or side effects.
    """Ctor.

    Args:
      max_inflight: Max number of inflight computations that is allowed. If <=0,
        it means no limit.
    """
    self._inflight_queue: queue.Queue[jax.Array] | None = None
    if max_inflight > 0:
      self._inflight_queue = queue.Queue[jax.Array](maxsize=max_inflight)

  def add_computation(self, computation: jax.Array):
    # [FUNCTION DOC]
    # Input: computation (jax.Array representing pending device work).
    #   If throttling is enabled, put it in the queue (blocks if full, so
    #   callers should call wait_for_next first). Otherwise no-op.
    """Adds an active on-device computation to the queue."""
    if self._inflight_queue:
      self._inflight_queue.put(computation)

  def wait_for_next(self):
    # [FUNCTION DOC]
    # If throttling is enabled and the queue is full, dequeue the oldest
    #   array and block until it is ready (device work finished), freeing a
    #   slot. Otherwise returns immediately. No output.
    """If the limit is reached, wait for the next computation to finish."""
    if self._inflight_queue and self._inflight_queue.full():
      self._inflight_queue.get().block_until_ready()

  def wait_for_all(self):
    # [FUNCTION DOC]
    # If throttling is enabled, repeatedly dequeue arrays and block until
    #   each is ready until the queue is empty. No output.
    # Note: the queue's truthiness is that of a Queue object (always true
    #   when present), so the checks effectively test "not None".
    """Wait for all inflight computations to finish."""
    if self._inflight_queue:
      while not self._inflight_queue.empty():
        self._inflight_queue.get().block_until_ready()
