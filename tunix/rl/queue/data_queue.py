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
# HELPER: Data queue abstraction used to pass items (e.g. rollout results or
# training examples) between producer and consumer threads. Defines a generic
# abstract interface (put/get/qsize/close) and a thread-safe in-memory
# implementation backed by the standard library queue.Queue. No import-time
# side effects.
"""Data queue interface."""

import abc
import queue
from typing import Generic, TypeVar

_T = TypeVar("_T")

ABC = abc.ABC
abstractmethod = abc.abstractmethod


class AbstractDataQueue(ABC, Generic[_T]):
  # [CLASS DOC]
  # Abstract generic queue interface over item type _T. Subclasses must
  # implement put, get, qsize and close. Cannot be instantiated directly.
  """Abstract base class for data queues."""

  @abstractmethod
  def put(self, item: _T) -> None:
    # [FUNCTION DOC]
    # Abstract. Input: item of type _T. Adds it to the queue; returns None.
    """Puts an item into the queue."""

  @abstractmethod
  def get(self, block: bool = True, timeout: float | None = None) -> _T:
    # [FUNCTION DOC]
    # Abstract. Inputs: block (wait if empty, default True), timeout (seconds or
    # None for no limit). Returns the next item; implementations raise
    # queue.Empty when no item is available within the constraints.
    """Gets an item from the queue."""

  @abstractmethod
  def qsize(self) -> int:
    # [FUNCTION DOC]
    # Abstract. Returns the approximate number of queued items (int).
    """Returns the approximate size of the queue."""

  @abstractmethod
  def close(self) -> None:
    # [FUNCTION DOC]
    # Abstract. Closes the queue / releases pending items; returns None.
    """Closes the queue."""


class SimpleDataQueue(AbstractDataQueue[_T]):
  # [CLASS DOC]
  # FIFO, thread-safe implementation of AbstractDataQueue wrapping a private
  # queue.Queue instance. Closing only drains items; it does not prevent later
  # puts or gets.
  """A simple data queue based on queue.Queue."""

  def __init__(self, maxsize: int = 0):
    # [FUNCTION DOC]
    # Input: maxsize (int, default 0 meaning unbounded). Creates the underlying
    # queue.Queue with that capacity and stores it in a private attribute.
    self._queue = queue.Queue(maxsize=maxsize)

  def put(self, item: _T) -> None:
    # [FUNCTION DOC]
    # Puts item into the underlying queue, blocking while the queue is full
    # (when bounded). Returns None.
    self._queue.put(item)

  def get(self, block: bool = True, timeout: float | None = None) -> _T:
    # [FUNCTION DOC]
    # Delegates to the underlying queue's get with the given block and timeout;
    # returns the item. Raises queue.Empty if non-blocking and empty, or if the
    # timeout elapses.
    return self._queue.get(block=block, timeout=timeout)

  def qsize(self) -> int:
    # [FUNCTION DOC]
    # Returns the underlying queue's approximate size.
    return self._queue.qsize()

  def close(self) -> None:
    # [FUNCTION DOC]
    # If the queue is non-empty, repeatedly removes items without blocking until
    # queue.Empty is raised, discarding them. Returns None. Side effect: all
    # pending items are lost. Does not mark the queue closed or wake blocked
    # consumers.
    if not self._queue.empty():
      while True:
        try:
          self._queue.get_nowait()
        except queue.Empty:
          break
