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
# Performance tracer for RL workflows. PerfTracer records, relative to a shared
# "born" time (perf_counter at construction), nested span groups on every
# timeline, thread spans on per-thread timelines, and device spans on
# per-device timelines whose end time is when given JAX values become ready
# (waited on in background threads). Collected timelines can be printed or
# passed (wrapped in metrics.PerfSpanQuery) to a user export function that
# returns metrics. NoopTracer has the same API and does nothing. Tracer is the
# union type of both.
# Timeline ids: threads are "thread-<thread ident>", devices are
# "<platform><id>" (for example "tpu3") or a caller-supplied string.
# Aliases: JaxDevice (Any), MetricsT, PerfSpanQuery, Span, SpanGroup.
"""APIs of performance metrics for RL workflows.

Collect API:

  tracer = PerfTracer(devices, export_fn)

  1. span group

    with tracer.span_group("global_step"):
      with tracer.span_group("mini_batch"):
        with tracer.span_group("micro_batch"):
          ...

  2. thread span

    with tracer.span("data_loading"):
      ...

  3. device span

    with tracer.span("rollout", mesh.devices) as span:
      ...
      span.device_end(waitlist)
"""

from __future__ import annotations

from concurrent import futures
import contextlib
import threading
import time
from typing import Any, Callable

from absl import logging
import jax
import jaxtyping
import numpy as np
from tunix.perf import metrics
from tunix.perf import span


JaxDevice = Any
MetricsT = metrics.MetricsT
PerfSpanQuery = metrics.PerfSpanQuery
Span = span.Span
SpanGroup = span.SpanGroup


def create_thread_timeline_id() -> str:
  # [FUNCTION DOC]
  # Returns "thread-" followed by the current thread's identifier.
  return "thread-" + str(threading.get_ident())


def create_device_timeline_id(id: str | JaxDevice) -> str:
  # [FUNCTION DOC]
  # Returns the id unchanged if it is a str; if it has "platform" and "id"
  # attributes (a JAX device), returns platform concatenated with the id as
  # text; otherwise ValueError ("Unsupport id type").
  if isinstance(id, str):
    return id
  elif hasattr(id, "platform") and hasattr(id, "id"):
    # if it's a JAX device object, convert to string
    return getattr(id, "platform") + str(getattr(id, "id"))
  else:
    raise ValueError(f"Unsupport id type: {type(id)}")


def create_device_timeline_ids(
    devices: list[str | JaxDevice] | np.ndarray | None,
) -> list[str]:
  # [FUNCTION DOC]
  # Converts None to an empty list; flattens a numpy array of devices to a
  # list; maps each element with create_device_timeline_id. Output: list of
  # id strings. Exceptions: ValueError for unsupported elements.
  if devices is None:
    return []
  if isinstance(devices, np.ndarray):
    devices = devices.flatten().tolist()
  return [create_device_timeline_id(device) for device in devices]


class NoopTracer:
  # [CLASS DOC]
  # Tracer with the full PerfTracer API that records nothing: synchronize and
  # print do nothing, export returns an empty dict, all_devices is empty,
  # span_group yields None, span yields a fresh, ignored device waitlist.
  """An no-op tracer that does nothing."""

  def synchronize(self) -> None:
    # [FUNCTION DOC]
    # No-op.
    pass

  def print(self) -> None:
    # [FUNCTION DOC]
    # No-op.
    pass

  def export(self) -> MetricsT:
    # [FUNCTION DOC]
    # Returns an empty metrics dict.
    return {}

  @property
  def all_devices(self) -> list[str | JaxDevice]:
    # [FUNCTION DOC]
    # Property: empty list.
    return []

  @contextlib.contextmanager
  def span_group(self, name: str):
    # [FUNCTION DOC]
    # Context manager yielding None; records nothing.
    yield

  @contextlib.contextmanager
  def span(
      self, name: str, devices: list[str | JaxDevice] | np.ndarray | None = None
  ):
    # [FUNCTION DOC]
    # Context manager yielding a new _DeviceWaitlist whose contents are
    # ignored; devices argument ignored.
    yield _DeviceWaitlist()


class PerfTracer(NoopTracer):
  # [CLASS DOC]
  # Recording tracer. State: optional export function, born time, main thread
  # timeline id, dict of thread timelines (initially only the constructing
  # thread's), dict of device timelines keyed by device id string.
  # Not thread-safe (timelines and stacks are mutated without locks).
  """Provides an API to collect events to construct thread and devices timelines."""

  def __init__(
      self,
      devices: list[str | JaxDevice] | np.ndarray | None = None,
      export_fn: Callable[[PerfSpanQuery], MetricsT] | None = None,
  ):
    # [FUNCTION DOC]
    # Inputs: devices (optional list/array of devices or id strings, each gets
    # a device timeline up front), export_fn (optional function from
    # PerfSpanQuery to metrics dict). Records born = perf_counter now and
    # creates the main-thread timeline. Exceptions: ValueError for unsupported
    # device ids.
    self._export_fn = export_fn

    # align all timelines with the same born time.
    self._born = time.perf_counter()

    self._main_thread_id = create_thread_timeline_id()

    self._thread_timelines: dict[str, ThreadTimeline] = {
        self._main_thread_id: ThreadTimeline(self._main_thread_id, self._born)
    }
    self._device_timelines: dict[str, DeviceTimeline] = {}
    if devices:
      for device in devices:
        self._get_or_create_device_timeline(device)

  def _get_timelines(self) -> dict[str, Timeline]:
    # [FUNCTION DOC]
    # HELPER: Returns a new dict id -> timeline of all thread timelines
    # followed by all device timelines.
    timelines: dict[str, Timeline] = {}
    for timeline in self._thread_timelines.values():
      timelines[timeline.id] = timeline
    for timeline in self._device_timelines.values():
      timelines[timeline.id] = timeline
    return timelines

  def _get_or_create_thread_timeline(self, id: str) -> ThreadTimeline:
    # [FUNCTION DOC]
    # HELPER: Returns the thread timeline for id, creating it on first use with
    # a clone of the main thread's current group stack (so the new thread's
    # spans nest under the currently open groups).
    if id not in self._thread_timelines:
      self._thread_timelines[id] = ThreadTimeline(
          id,
          self._born,
          span.span_group_stack_clone(
              self._thread_timelines[self._main_thread_id].stack
          ),
      )
    return self._thread_timelines[id]

  def _get_or_create_device_timeline(
      self, id: str | JaxDevice
  ) -> DeviceTimeline:
    # [FUNCTION DOC]
    # HELPER: Returns (creating if absent) the device timeline for the
    # device's id string.
    tid = create_device_timeline_id(id)

    if tid not in self._device_timelines:
      self._device_timelines[tid] = DeviceTimeline(tid, self._born)
    return self._device_timelines[tid]

  def _get_or_create_device_timelines(
      self, ids: list[str | JaxDevice] | np.ndarray | None
  ) -> BatchDeviceTimeline:
    # [FUNCTION DOC]
    # HELPER: Wraps the (created on demand) device timelines for the given
    # devices in a BatchDeviceTimeline (empty when devices is None).
    return BatchDeviceTimeline([
        self._get_or_create_device_timeline(id)
        for id in create_device_timeline_ids(ids)
    ])

  def synchronize(self) -> None:
    # [FUNCTION DOC]
    # Blocks until all devices finish outstanding work (puts and waits on a
    # scalar per JAX device) and joins all pending device-span waiter threads.
    _synchronize_devices()
    for timeline in self._device_timelines.values():
      timeline.wait_pending_spans()

  def print(self) -> None:
    # [FUNCTION DOC]
    # Synchronizes, then prints each timeline's id header and its span tree
    # (times relative to born) to stdout.
    self.synchronize()
    for timeline in self._get_timelines().values():
      print(f"\n[{timeline.id}]")
      span.span_group_print(timeline.root, self._born)

  def export(self) -> MetricsT:
    # [FUNCTION DOC]
    # If an export function exists, calls it with a PerfSpanQuery over all
    # timelines and the main thread id and returns its result; else {}.
    # Does not synchronize first (pending device spans may be missing).
    if self._export_fn is not None:
      query = PerfSpanQuery(self._get_timelines(), self._main_thread_id)
      return self._export_fn(query)
    else:
      return {}

  @property
  def all_devices(self) -> list[str | JaxDevice]:
    # [FUNCTION DOC]
    # Property: list of device timeline id strings.
    # KNOWN ISSUE: annotated as devices but returns id strings.
    return list(self._device_timelines.keys())

  @contextlib.contextmanager
  def span_group(self, name: str):
    # [FUNCTION DOC]
    # Context manager: at entry, begins a group named name at the current time
    # on every existing timeline; at exit (also on exception) ends the top
    # group on every timeline at the exit time. Timelines created inside the
    # block (new threads/devices) get only the end call; a device timeline
    # created inside will raise ValueError when its stack has only the root.
    begin = time.perf_counter()
    for timeline in self._get_timelines().values():
      timeline.span_group_begin(name, begin)
    try:
      yield
    finally:
      end = time.perf_counter()
      for timeline in self._get_timelines().values():
        timeline.span_group_end(end)

  @contextlib.contextmanager
  def span(
      self, name: str, devices: list[str | JaxDevice] | np.ndarray | None = None
  ):
    # [FUNCTION DOC]
    # Context manager recording a span named name. On entry: records begin
    # time, gets/creates the current thread's timeline, starts a thread span
    # (ValueError if one is already open on that thread), yields a
    # _DeviceWaitlist. On exit (also on exception): ends the thread span and,
    # for each given device, records a device span starting at the thread
    # begin whose end is when all values passed to device_end are ready
    # (immediately if none were passed).
    begin = time.perf_counter()
    thread_timeline = self._get_or_create_thread_timeline(
        create_thread_timeline_id()
    )
    thread_timeline.span_begin(name, begin)
    device_waitlist = _DeviceWaitlist()
    try:
      yield device_waitlist
    finally:
      end = time.perf_counter()
      thread_timeline.span_end(end)
      self._get_or_create_device_timelines(devices).span(name, begin, device_waitlist._data)  # pylint: disable=protected-access


Tracer = PerfTracer | NoopTracer


class Timeline:
  # [CLASS DOC]
  # Base timeline: id, born, root group (named "root", begin = born), and a
  # stack of currently open groups (root at bottom). Tracks the last recorded
  # span to enforce that spans on one timeline are sequential, not nested.
  """Provides an API to collect events to construct a tree of spans and groups."""

  id: str
  born: float
  root: SpanGroup
  stack: list[SpanGroup]

  def __init__(
      self, id: str, born: float, stack: list[SpanGroup] | None = None
  ):
    # [FUNCTION DOC]
    # Inputs: id, born, optional existing stack. Without a stack a new root is
    # created; with a stack, its first element is the root. Last span starts
    # as None.
    self.id = id
    self.born = born
    if stack is None:
      self.root = SpanGroup("root", None)
      self.root.begin = born
      self.stack = [self.root]
    else:
      self.root = stack[0]
      self.stack = stack

    self._last_span: Span | None = None
    # TODO(yangmu): add lock to protect stack and _last_span.

  def _stack_debug(self) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns the stack names joined as "root -> a -> b".
    out = f"{self.root.name}"
    for group in self.stack[1:]:
      out += f" -> {group.name}"
    return out

  def span_group_begin(self, name: str, begin: float) -> SpanGroup:
    # [FUNCTION DOC]
    # Opens a child group of the top group with the given begin time and pushes
    # it; warns (does not fail) if the last span is still open. Returns it.
    if self._last_span and not self._last_span.ended:
      logging.warning(
          f"{self.id}: last span '{self._last_span.name}' is not ended. current"
          f" group stack: {self._stack_debug()}"
      )
    inner = SpanGroup(name, self.stack[-1])
    inner.begin = begin
    self.stack.append(inner)
    # print(f"{self.id}: begin {name}")
    return inner

  def span_group_end(self, end: float) -> None:
    # [FUNCTION DOC]
    # Closes the top group by setting its end and popping it. ValueError if
    # only the root remains. Warns if the last span is still open.
    if len(self.stack) == 1:
      raise ValueError(f"{self.id}: no more span groups to end.")
    if self._last_span and not self._last_span.ended:
      logging.warning(
          f"{self.id}: last span '{self._last_span.name}' is not ended. current"
          f" group stack: {self._stack_debug()}"
      )
    inner = self.stack.pop()
    inner.end = end
    # print(f"{self.id}: end {inner.name}")

  def device_span(
      self,
      name: str,
      *,
      thread_begin: float,
      end: float,
      group: SpanGroup | None = None,
  ) -> None:
    # [FUNCTION DOC]
    # Appends a completed span to group (or the current top group). Begin =
    # max(thread_begin, end of the last span on this timeline) so device spans
    # are serialized; end as given. ValueError if the last span is still open.
    # Updates the last span.
    """Records a device span within the timeline.

    The span's begin time is the maximum of `thread_begin` and the end time of
    the last recorded span on this timeline. The end time is given by `end`.

    Args:
      name: The name of the device span.
      thread_begin: The time when the thread initiated the device operation.
      end: The time when the device operation completed.
      group: The SpanGroup to which this device span belongs. If None, the
        current group on the stack is used.

    Raises:
      ValueError: If the last recorded span on this timeline has not been ended
        before calling this method.
    """

    if self._last_span and not self._last_span.ended:
      raise ValueError(
          f"{self.id}: last span '{self._last_span.name}' is not ended. current"
          f" group stack: {self._stack_debug()}"
      )

    if self._last_span and self._last_span.end > thread_begin:
      inner = Span(name, self._last_span.end)
    else:
      inner = Span(name, thread_begin)
    inner.end = end
    active_group = group or self.stack[-1]
    active_group.inner.append(inner)

    self._last_span = inner

  def thread_span_begin(self, name: str, begin: float) -> Span:
    # [FUNCTION DOC]
    # Opens a span at begin under the current top group and records it as last
    # span. ValueError if the previous span is still open. Returns the span.
    if self._last_span and not self._last_span.ended:
      raise ValueError(
          f"{self.id}: last span '{self._last_span.name}' is not ended. current"
          f" group stack: {self._stack_debug()}"
      )

    inner = Span(name, begin)
    self.stack[-1].inner.append(inner)

    self._last_span = inner
    return inner

  def thread_span_end(self, end: float) -> None:
    # [FUNCTION DOC]
    # Sets the end time of the last span. ValueError if there is no span or it
    # has already ended.
    if self._last_span is None:
      raise ValueError(f"{self.id}: no span to end.")
    if self._last_span.ended:
      raise ValueError(
          f"{self.id}: span '{self._last_span.name}' is already ended."
      )
    self._last_span.end = end


class ThreadTimeline(Timeline):
  # [CLASS DOC]
  # Timeline for a host thread; adds span_begin/span_end aliases.

  def span_begin(self, name: str, begin: float) -> None:
    # [FUNCTION DOC]
    # Delegates to thread_span_begin (return value discarded).
    self.thread_span_begin(name, begin)

  def span_end(self, end: float) -> None:
    # [FUNCTION DOC]
    # Delegates to thread_span_end.
    self.thread_span_end(end)


class DeviceTimeline(Timeline):
  # [CLASS DOC]
  # Timeline for one accelerator device; device spans are completed
  # asynchronously by waiter threads tracked in a list.
  """Manages a custom-annotated timeline for a device (e.g. TPU)."""

  def __init__(self, id: str, born: float):
    # [FUNCTION DOC]
    # Creates a fresh root/stack and an empty list of waiter threads.
    super().__init__(id, born)

    # wait pending data.
    self._threads: list[threading.Thread] = []

  def span(
      self, name: str, thread_span_begin: float, waitlist: jaxtyping.PyTree
  ) -> None:
    # [FUNCTION DOC]
    # Records a device span. Captures the current top group as parent. If
    # waitlist is empty/falsy, records the span immediately with end = now;
    # otherwise starts a background thread that blocks until the waitlist is
    # ready and then records the span (end = completion time) under the
    # captured parent, keeping the thread for later joining.
    # KNOWN ISSUE: when waiting fails, the re-raised exception happens in a
    # future callback and is only logged by concurrent.futures, never surfaced;
    # spans from concurrent waiters are appended in completion order without
    # locking; the waiter thread list is never cleared.
    """Record a new span for device (e.g. TPU).

    The span begin time is inferred from the thread span begin time (i.e. thread
    launches a computation on the device) and the end time of prevous span
    on the same device, the late one is used.

    The span end time is determined when all JAX computations associated
    with 'waitlist' finish.

    Args:
      name: The name of the span.
      thread_span_begin: The begin time of the span on the thread, used to infer
        the begin time of the span on the device.
      waitlist: The JAX computation to be tracked, used to infer the end time of
        the span on the device.
    """
    # Capture the current group when the span is initiated.
    # This ensures that even if the stack changes (e.g. a new group is pushed)
    # before the async wait finishes, the span is attached to the correct
    # parent group.
    # TODO(b/481789498): With multiple threads, this might still not be assigned
    # to the correct group in rare cases. We need a proper design to track the
    # true parent group for each device span.
    parent_group = self.stack[-1]

    def on_success():
      # [FUNCTION DOC]
      # HELPER: Records the device span with end = current perf_counter time
      # in the captured parent group.
      self.device_span(
          name,
          thread_begin=thread_span_begin,
          end=time.perf_counter(),
          group=parent_group,
      )

    def on_failure(e: Exception):
      # [FUNCTION DOC]
      # HELPER: Re-raises the waiting exception.
      raise e

    if not waitlist:
      on_success()
    else:
      t = _async_wait(waitlist=waitlist, success=on_success, failure=on_failure)
      self._threads.append(t)

  def wait_pending_spans(self) -> None:
    # [FUNCTION DOC]
    # Joins every waiter thread started so far (blocking).
    for t in self._threads:
      t.join()


class BatchDeviceTimeline:
  # [CLASS DOC]
  # Fan-out wrapper applying one device span to several DeviceTimelines.

  def __init__(self, timelines: list[DeviceTimeline]):
    # [FUNCTION DOC]
    # Stores the list of device timelines.
    self._timelines = timelines

  def span(
      self, name: str, thread_span_begin: float, waitlist: jaxtyping.PyTree
  ):
    # [FUNCTION DOC]
    # Calls span(name, thread_span_begin, waitlist) on each timeline (each
    # waits independently on the same waitlist).
    for timeline in self._timelines:
      timeline.span(name, thread_span_begin, waitlist)


########################################################################
#
# internal only - utility classes and functions
#
########################################################################


class _DeviceWaitlist:
  # [CLASS DOC]
  # HELPER: Collects JAX values (pytrees) inside a tracer span whose readiness
  # marks the device span end; stored in a list attribute "_data".
  """Provides an interface to collect waitlist for PerfTracer span()."""

  def __init__(self):
    # [FUNCTION DOC]
    # Initializes an empty list.
    self._data = []

  def device_end(self, waitlist: jaxtyping.PyTree) -> None:
    # [FUNCTION DOC]
    # Appends a pytree to the waitlist.
    self._data.append(waitlist)


# TODO(yangmu): maybe reuse `callback_on_ready` in tunix.rl.
def _async_wait(
    waitlist: jaxtyping.PyTree,
    success: Callable[[], None],
    failure: Callable[[Exception], None],
) -> threading.Thread:
  # [FUNCTION DOC]
  # HELPER: Starts and returns a thread that blocks until all arrays in
  # waitlist are ready, then resolves a Future; the Future's done callback
  # calls success() on success or failure(exception) on error (run on the
  # waiter thread).
  """Asynchronously wait for all JAX computations to finish."""
  fut = futures.Future()

  def callback(f):
    # [FUNCTION DOC]
    # HELPER: Future done callback dispatching to success or failure.
    e = f.exception()
    if e is None:
      success()
    else:
      failure(e)

  fut.add_done_callback(callback)

  def wait():
    # [FUNCTION DOC]
    # HELPER: Thread body: block until ready, then set the future's result or
    # exception.
    try:
      jax.block_until_ready(waitlist)
    except Exception as e:  # pylint: disable=broad-exception-caught
      fut.set_exception(e)
    else:
      fut.set_result(waitlist)

  t = threading.Thread(target=wait)
  t.start()
  return t


def _synchronize_devices():
  # [FUNCTION DOC]
  # HELPER: For every JAX device, places a scalar 0.0 on it and waits for it,
  # acting as a barrier on all devices' queued work.
  for device in jax.devices():
    jax.device_put(jax.numpy.array(0.0), device=device).block_until_ready()
