# Copyright 2025 Google LLC

# [MODULE DOC]
# Unit tests for tunix.perf.trace (perf v1 tracer): PerfTracer records host
# spans per thread timeline and device spans per device timeline using
# time.perf_counter (mocked with scripted values); span(..., devices=...)
# yields an object whose device_end(waitlist) waits for arrays
# (block_until_ready, possibly asynchronously via _async_wait) and ends the
# device span then; a device span begins at the later of its host begin and
# the previous device span's end; span_group creates nested groups mirrored
# on all timelines; nested plain spans raise ValueError; NoopTracer exposes
# the same public interface. Module aliases Timeline, DeviceTimeline,
# ThreadTimeline, NoopTracer, PerfTracer, patch and Mock for brevity.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading
import time
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from tunix.perf import span
from tunix.perf import trace

Timeline = trace.Timeline
DeviceTimeline = trace.DeviceTimeline
ThreadTimeline = trace.ThreadTimeline
NoopTracer = trace.NoopTracer
PerfTracer = trace.PerfTracer
patch = mock.patch
Mock = mock.Mock


def mock_array():
  # [FUNCTION DOC]
  # HELPER: returns a Mock spec'd on a local class with a no-op
  # block_until_ready, standing in for a JAX array waitlist.
  class JaxArray:
    # [CLASS DOC]
    # HELPER: minimal JAX-array-like spec with block_until_ready.

    def block_until_ready(self):
      # [FUNCTION DOC]
      # HELPER: no-op.
      pass

  return Mock(spec=JaxArray)


def timeline_tostring(timeline: Timeline) -> str:
  # [FUNCTION DOC]
  # HELPER: renders a timeline as '<id>: born=<t> stack=<stack>' plus the span
  # tree text from span.span_group_tostring relative to born; thread timeline
  # ids starting with 'thread-' are normalized to 'thread'.
  # ignore thread id diffs.
  timeline_id = (
      timeline.id if not timeline.id.startswith("thread-") else "thread"
  )
  return (
      f"{timeline_id}:"
      + f" born={timeline.born:.6f} stack={timeline._stack_debug()}\n"
      + span.span_group_tostring(timeline.root, timeline.born)
  )


def timelines_tostring(tracer: PerfTracer) -> list[str]:
  # [FUNCTION DOC]
  # HELPER: returns the rendered strings of all tracer timelines in order.
  return [
      timeline_tostring(timeline)
      for timeline in tracer._get_timelines().values()
  ]


class TracerTest(parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for PerfTracer and timelines.

  @mock.patch.object(time, "perf_counter", autospec=True)
  def test_host_ok(self, mock_perf_counter):
    # [FUNCTION DOC]
    # Verifies a host-only span 'x' is recorded on the thread timeline from 2 to
    # 3 under root (born 0). Input: patched perf_counter.
    mock_perf_counter.side_effect = [0.0, 2.0, 3.0]

    tracer = PerfTracer()
    with tracer.span("x"):
      pass
    tracer.synchronize()

    self.assertListEqual(
        timelines_tostring(tracer),
        [
            "thread: born=0.000000 stack=root\n"
            "- root (0.000000, inf)\n"
            "  - x (2.000000, 3.000000)\n"
        ],
    )

  @mock.patch.object(time, "perf_counter", autospec=True)
  def test_device_ok(self, mock_perf_counter):
    # [FUNCTION DOC]
    # Verifies a device span on 'tpu0' blocks on the waitlist once and records
    # host (2,3) on the thread timeline and (2,5) on the tpu0 timeline.
    mock_perf_counter.side_effect = [0.0, 2.0, 3.0, 5.0]
    waitlist = mock_array()

    tracer = PerfTracer()

    with tracer.span("x", devices=["tpu0"]) as span:
      span.device_end(waitlist)

    tracer.synchronize()

    waitlist.block_until_ready.assert_called_once()
    self.assertListEqual(
        timelines_tostring(tracer),
        [
            (
                "thread: born=0.000000 stack=root\n"
                "- root (0.000000, inf)\n"
                "  - x (2.000000, 3.000000)\n"
            ),
            (
                "tpu0: born=0.000000 stack=root\n"
                "- root (0.000000, inf)\n"
                "  - x (2.000000, 5.000000)\n"
            ),
        ],
    )

  @mock.patch.object(time, "perf_counter", autospec=True)
  def test_device_multi_ok(self, mock_perf_counter):
    # [FUNCTION DOC]
    # Verifies with tracer devices tpu0 and tpu1 only the targeted tpu0 gets the
    # span, while tpu1 has an empty root.
    mock_perf_counter.side_effect = [0.0, 2.0, 3.0, 5.0]
    waitlist = mock_array()

    tracer = PerfTracer(devices=["tpu0", "tpu1"])

    with tracer.span("int", devices=["tpu0"]) as span:
      span.device_end(waitlist)

    tracer.synchronize()

    waitlist.block_until_ready.assert_called_once()
    self.assertListEqual(
        timelines_tostring(tracer),
        [
            (
                "thread: born=0.000000 stack=root\n"
                "- root (0.000000, inf)\n"
                "  - int (2.000000, 3.000000)\n"
            ),
            (
                "tpu0: born=0.000000 stack=root\n"
                "- root (0.000000, inf)\n"
                "  - int (2.000000, 5.000000)\n"
            ),
            "tpu1: born=0.000000 stack=root\n- root (0.000000, inf)\n",
        ],
    )

  @mock.patch.object(time, "perf_counter", autospec=True)
  def test_device_span_begin_algorithm(self, mock_perf_counter):
    # [FUNCTION DOC]
    # Verifies a second device span starts at the previous device end (5) rather
    # than its host begin (4).
    mock_perf_counter.side_effect = [0.0, 2.0, 3.0, 5.0, 4.0, 6.0, 7.0]
    waitlist = mock_array()

    tracer = PerfTracer()

    with tracer.span("step1", devices=["tpu0"]) as span:
      span.device_end(waitlist)
    with tracer.span("step2", devices=["tpu0"]) as span:
      span.device_end(waitlist)

    tracer.synchronize()

    self.assertEqual(waitlist.block_until_ready.call_count, 2)
    self.assertListEqual(
        timelines_tostring(tracer),
        [
            (
                "thread: born=0.000000 stack=root\n"
                "- root (0.000000, inf)\n"
                "  - step1 (2.000000, 3.000000)\n"
                "  - step2 (4.000000, 6.000000)\n"
            ),
            (
                "tpu0: born=0.000000 stack=root\n"
                "- root (0.000000, inf)\n"
                "  - step1 (2.000000, 5.000000)\n"
                "  - step2 (5.000000, 7.000000)\n"  # begin is 5, not 4
            ),
        ],
    )

  @mock.patch.object(time, "perf_counter", autospec=True)
  def test_device_all_matcher(self, mock_perf_counter):
    # [FUNCTION DOC]
    # Verifies devices=tracer.all_devices targets every device, blocking once per
    # device.
    mock_perf_counter.side_effect = [0.0, 2.0, 3.0, 5.0, 5.0]
    waitlist = mock_array()

    tracer = PerfTracer(devices=["tpu0", "tpu1"])

    with tracer.span("x", devices=tracer.all_devices) as span:
      span.device_end(waitlist)

    tracer.synchronize()

    self.assertEqual(waitlist.block_until_ready.call_count, 2)
    self.assertListEqual(
        timelines_tostring(tracer),
        [
            (
                "thread: born=0.000000 stack=root\n"
                "- root (0.000000, inf)\n"
                "  - x (2.000000, 3.000000)\n"
            ),
            (
                "tpu0: born=0.000000 stack=root\n"
                "- root (0.000000, inf)\n"
                "  - x (2.000000, 5.000000)\n"
            ),
            (
                "tpu1: born=0.000000 stack=root\n"
                "- root (0.000000, inf)\n"
                "  - x (2.000000, 5.000000)\n"
            ),
        ],
    )

  @mock.patch.object(time, "perf_counter", autospec=True)
  def test_span_group_ok(self, mock_perf_counter):
    # [FUNCTION DOC]
    # Verifies nested span groups appear on thread and device timelines, with
    # device spans only on their target devices and groups mirrored on all.
    mock_perf_counter.side_effect = [
        0.0,
        1.0,
        2.0,
        3.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
    ]
    waitlist = mock_array()

    tracer = PerfTracer(["tpu0", "tpu1"])

    with tracer.span_group("outer_loop"):
      with tracer.span("step1", devices=["tpu0"]) as span:
        span.device_end(waitlist)
      with tracer.span_group("inner_loop"):
        with tracer.span("step2", devices=["tpu1"]) as span:
          span.device_end(waitlist)
      with tracer.span("step3"):
        pass

    tracer.synchronize()

    self.assertEqual(waitlist.block_until_ready.call_count, 2)
    self.assertListEqual(
        timelines_tostring(tracer),
        [
            (
                "thread: born=0.000000 stack=root\n"
                "- root (0.000000, inf)\n"
                "  - outer_loop (1.000000, 13.000000)\n"
                "    - step1 (2.000000, 3.000000)\n"
                "    - inner_loop (6.000000, 10.000000)\n"
                "      - step2 (7.000000, 8.000000)\n"
                "    - step3 (11.000000, 12.000000)\n"
            ),
            (
                "tpu0: born=0.000000 stack=root\n"
                "- root (0.000000, inf)\n"
                "  - outer_loop (1.000000, 13.000000)\n"
                "    - step1 (2.000000, 5.000000)\n"
                "    - inner_loop (6.000000, 10.000000)\n"
            ),
            (
                "tpu1: born=0.000000 stack=root\n"
                "- root (0.000000, inf)\n"
                "  - outer_loop (1.000000, 13.000000)\n"
                "    - inner_loop (6.000000, 10.000000)\n"
                "      - step2 (7.000000, 9.000000)\n"
            ),
        ],
    )

  def test_nested_span_raise_exception(self):
    # [FUNCTION DOC]
    # Verifies opening a span inside another span raises ValueError.
    tracer = PerfTracer()
    with tracer.span("outer"):
      with self.assertRaises(ValueError):
        with tracer.span("inner"):
          pass

  def test_noop_interface_is_same(self):
    # [FUNCTION DOC]
    # Verifies NoopTracer and PerfTracer expose identical public attribute names.
    noop_public_attrs = [
        name for name in dir(NoopTracer()) if not name.startswith("_")
    ]
    perf_public_attrs = [
        name for name in dir(PerfTracer()) if not name.startswith("_")
    ]
    self.assertEqual(noop_public_attrs, perf_public_attrs)

  def test_device_span_with_explicit_group(self):
    # [FUNCTION DOC]
    # Verifies DeviceTimeline.device_span with explicit group attaches to that
    # group, while without group it attaches to the innermost open group.
    timeline = DeviceTimeline("tpu0", 0.0)

    # Create nested structure: group1 > group2
    group1 = timeline.span_group_begin("group1", 0.5)
    group2 = timeline.span_group_begin("group2", 0.6)

    # Add a span to group1 explicitly
    timeline.device_span(
        "span_in_group1", thread_begin=0.7, end=0.8, group=group1
    )

    # Add a span implicitly (should go to group2)
    timeline.device_span("span_in_group2", thread_begin=0.9, end=1.0)

    with self.subTest("Verify group1 structure"):
      self.assertLen(group1.inner, 2)
      self.assertIs(group1.inner[0], group2)
      self.assertEqual(group1.inner[1].name, "span_in_group1")

    with self.subTest("Verify group2 structure"):
      self.assertLen(group2.inner, 1)
      self.assertEqual(group2.inner[0].name, "span_in_group2")

  @mock.patch.object(trace, "_async_wait", autospec=True)
  @mock.patch.object(time, "perf_counter", autospec=True)
  def test_device_span_async_parent_capture(
      self, mock_perf_counter, mock_async_wait
  ):
    # [FUNCTION DOC]
    # Verifies an asynchronously completed device span is attached to the group
    # active when it began (GroupA), not the group active when it completed
    # (GroupB). _async_wait is patched to capture the success callback, which is
    # invoked manually inside GroupB.
    mock_perf_counter.side_effect = [
        0.0,  # 1. __init__ (born)
        1.0,  # 2. span_group("GroupA") begin
        2.0,  # 3. span("Span1") begin (thread)
        3.0,  # 4. span("Span1") end (thread)
        4.0,  # 5. span_group("GroupB") begin
        5.0,  # 6. on_success callback for Span1 (device end time)
        6.0,  # 7. span_group("GroupB") end
        7.0,  # 8. span_group("GroupA") end
    ]
    waitlist = mock_array()

    captured_success_callback = []

    def fake_async_wait(waitlist, success, failure):
      # [FUNCTION DOC]
      # HELPER: fake _async_wait recording the success callback and returning an
      # autospec'd Thread mock.
      del waitlist, failure  # Unused
      captured_success_callback.append(success)
      return mock.create_autospec(threading.Thread, instance=True)

    mock_async_wait.side_effect = fake_async_wait

    tracer = PerfTracer(devices=["tpu0"])

    with tracer.span_group("GroupA"):
      # Span1 is initiated in GroupA.
      with tracer.span("Span1", devices=["tpu0"]) as span:
        span.device_end(waitlist)
      # Span1 is still active when we enter GroupB, and ends in GroupB.
      with tracer.span_group("GroupB"):
        if captured_success_callback:
          captured_success_callback[0]()

    device_timeline = tracer._get_or_create_device_timeline("tpu0")
    root = device_timeline.root

    with self.subTest("Root contains GroupA"):
      self.assertLen(root.inner, 1)
      group_a = root.inner[0]
      self.assertEqual(group_a.name, "GroupA")

    with self.subTest("GroupA contains Span1 and GroupB"):
      group_a_children_names = [child.name for child in group_a.inner]
      self.assertIn("Span1", group_a_children_names)
      self.assertIn("GroupB", group_a_children_names)

    with self.subTest("GroupB should NOT have Span1"):
      group_b = [child for child in group_a.inner if child.name == "GroupB"][0]
      self.assertEmpty(group_b.inner)


if __name__ == "__main__":
  absltest.main()
