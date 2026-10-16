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
# Unit tests for tunix.perf.experimental.timeline (perf v2): Span (name,
# begin, id, end defaulting to infinity, ended/duration, tags with add_tag
# warning on overwrite, repr with tags), Timeline (stack-based
# start_span/stop_span with parent ids, cur_step as dict id -> span,
# commit_step moving completed spans into committed_steps and purging
# uncompleted ones with a warning, all_steps, repr), AsyncTimeline (span()
# records a span completed when its waitlist is ready, via _async_wait
# running jax.block_until_ready in a thread; the async span's parent is the
# innermost open span; failures are logged, not raised; wait_pending_spans
# joins and clears threads) and BatchAsyncTimelines (broadcasts span()).
import threading
import time
from unittest import mock
from absl.testing import absltest
from tunix.perf.experimental import constants
from tunix.perf.experimental import timeline


class SpanTest(absltest.TestCase):
  # [CLASS DOC]
  # Test case for timeline.Span.

  def test_span(self):
    # [FUNCTION DOC]
    # Verifies a new Span has the given name/begin, infinite end, ended False
    # and infinite duration.
    s = timeline.Span(name="test", begin=1.0, id=0)
    self.assertEqual(s.name, "test")
    self.assertEqual(s.begin, 1.0)
    self.assertEqual(s.end, float("inf"))
    self.assertFalse(s.ended)
    self.assertEqual(s.duration, float("inf"))

  def test_span_with_tags(self):
    # [FUNCTION DOC]
    # Verifies tags passed at construction are stored and shown in repr.
    tags_dict = {constants.STEP: 1, "custom_tag": "value"}
    s = timeline.Span(name="test_tags", begin=1.0, id=0, tags=tags_dict)
    self.assertEqual(s.tags, tags_dict)
    self.assertIn("tags=", repr(s))
    self.assertIn("step", repr(s))

  def test_add_tag(self):
    # [FUNCTION DOC]
    # Verifies add_tag inserts tags (constants.STEP is the key 'step').
    s = timeline.Span(name="test_add_tag", begin=1.0, id=0)
    s.add_tag("foo", "bar")
    self.assertEqual(s.tags, {"foo": "bar"})
    s.add_tag(constants.STEP, 100)
    self.assertEqual(s.tags, {"foo": "bar", "step": 100})

  def test_add_tag_overwrite_warning(self):
    # [FUNCTION DOC]
    # Verifies overwriting a tag replaces the value and logs a WARNING naming
    # old and new values.
    s = timeline.Span(name="test_add_tag_overwrite", begin=1.0, id=0)
    s.add_tag("foo", "bar")
    with self.assertLogs(level="WARNING") as cm:
      s.add_tag("foo", "baz")
    self.assertEqual(s.tags, {"foo": "baz"})
    self.assertTrue(
        any(
            "Tag 'foo' already exists with value 'bar'. Overwriting with 'baz'."
            in o
            for o in cm.output
        )
    )


class TimelineTest(absltest.TestCase):
  # [CLASS DOC]
  # Test case for synchronous timeline.Timeline behaviour.

  def test_commit_step(self):
    # [FUNCTION DOC]
    # Verifies commit_step keeps the completed span, purges the open one with a
    # 'Purging uncompleted span' warning, clears cur_step, and all_steps lists
    # committed steps plus the (empty) current step.
    t = timeline.Timeline("test_tl", 0.0)

    # Start and stop a span
    s1 = t.start_span("span1", 1.0)
    t.stop_span(2.0)

    # Start an uncompleted span
    s2 = t.start_span("span2", 3.0)

    with self.subTest("Pre-commit state"):
      self.assertLen(t.cur_step, 2)
      self.assertEmpty(t.committed_steps)

    # Commit step
    with self.assertLogs(level="WARNING") as cm:
      t.commit_step()

    with self.subTest("Post-commit purging"):
      # Verify uncompleted span was purged
      self.assertTrue(
          any("Purging uncompleted span 'span2'" in o for o in cm.output)
      )

    with self.subTest("Committed steps"):
      self.assertLen(t.committed_steps, 1)
      self.assertIn(s1.id, t.committed_steps[0])
      self.assertNotIn(s2.id, t.committed_steps[0])

    with self.subTest("All steps and current step"):
      self.assertEmpty(t.cur_step)
      self.assertLen(t.all_steps, 2)
      self.assertEqual(t.all_steps[0], t.committed_steps[0])
      self.assertEqual(t.all_steps[1], {})

  def test_basic_span_lifecycle(self):
    # [FUNCTION DOC]
    # Verifies start_span returns span id 0 with no parent and stop_span sets
    # end and ended.
    t = timeline.Timeline("test_tl", 100.0)
    s = t.start_span("span1", 101.0)
    with self.subTest("Span started"):
      self.assertEqual(s.name, "span1")
      self.assertEqual(s.begin, 101.0)
      self.assertEqual(s.id, 0)
      self.assertIsNone(s.parent_id)
      self.assertFalse(s.ended)

    t.stop_span(102.0)
    with self.subTest("Span stopped"):
      self.assertTrue(s.ended)
      self.assertEqual(s.end, 102.0)

  def test_nested_spans(self):
    # [FUNCTION DOC]
    # Verifies nested spans get parent_id of the enclosing span and stop_span
    # ends the innermost span first.
    t = timeline.Timeline("test_tl", 0.0)
    s1 = t.start_span("root", 1.0)
    s2 = t.start_span("child", 2.0)

    self.assertEqual(s2.parent_id, s1.id)

    t.stop_span(3.0)  # stops s2
    self.assertEqual(s2.end, 3.0)

    t.stop_span(4.0)  # stops s1
    self.assertEqual(s1.end, 4.0)

  def test_stop_span_error_cases(self):
    # [FUNCTION DOC]
    # Verifies stop_span raises ValueError when no span is open, and when the end
    # time precedes the span's begin.
    t = timeline.Timeline("test_tl", 0.0)
    with self.assertRaisesRegex(ValueError, "no more spans to end"):
      t.stop_span(1.0)

    s = t.start_span("s1", 2.0)
    # End before begin
    with self.assertRaisesRegex(ValueError, "ended at .* before it began"):
      t.stop_span(1.0)

  def test_nested_timeline_with_tags_repr(self):
    # [FUNCTION DOC]
    # Verifies the exact repr: header with id and born, 'Current Step -0:' and
    # one line per span with times relative to born, parent and tags.
    born = 1000.0
    t = timeline.Timeline("test_tl", born)

    # Start root
    s_root = t.start_span("root", born + 1.0)
    s_root.add_tag("type", "root_span")

    # Start nested
    s_child = t.start_span("child", born + 2.0)
    s_child.add_tag("iter", 1)

    # Stop nested
    t.stop_span(born + 3.0)

    # Stop root
    t.stop_span(born + 4.0)

    # Check tags are stored correctly
    self.assertEqual(s_root.tags, {"type": "root_span"})
    self.assertEqual(s_child.tags, {"iter": 1})

    # Check full repr string
    expected_repr = (
        f"Timeline(test_tl, {born:.6f})\n"
        "Current Step -0:\n"
        "  [0] root: 1.000000, 4.000000, tags={'type': 'root_span'}\n"
        "  [1] child: 2.000000, 3.000000 (parent=0), tags={'iter': 1}\n"
    )
    self.assertEqual(repr(t), expected_repr)


class AsyncTimelineTest(absltest.TestCase):
  # [CLASS DOC]
  # Test case for AsyncTimeline with timeline._async_wait patched (by default
  # calling success immediately and returning a Thread mock).

  def setUp(self):
    # [FUNCTION DOC]
    # HELPER: fixture starting the _async_wait patch with immediate success.
    # Side effect: patches the module attribute until tearDown.
    self.patcher = mock.patch.object(timeline, "_async_wait")
    self.mock_async_wait = self.patcher.start()

    # Setup mock behavior for _async_wait to immediately succeed by default
    def default_wait(waitlist, success, failure):
      # [FUNCTION DOC]
      # HELPER: fake wait invoking success at once; returns an autospec'd Thread.
      success()
      return mock.create_autospec(threading.Thread, instance=True)

    self.mock_async_wait.side_effect = default_wait

  def tearDown(self):
    # [FUNCTION DOC]
    # HELPER: stops the _async_wait patch.
    self.patcher.stop()

  def test_span_success(self):
    # [FUNCTION DOC]
    # Verifies span() with a waitlist calls _async_wait once and records an
    # ended span with the given name and begin.
    t = timeline.AsyncTimeline("dev", 0.0)
    waitlist = ["thing"]

    t.span("async_op", 1.0, waitlist)

    self.mock_async_wait.assert_called_once()
    self.assertLen(t.cur_step, 1)
    s = t.cur_step[0]
    self.assertEqual(s.name, "async_op")
    self.assertEqual(s.begin, 1.0)
    self.assertTrue(s.ended)  # Ended because mock calls success immediately

  def test_span_with_no_waitlist(self):
    # [FUNCTION DOC]
    # Verifies an empty waitlist records an ended span without _async_wait.
    t = timeline.AsyncTimeline("dev", 0.0)
    t.span("immediate", 1.0, [])
    self.mock_async_wait.assert_not_called()
    self.assertLen(t.cur_step, 1)
    self.assertTrue(t.cur_step[0].ended)

  def test_delayed_completion(self):
    # [FUNCTION DOC]
    # Verifies a span is recorded only when the success callback fires, with end
    # taken from time.perf_counter at that moment (patched to 5.0).
    t = timeline.AsyncTimeline("dev", 0.0)

    # Capture callbacks
    callbacks = {}

    def capture_wait(waitlist, success, failure):
      # [FUNCTION DOC]
      # HELPER: fake wait capturing success/failure callbacks without calling them.
      callbacks["success"] = success
      callbacks["failure"] = failure
      return mock.create_autospec(threading.Thread, instance=True)

    self.mock_async_wait.side_effect = capture_wait

    t.span("delayed", 1.0, ["wait"])
    self.assertEmpty(t.cur_step)  # Not yet recorded

    # Simulate completion
    with mock.patch.object(time, "perf_counter", return_value=5.0):
      callbacks["success"]()

    self.assertLen(t.cur_step, 1)
    s = t.cur_step[0]
    self.assertEqual(s.end, 5.0)

  def test_nested_async_span_parent(self):
    # [FUNCTION DOC]
    # Verifies an async span's parent_id is the innermost open sync span.
    t = timeline.AsyncTimeline("dev", 0.0)

    # Start sync spans to populate the stack with multiple items
    s0 = t.start_span("root", 1.0)
    s1 = t.start_span("child1", 2.0)
    s2 = t.start_span("child2", 3.0)

    # Now create an async span
    t.span("async_op", 4.0, ["wait"])

    # Find the async span.
    async_s = None
    for span in t.cur_step.values():
      if span.name == "async_op":
        async_s = span
        break

    self.assertIsNotNone(async_s)
    self.assertEqual(async_s.parent_id, s2.id)

    # Protect against index hardcoding bugs (e.g. self._spans_stack[1])
    self.assertNotEqual(async_s.parent_id, s1.id)

  def test_failure(self):
    # [FUNCTION DOC]
    # Verifies a failure callback logs via logging.error with the format
    # "Timeline span '%s' (id=%d) failed: %s", name, id, error and exc_info,
    # without raising.
    t = timeline.AsyncTimeline("dev", 0.0)

    def fail_wait(waitlist, success, failure):
      # [FUNCTION DOC]
      # HELPER: fake wait invoking failure with RuntimeError('failed').
      failure(RuntimeError("failed"))
      return mock.create_autospec(threading.Thread, instance=True)

    self.mock_async_wait.side_effect = fail_wait

    with mock.patch.object(timeline.logging, "error") as mock_log_error:
      t.span("failed_op", 1.0, ["wait"])
      # Exception is caught and logged, no exception is raised to the caller.
      mock_log_error.assert_called_once()
      args, kwargs = mock_log_error.call_args
      format_str, name, span_id, err = args
      self.assertEqual(format_str, "Timeline span '%s' (id=%d) failed: %s")
      self.assertEqual(name, "failed_op")
      self.assertEqual(span_id, 0)
      self.assertIsInstance(err, RuntimeError)
      self.assertEqual(str(err), "failed")
      self.assertIn("exc_info", kwargs)
      self.assertIsInstance(kwargs["exc_info"], RuntimeError)

  def test_wait_pending_spans_clears_threads(self):
    # [FUNCTION DOC]
    # Verifies wait_pending_spans clears the tracked waiter threads.
    t = timeline.AsyncTimeline("test_tl", 0.0)
    t.span("s1", 1.0, ["wait"])
    t.span("s2", 2.0, ["wait"])

    self.assertLen(t._threads, 2)
    t.wait_pending_spans()
    self.assertLen(t._threads, 0)

  def test_async_wait_helper(self):
    # [FUNCTION DOC]
    # Verifies the real _async_wait starts a thread that calls
    # jax.block_until_ready(waitlist) (patched) then success, not failure.
    # Temporarily stops the fixture patch and restarts it afterwards.
    # Temporarily unpatch _async_wait to test the real implementation
    self.patcher.stop()
    try:
      waitlist = ["data"]
      success_cb = mock.create_autospec(lambda: None)
      failure_cb = mock.create_autospec(lambda e: None)

      # Mock jax.block_until_ready to avoid actual JAX calls
      with mock.patch.object(timeline.jax, "block_until_ready") as mock_block:
        t = timeline._async_wait(waitlist, success_cb, failure_cb)

        # Verify it returned a thread and started it
        with self.subTest("Thread execution"):
          self.assertIsInstance(t, threading.Thread)
          self.assertIsNotNone(t.ident)
        t.join()

        with self.subTest("Callbacks and JAX integration"):
          mock_block.assert_called_once_with(waitlist)
          success_cb.assert_called_once()
          failure_cb.assert_not_called()
    finally:
      self.patcher.start()


class BatchAsyncTimelinesTest(absltest.TestCase):
  # [CLASS DOC]
  # Test case for BatchAsyncTimelines.

  def test_span_broadcast(self):
    # [FUNCTION DOC]
    # Verifies span() is forwarded with identical args and tags to every
    # contained AsyncTimeline.
    t1 = mock.create_autospec(timeline.AsyncTimeline, instance=True)
    t2 = mock.create_autospec(timeline.AsyncTimeline, instance=True)
    batch = timeline.BatchAsyncTimelines([t1, t2])

    waitlist = ["thing"]
    tags = {"foo": "bar"}
    batch.span("test_span", 100.0, waitlist, tags=tags)

    t1.span.assert_called_once_with("test_span", 100.0, waitlist, tags=tags)
    t2.span.assert_called_once_with("test_span", 100.0, waitlist, tags=tags)


if __name__ == "__main__":
  absltest.main()
