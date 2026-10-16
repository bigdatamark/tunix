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
# Tests for tunix.generate.vllm_async_driver.VLLMInProcessDriver using a fake in-process LLM engine.
# Verifies that request futures resolve as the engine completes them (in completion order, not
# submission order) and that the background log thread periodically calls the engine's do_log_stats.
# Inputs: none. Outputs: test results. Side effects: starts driver background threads (shut down
# via test cleanup).

import threading
import time
from typing import Iterable

from absl.testing import absltest
from tunix.generate.vllm_async_driver import VLLMInProcessDriver


# TODO(b/453660461): Add extensive concurrency tests.


class _DummyCompletionOutput:
  # [CLASS DOC]
  # HELPER: stand-in for a vLLM CompletionOutput.

  def __init__(self, request_id: str):
    # [FUNCTION DOC]
    # HELPER: stores token_ids as a one-element list holding the request id, logprobs as a single 0.0,
    # and text as "response_for_" followed by the request id.
    self.token_ids = [request_id]
    self.logprobs = [0.0]
    self.text = f"response_for_{request_id}"


class _DummyRequestOutput:
  # [CLASS DOC]
  # HELPER: stand-in for a vLLM RequestOutput carrying one finished completion.

  def __init__(self, request_id: str):
    # [FUNCTION DOC]
    # HELPER: sets request_id, prompt None, prompt_token_ids as a list with the id, prompt_logprobs None,
    # outputs as one dummy completion for the id, finished True, kv_transfer_params None,
    # num_cached_tokens 0 and metrics None.
    self.request_id = request_id
    self.prompt = None
    self.prompt_token_ids = [request_id]
    self.prompt_logprobs = None
    self.outputs = [_DummyCompletionOutput(request_id)]
    self.finished = True
    self.kv_transfer_params = None
    self.num_cached_tokens = 0
    self.metrics = None


class _StubEngineCore:
  # [CLASS DOC]
  # HELPER: stub engine core exposing only a no-op shutdown.

  def shutdown(self):
    # [FUNCTION DOC]
    # HELPER: no-op shutdown.
    pass


class _FakeLLMEngine:
  # [CLASS DOC]
  # HELPER: thread-safe fake LLMEngine. Holds a fixed completion order and a pending list guarded by a
  # lock; step() completes only the next request in the order once it has been submitted. Exposes an
  # engine_core stub and a threading Event log_called set when stats are logged.
  """Minimal synchronous engine that emits completions in a fixed order."""

  def __init__(self, completion_order: Iterable[str]):
    # [FUNCTION DOC]
    # HELPER: input completion_order (iterable of request id strings), copied to a list. Initialises an
    # empty pending list, a lock, a stub engine_core and an unset log_called event.
    self._completion_order = list(completion_order)
    self._pending: list[str] = []
    self._lock = threading.Lock()
    self.engine_core = _StubEngineCore()
    self.log_called = threading.Event()

  # The driver only exercises a subset of the LLMEngine surface.
  def add_request(self, request_id: str, *_, **__):
    # [FUNCTION DOC]
    # HELPER: appends request_id to the pending list under the lock; other arguments are ignored.
    with self._lock:
      self._pending.append(request_id)

  def has_unfinished_requests(self) -> bool:
    # [FUNCTION DOC]
    # HELPER: returns True if any request is pending (under the lock).
    with self._lock:
      return bool(self._pending)

  def get_num_unfinished_requests(self) -> int:
    # [FUNCTION DOC]
    # HELPER: returns the number of pending requests (under the lock).
    with self._lock:
      return len(self._pending)

  def step(self):
    # [FUNCTION DOC]
    # HELPER: one engine step. Under the lock: if there is no remaining order or nothing pending, return
    # an empty list; if the next request in the order has not been submitted yet, sleep 1 ms and return an
    # empty list; otherwise pop it from the order and the pending list. Returns a list with one dummy
    # request output for that id.
    with self._lock:
      if not self._completion_order or not self._pending:
        return []

      next_request = self._completion_order[0]
      if next_request not in self._pending:
        # Wait for the next request in the completion order to arrive.
        time.sleep(0.001)
        return []

      self._completion_order.pop(0)
      self._pending.remove(next_request)

    return [_DummyRequestOutput(next_request)]

  def abort_request(self, *_args, **_kwargs):
    # [FUNCTION DOC]
    # HELPER: no-op abort that accepts any arguments.
    pass

  # Log stats API exercised by the driver's log thread.
  def do_log_stats(self):
    # [FUNCTION DOC]
    # HELPER: sets the log_called event and returns None.
    # Signal that the log stats method was called.
    self.log_called.set()
    return None


class VllmDriverAsyncTest(absltest.TestCase):
  # [CLASS DOC]
  # Tests concurrency behavior of VLLMInProcessDriver.

  def test_out_of_order_completions_preserved(self):
    # [FUNCTION DOC]
    # Verifies that ten requests submitted in order req-0..req-9 all resolve (within 5 s each), that
    # done-callbacks fire in the fake engine's scrambled completion order, and that this order differs
    # from submission order.
    request_ids = [f"req-{i}" for i in range(10)]
    completion_order = [
        "req-0",
        "req-3",
        "req-1",
        "req-7",
        "req-2",
        "req-9",
        "req-4",
        "req-6",
        "req-5",
        "req-8",
    ]

    engine = _FakeLLMEngine(completion_order)
    driver = VLLMInProcessDriver(llm_engine=engine, auto_start=True)
    self.addCleanup(driver.shutdown)

    finished_order: list[str] = []
    futures = []
    for request_id in request_ids:
      future = driver.submit_request(
          request_id=request_id,
          prompt={"prompt_token_ids": [1]},
          params=object(),
      )
      future.add_done_callback(
          lambda f: finished_order.append(f.result().request_id)
      )
      futures.append(future)

    results = [future.result(timeout=5.0) for future in futures]

    # Ensure all requests completed.
    self.assertCountEqual(
        [res.request_id for res in results],
        request_ids,
    )

    # All completions should be observed, but not necessarily in submit order.
    self.assertEqual(finished_order, completion_order)
    self.assertNotEqual(finished_order, request_ids)

  def test_log_thread_calls_do_log_stats(self):
    # [FUNCTION DOC]
    # Verifies that with log_stats_interval_s of 0.01 the driver's log thread calls do_log_stats within
    # one second.
    engine = _FakeLLMEngine([])
    driver = VLLMInProcessDriver(
        llm_engine=engine, log_stats_interval_s=0.01, auto_start=True
    )
    self.addCleanup(driver.shutdown)

    # Wait for the log thread to call into the engine's do_log_stats.
    self.assertTrue(engine.log_called.wait(timeout=1.0))


if __name__ == "__main__":
  absltest.main()
