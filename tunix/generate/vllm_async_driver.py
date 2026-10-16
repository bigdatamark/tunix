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
# Single-process driver for the vLLM V1 engine: keeps the LLMEngine (and its
# EngineCore) in the current process and runs the continuous-batching step
# loop on a background daemon Python thread, resolving one
# concurrent.futures.Future per submitted request when that request finishes.
# Intended for TPU/JAX setups where vLLM multiprocessing is undesirable.
# Import-time side effects: sets the environment variable
# VLLM_ENABLE_V1_MULTIPROCESSING to the string 0 (if not already 0) and sets
# vllm envs.VLLM_ENABLE_V1_MULTIPROCESSING to False, so any engine built in this
# process afterwards is in-process.
# Type aliases: StreamCallback is a callable taking a RequestOutput or
# PoolingRequestOutput and returning nothing; RequestFuture is a Future of
# either output type.
"""Single-process driver for the vLLM V1 EngineCore.

This driver keeps the EngineCore inside the current process and runs the
continuous batching loop on a Python thread. It is intended for TPU setups
where multiprocessing is undesirable (e.g. JAX integration).
"""

from __future__ import annotations

from concurrent.futures import Future
import os
import threading
import time
from typing import Any, Callable, Dict, Optional, Union

from absl import logging
from vllm import envs
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptType
from vllm.lora.request import LoRARequest
from vllm.outputs import PoolingRequestOutput
from vllm.outputs import RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.llm_engine import LLMEngine

# Ensure multiprocessing is disabled before the engine is constructed.
if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
  os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
envs.VLLM_ENABLE_V1_MULTIPROCESSING = False

StreamCallback = Callable[[Union[RequestOutput, PoolingRequestOutput]], None]
RequestFuture = Future[Union[RequestOutput, PoolingRequestOutput]]


class VLLMInProcessDriver:
  # [CLASS DOC]
  # Owns an LLMEngine and two daemon threads: a step loop that drives the engine
  # whenever there is unfinished work, and a stats logger calling do_log_stats
  # periodically. Thread safety: every engine call and every access to the
  # pending-futures dict is guarded by one lock. State: _pending maps request id
  # to its Future; _work_event signals new work; _stop_event signals shutdown;
  # _last_error stores the exception that killed the loop. Supports use as a
  # context manager (exit calls shutdown).
  """Runs a V1 LLMEngine in-process and polls for finished outputs."""

  def __init__(
      self,
      llm_engine: LLMEngine,
      *,
      poll_interval_s: float = 0.005,
      log_stats_interval_s: float = 10.0,
      stream_callback: Optional[StreamCallback] = None,
      auto_start: bool = True,
  ) -> None:
    # [FUNCTION DOC]
    # Initializes state with the given engine, poll interval (seconds, default
    # 0.005), stats logging interval (default 10 s), optional stream callback for
    # non-final outputs, and auto_start flag. Creates the lock, events, empty
    # pending dict, no threads and no error. If auto_start (default True), calls
    # start, which spawns the threads.
    self._llm_engine = llm_engine
    self._poll_interval_s = poll_interval_s
    self._stream_callback = stream_callback

    self._engine_lock = threading.Lock()
    self._work_event = threading.Event()
    self._stop_event = threading.Event()
    self._loop_thread: Optional[threading.Thread] = None
    self._log_thread: Optional[threading.Thread] = None
    self._log_stats_interval_s: float = log_stats_interval_s

    self._pending: Dict[str, RequestFuture] = {}
    self._last_error: Optional[Exception] = None

    if auto_start:
      self.start()

  @classmethod
  def from_engine_args(
      cls,
      engine_args: EngineArgs,
      *,
      usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
      poll_interval_s: float = 0.005,
      log_stats_interval_s: float = 1.0,
      stream_callback: Optional[StreamCallback] = None,
      auto_start: bool = True,
  ) -> "VLLMInProcessDriver":
    # [FUNCTION DOC]
    # Class method: builds an LLMEngine via LLMEngine.from_engine_args with the
    # given usage context and multiprocessing disabled, logs the args at debug
    # level, and returns a driver wrapping it with the poll interval, stream
    # callback and auto_start. Engine construction errors propagate.
    # KNOWN ISSUE: the log_stats_interval_s argument (default 1.0 here) is accepted
    # but not forwarded, so the instance always uses the constructor default of
    # 10 seconds.
    logging.debug(
        f"Creating VLLMInProcessDriver with engine_args: {engine_args} and"
        f" usage_context: {usage_context}"
    )
    llm_engine = LLMEngine.from_engine_args(
        engine_args,
        usage_context=usage_context,
        enable_multiprocessing=False,
    )
    return cls(
        llm_engine,
        poll_interval_s=poll_interval_s,
        stream_callback=stream_callback,
        auto_start=auto_start,
    )

  def submit_request(
      self,
      request_id: str,
      prompt: Union[EngineCoreRequest, PromptType],
      params: Union[SamplingParams, PoolingParams],
      *,
      arrival_time: Optional[float] = None,
      lora_request: Optional[LoRARequest] = None,
      tokenization_kwargs: Optional[dict[str, Any]] = None,
      trace_headers: Optional[dict[str, str]] = None,
      priority: int = 0,
  ) -> RequestFuture:
    # [FUNCTION DOC]
    # Submits a request and returns a Future resolved with the final output.
    # Under the lock: raises ValueError if the id is already pending; registers a
    # new Future; logs; calls engine add_request with request id, prompt (a prompt
    # or a prebuilt EngineCoreRequest), sampling or pooling params, arrival time,
    # LoRA request, tokenization kwargs, trace headers and priority; sets the work
    # event to wake the loop.
    # KNOWN ISSUE: if add_request raises, the Future is left registered in
    # _pending (never resolved) and the same id can no longer be resubmitted.
    future: RequestFuture = Future()
    with self._engine_lock:
      if request_id in self._pending:
        raise ValueError(f"Request {request_id} already pending.")
      self._pending[request_id] = future
      logging.debug(
          f"VLLMInProcessDriver submitting request {request_id} with prompt"
          f" {prompt} and sampling params {params} to vLLM engine."
      )
      self._llm_engine.add_request(
          request_id=request_id,
          prompt=prompt,
          params=params,
          arrival_time=arrival_time,
          lora_request=lora_request,
          tokenization_kwargs=tokenization_kwargs,
          trace_headers=trace_headers,
          priority=priority,
      )
      self._work_event.set()
    return future

  def start(self) -> None:
    # [FUNCTION DOC]
    # Starts the background threads unless the loop thread is already alive.
    # Clears the stop event, then starts daemon threads named
    # VLLMInProcessDriverLoop (running the step loop) and VLLMLogStats (running
    # the stats loop).
    # KNOWN ISSUE: if the loop thread died (e.g. after an error) while the log
    # thread is still alive, a second log thread is started alongside the old one.
    if self._loop_thread and self._loop_thread.is_alive():
      return
    self._stop_event.clear()
    self._loop_thread = threading.Thread(
        target=self._loop, name="VLLMInProcessDriverLoop", daemon=True
    )
    self._loop_thread.start()

    self._log_thread = threading.Thread(
        target=self._log_loop, name="VLLMLogStats", daemon=True
    )
    self._log_thread.start()

  def _log_loop(self) -> None:
    # [FUNCTION DOC]
    # HELPER: thread body. Until the stop event is set, calls engine
    # do_log_stats (logging and swallowing any exception) and then waits on the
    # stop event for the stats interval. Note it calls the engine without holding
    # the engine lock.
    while not self._stop_event.is_set():
      try:
        self._llm_engine.do_log_stats()
      except Exception:  # pylint: disable=broad-exception-caught
        logging.exception("log_stats failed")
      self._stop_event.wait(self._log_stats_interval_s)

  def cancel(self, request_id: str) -> None:
    # [FUNCTION DOC]
    # Cancels a request under the lock: removes its Future from pending and
    # cancels it if not done; tells the engine to abort the request id; clears
    # the work event if the engine has no unfinished requests. Unknown ids are
    # tolerated.
    with self._engine_lock:
      future = self._pending.pop(request_id, None)
      if future is not None and not future.done():
        future.cancel()
      self._llm_engine.abort_request([request_id])
      if not self._llm_engine.has_unfinished_requests():
        self._work_event.clear()

  def shutdown(self) -> None:
    # [FUNCTION DOC]
    # Stops the threads (via stop), then fails every still-pending Future with
    # RuntimeError (message Driver shut down), and finally shuts down the engine
    # core under the lock. Not idempotent-safe if the engine core shutdown fails
    # on a second call.
    self.stop()
    with self._engine_lock:
      pending = list(self._pending.values())
      self._pending.clear()
    for future in pending:
      if not future.done():
        future.set_exception(RuntimeError("Driver shut down."))
    with self._engine_lock:
      self._llm_engine.engine_core.shutdown()

  def stop(self) -> None:
    # [FUNCTION DOC]
    # Sets the stop and work events (to wake any waiter), then joins and clears
    # the loop thread and the log thread if they exist. Blocks until the current
    # engine step and stats wait finish.
    self._stop_event.set()
    self._work_event.set()
    if self._loop_thread is not None:
      self._loop_thread.join()
      self._loop_thread = None
    if self._log_thread is not None:
      self._log_thread.join()
      self._log_thread = None

  def pause(self) -> None:
    # [FUNCTION DOC]
    # Placeholder; always raises RuntimeError (pause not implemented).
    raise RuntimeError("Pause feature WIP")

  def resume(self) -> None:
    # [FUNCTION DOC]
    # Placeholder; always raises RuntimeError (resume not implemented).
    raise RuntimeError("Resume feature WIP")

  def _loop(self) -> None:
    # [FUNCTION DOC]
    # HELPER: main loop thread body. Until stop is requested: wait for work
    # (skip iteration if none); run one engine step; log the output request ids
    # every 40th iteration at debug level; dispatch each output to the output
    # handler, or sleep for the poll interval when the step yielded nothing.
    # Any exception ends the loop and is passed to the error recorder.
    # KNOWN ISSUE: after an error the loop thread exits permanently; later
    # submissions are accepted but never processed (their Futures hang) unless
    # start is called again.
    try:
      while not self._stop_event.is_set():
        if not self._wait_for_work():
          continue
        outputs = self._step_engine()
        logging.log_every_n(
            logging.DEBUG,
            "VLLMInProcessDriver loop step outputs:"
            f" {[output.request_id for output in outputs]}",
            40,
        )
        if outputs:
          for output in outputs:
            self._handle_output(output)
        else:
          time.sleep(self._poll_interval_s)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      self._record_error(exc)

  def _wait_for_work(self) -> bool:
    # [FUNCTION DOC]
    # HELPER: returns True immediately if the engine has unfinished requests
    # (checked under the lock). Otherwise clears the work event, waits up to the
    # poll interval for it to be set, and returns True unless stopping (a True
    # return then only means another check should be made).
    with self._engine_lock:
      has_work = self._llm_engine.has_unfinished_requests()
      if has_work:
        return True
      self._work_event.clear()

    self._work_event.wait(timeout=self._poll_interval_s)
    return not self._stop_event.is_set()

  def _step_engine(
      self,
  ) -> list[Union[RequestOutput, PoolingRequestOutput]]:
    # [FUNCTION DOC]
    # HELPER: under the lock, runs one engine step if there are unfinished
    # requests and returns its list of outputs, else returns an empty list. Logs
    # the unfinished request count every 100 calls at debug level.
    logging.log_every_n(
        logging.DEBUG,
        f"VLLMInProcessDriver loop waking up to process one step of requests.",
        100,
    )
    with self._engine_lock:
      logging.log_every_n(
          logging.DEBUG,
          "VLLMInProcessDriver has"
          f" {self._llm_engine.get_num_unfinished_requests()} pending"
          " requests.",
          100,
      )
      if self._llm_engine.has_unfinished_requests():
        return self._llm_engine.step()
      return []

  def _handle_output(
      self, output: Union[RequestOutput, PoolingRequestOutput]
  ) -> None:
    # [FUNCTION DOC]
    # HELPER: routes one engine output. Unfinished outputs are passed to the
    # stream callback if set (finished outputs are not). Finished outputs pop the
    # matching Future under the lock and, unless missing or already done
    # (e.g. cancelled), set it to the output.
    if not output.finished:
      callback = self._stream_callback
      if callback is not None:
        callback(output)
      return
    with self._engine_lock:
      future = self._pending.pop(output.request_id, None)
    if future is None or future.done():
      return
    logging.debug(
        f"VLLMInProcessDriver completed request id: {output.request_id}."
    )
    future.set_result(output)

  def _record_error(self, exc: Exception) -> None:
    # [FUNCTION DOC]
    # HELPER: logs the error, stores it as last_error, removes all pending
    # Futures under the lock, and sets the exception on each one not yet done.
    logging.debug("VLLMInProcessDriver encountered an error: %s", exc)
    self._last_error = exc
    with self._engine_lock:
      pending = list(self._pending.values())
      self._pending.clear()
    for future in pending:
      if not future.done():
        future.set_exception(exc)

  @property
  def llm_engine(self) -> LLMEngine:
    # [FUNCTION DOC]
    # Read-only property returning the wrapped LLMEngine.
    return self._llm_engine

  @property
  def last_error(self) -> Optional[Exception]:
    # [FUNCTION DOC]
    # Read-only property returning the exception that terminated the loop, or
    # None.
    return self._last_error

  def __enter__(self) -> "VLLMInProcessDriver":
    # [FUNCTION DOC]
    # Context manager entry; returns the driver itself (threads already started
    # if auto_start was True).
    return self

  def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001, ANN201
    # [FUNCTION DOC]
    # Context manager exit; calls shutdown and does not suppress exceptions.
    self.shutdown()
