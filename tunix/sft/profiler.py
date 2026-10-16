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
# Step-windowed JAX profiler control for Tunix trainers.
# Defines ProfilerOptions (frozen config) and Profiler, which starts a JAX
#   trace (jax.profiler.start_trace) when the trainer reaches a configured
#   first step and stops it at a configured last step, only on process 0,
#   and ensures at most one profiler is active per Python process via a
#   class-level lock and flag. No side effects at import.
"""Profiler class for Tunix trainers."""

import dataclasses
import threading

from absl import logging
import jax


@dataclasses.dataclass(frozen=True)
class ProfilerOptions:
  # [CLASS DOC]
  # Frozen dataclass of profiling settings: log_dir (trace output dir),
  # skip_first_n_steps (steps after the initial step before starting),
  # profiler_steps (number of steps to trace), set_profile_options (if True
  # pass a jax.profiler.ProfileOptions with the tracer levels below),
  # host_tracer_level (default 2; captures HBM profiles), python_tracer_level
  # (default 1).
  # Directory to write the profile to.
  log_dir: str
  # Number of steps to skip before profiling.
  skip_first_n_steps: int
  # Number of steps to profile.
  profiler_steps: int
  # Whether to set the profile options.
  set_profile_options: bool = True
  # https://github.com/jax-ml/jax/blob/0b1b909dd66a113ee0d7e54e55d0efef480e2a8a/docs/profiling.md?plain=1#L285
  host_tracer_level: int = 2  # set to 2 to capture HBM profiles.
  # https://github.com/jax-ml/jax/blob/0b1b909dd66a113ee0d7e54e55d0efef480e2a8a/docs/profiling.md?plain=1#L300
  python_tracer_level: int = 1


class Profiler:
  # [CLASS DOC]
  # Profiler window controller. Class state: _lock (threading.Lock) and
  # _is_active (bool) shared across all instances. Instance state:
  # _do_not_profile, _profiler_options, _output_path, _first_profile_step
  # (= initial_step + skip_first_n_steps), _last_profile_step (=
  # first + profiler_steps, capped at max_step), _started_by_this_instance.
  # The trainer calls maybe_activate(step) and maybe_deactivate(step) every
  # step; only exact equality with the window bounds triggers actions.
  """Activate/deactivate a profiler based on the ProfilerOptions."""

  _lock = threading.Lock()
  _is_active: bool = False

  def __init__(
      self,
      initial_step: int,
      max_step: int | None,
      profiler_options: ProfilerOptions | None,
  ):
    # [FUNCTION DOC]
    # Inputs: initial_step (step training resumes from), max_step (optional
    #   cap), profiler_options (None disables).
    # Behavior: if not process 0 or options is None, set _do_not_profile
    #   True and return early (other attributes unset). Otherwise store
    #   options/log_dir, compute first and last profile steps (see class),
    #   and raise ValueError if first >= last. Set
    #   _started_by_this_instance False. No side effects.
    # KNOWN ISSUE: the error message says "cannot be greater than" but
    #   equality also raises.
    if jax.process_index() != 0 or profiler_options is None:
      self._do_not_profile = True
      return
    self._profiler_options = profiler_options
    self._do_not_profile = False
    self._output_path = profiler_options.log_dir
    # This is step number, starting from 0.
    self._first_profile_step = (
        initial_step + profiler_options.skip_first_n_steps
    )
    # This is step number + 1, starting from 1.
    self._last_profile_step = self._set_last_profile_step(
        profiler_options.profiler_steps, max_step
    )
    # We use >= instead of > because last_profile_step is step number + 1.
    if self._first_profile_step >= self._last_profile_step:
      raise ValueError(
          f"First profile step {self._first_profile_step} cannot be greater"
          f" than the last profile step {self._last_profile_step}."
      )
    self._started_by_this_instance = False

  def maybe_activate(self, step: int):
    # [FUNCTION DOC]
    # Input: step. No-op unless profiling is enabled and step equals
    #   _first_profile_step. Under the class lock: if another profiler is
    #   active, log a warning and return; otherwise log, start the JAX trace
    #   to _output_path (with ProfileOptions carrying host/python tracer
    #   levels if set_profile_options, else default options), and set the
    #   class active flag and this instance's started flag.
    # Exceptions: from jax.profiler.start_trace. Side effects: starts trace.
    """Start the profiler."""
    if self._do_not_profile or step != self._first_profile_step:
      return
    with Profiler._lock:
      if Profiler._is_active:
        logging.warning(
            "A JAX profiler is already active. Skipping activation of this"
            " profiler."
        )
        return
      logging.info("Starting JAX profiler at step %d.", step)
      if self._profiler_options.set_profile_options:
        profile_options = jax.profiler.ProfileOptions()
        profile_options.host_tracer_level = (
            self._profiler_options.host_tracer_level
        )
        profile_options.python_tracer_level = (
            self._profiler_options.python_tracer_level
        )
        jax.profiler.start_trace(
            log_dir=self._output_path, profiler_options=profile_options
        )
      else:
        jax.profiler.start_trace(log_dir=self._output_path)
      Profiler._is_active = True
      self._started_by_this_instance = True

  def maybe_deactivate(self, step: int):
    # [FUNCTION DOC]
    # Input: step. No-op unless profiling is enabled and step equals
    #   _last_profile_step. Under the lock: if this instance did not start
    #   the trace, warn and return; else log, stop the trace, clear the
    #   class active flag and this instance's started flag.
    # Exceptions: from jax.profiler.stop_trace. Side effects: writes trace.
    """End the profiler."""
    if self._do_not_profile or step != self._last_profile_step:
      return
    with Profiler._lock:
      if not self._started_by_this_instance:
        logging.warning(
            "This profiler instance did not start JAX profiler. Skipping"
            " deactivation."
        )
        return
      logging.info("Stopping JAX profiler at step %d.", step)
      jax.profiler.stop_trace()
      Profiler._is_active = False
      self._started_by_this_instance = False

  def _set_last_profile_step(self, profiler_steps, max_step):
    # [FUNCTION DOC]
    # HELPER: Inputs: profiler_steps, max_step. Returns
    #   _first_profile_step + profiler_steps, or the min of that and
    #   max_step when max_step is not None. Requires _first_profile_step set.
    calculated_last_step = self._first_profile_step + profiler_steps
    if max_step is None:
      return calculated_last_step
    return min(calculated_last_step, max_step)
