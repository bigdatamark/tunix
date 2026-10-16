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
# Builders for perf metrics export functions (v1) for the GRPO RL workflow.
# An export function takes a metrics.PerfSpanQuery over tracer timelines and
# returns MetricsT entries named "perf/..." (each value paired with None as
# aggregation function), optionally also writing a Perfetto trace.
# Three layouts are supported, chosen from the role -> device id mapping:
# colocated (rollout, actor, reference share devices), rollout on one mesh with
# actor+reference on another, and fully disaggregated.
# Expected trace structure (span/group names): on all timelines a
# "global_step" group containing "mini_batch_step" groups containing
# "micro_batch_steps" groups; device timelines hold "rollout" spans,
# "refer_inference" spans (directly or inside "actor_training" groups) and
# "actor_training" groups; the main thread holds "peft_train_step" spans inside
# "actor_training" groups; the main thread global_step may contain a
# "weight_sync" span.
# Aliases: ClusterConfig, MetricsT, partial (functools.partial),
# PerfSpanQuery, Span, SpanGroup, MetricsExportFn, PerfettoTraceWriter.
"""Helper functions for metrics export."""

from __future__ import annotations

import functools
from typing import Callable
from typing import Protocol

from absl import logging
import numpy as np
from tunix.perf import metrics
from tunix.perf import perfetto
from tunix.perf import span
from tunix.perf import trace
from tunix.rl import rl_cluster


ClusterConfig = rl_cluster.ClusterConfig
MetricsT = metrics.MetricsT
partial = functools.partial
PerfSpanQuery = metrics.PerfSpanQuery
Span = span.Span
SpanGroup = span.SpanGroup

MetricsExportFn = Callable[[PerfSpanQuery], MetricsT]
PerfettoTraceWriter = perfetto.PerfettoTraceWriter


class _GrpoExtractSpansFn(Protocol):
  # [CLASS DOC]
  # HELPER: Typing protocol for span extraction callables: called with keyword
  # query, returning (ok flag, global step groups, rollout spans, reference
  # inference spans, actor train groups, actor train step spans).

  def __call__(
      self, query: PerfSpanQuery
  ) -> tuple[
      bool, list[SpanGroup], list[Span], list[Span], list[SpanGroup], list[Span]
  ]:
    # [FUNCTION DOC]
    # HELPER: Protocol signature; the default body returns (False, None x5).
    return (False, None, None, None, None, None)


class PerfMetricsExport:
  # [CLASS DOC]
  # Namespace of static factory methods and the GRPO metric computations they
  # bind. Not meant to be instantiated.
  """Provides helper functions to create metrics export functions.

  1. from role to devices mapping

    role_to_devices = {
        "rollout": ["tpu0", "tpu1"],
        "actor": ["tpu2", "tpu3"],
        "refer": ["tpu4", "tpu5"],
    }
    export_fn = PerfMetricsExport.from_role_to_devices(role_to_devices)

  2. from cluster config

   export_fn = PerfMetricsExport.from_cluster_config(cluster_config)

   # DEPRECATED: use from_cluster_config instead.
   export_fn = PerfMetricsExport.create_metrics_export_fn(cluster_config)
  """

  @staticmethod
  def from_role_to_devices(
      role_to_devices: dict[str, list[str]],
      trace_writer: PerfettoTraceWriter | None = None,
      log_rollout_time_at_micro_batch_level: bool = False,
      log_actor_train_time_at_micro_batch_level: bool = False,
  ) -> MetricsExportFn:
    # [FUNCTION DOC]
    # Creates an export function from role ("rollout", "actor", "refer") to
    # device timeline id lists. Chooses the metric function by list equality:
    # all three equal -> colocated; rollout differs and actor equals refer ->
    # rollout_1_actor_2_reference_2; rollout differs from actor and actor
    # differs from refer -> fully disaggregated; otherwise ValueError
    # ("Unsupported mesh configuration"). Logs the chosen layout. Binds
    # _grpo_extract_spans_and_groups with the mapping and the two micro-batch
    # logging flags, and returns a partial of the metric function taking
    # (extract fn, trace_writer) so that the result only needs the query.
    # Exceptions: KeyError if a role is missing.
    # KNOWN ISSUE: rollout == actor != refer raises ValueError, and
    # rollout == refer != actor is classified as fully disaggregated.
    """Creates a metrics export function based on the role to devices mapping.

    Args:
      role_to_devices: A dictionary mapping role names to a list of device
        identifiers.
      trace_writer: An optional PerfettoTraceWriter to log performance traces.
      log_rollout_time_at_micro_batch_level: Whether to log rollout time at the
        micro batch level. This is a temporary flag. It will be removed once
        metrics are exported at the micro batch.
      log_actor_train_time_at_micro_batch_level: Whether to log actor train time
        at the micro batch level. This is a temporary flag. It will be removed
        once metrics are exported at the micro batch.

    Returns:
      A callable function that takes a PerfSpanQuery and returns MetricsT.
    """

    r2d = role_to_devices
    if r2d["rollout"] == r2d["actor"] == r2d["refer"]:
      logging.info(
          "Collecting perf metrics with rollout, actor and reference colocated."
      )
      export_fn = PerfMetricsExport._grpo_metrics_colocated
    elif r2d["rollout"] != r2d["actor"] == r2d["refer"]:
      logging.info(
          "Collecting perf metrics with rollout on one mesh, and actor and"
          " reference on another mesh."
      )
      export_fn = PerfMetricsExport._grpo_metrics_rollout_1_actor_2_reference_2
    elif r2d["rollout"] != r2d["actor"] != r2d["refer"]:
      logging.info(
          "Collecting perf metrics fully disaggregated: rollout, actor and"
          " reference on three different meshes."
      )
      export_fn = PerfMetricsExport._grpo_metrics_fully_disaggregated
    else:
      raise ValueError("Unsupported mesh configuration.")

    extract_spans_fn = partial(
        PerfMetricsExport._grpo_extract_spans_and_groups,
        role_to_devices=role_to_devices,
        log_rollout_time_at_micro_batch_level=log_rollout_time_at_micro_batch_level,
        log_actor_train_time_at_micro_batch_level=log_actor_train_time_at_micro_batch_level,
    )
    return partial(export_fn, extract_spans_fn, trace_writer)

  @staticmethod
  def from_cluster_config(
      cluster_config: ClusterConfig,
      log_rollout_time_at_micro_batch_level: bool = False,
      log_actor_train_time_at_micro_batch_level: bool = False,
  ) -> MetricsExportFn:
    # [FUNCTION DOC]
    # Creates an export function from an RL cluster config: reads the
    # ROLLOUT, ACTOR and REFERENCE meshes, converts their flattened devices to
    # timeline ids; if training_config.perf_metrics_options exists and
    # enables the trace writer, creates a PerfettoTraceWriter with trace_dir
    # (or None for the default directory), else no writer. Delegates to
    # from_role_to_devices with the flags. Side effects: writer creation may
    # create directories.
    """Creates a metrics export function based on the mesh topology in cluster config.

    This function extracts the device mappings from the `cluster_config` to
    determine the mesh configuration (colocated, partially disaggregated, or
    fully disaggregated) and returns the appropriate metrics export function.

    Args:
      cluster_config: The cluster configuration containing role to mesh
        mappings.
      log_rollout_time_at_micro_batch_level: Whether to log rollout time at
        micro batch level. This is a temporary flag. It will be removed once
        metrics are exported at the micro batch.
      log_actor_train_time_at_micro_batch_level: Whether to log actor train time
        at micro batch level. This is a temporary flag. It will be removed once
        metrics are exported at the micro batch.

    Returns:
      A callable function that takes a PerfSpanQuery and returns MetricsT.
    """

    rollo_mesh = cluster_config.role_to_mesh[rl_cluster.Role.ROLLOUT]
    actor_mesh = cluster_config.role_to_mesh[rl_cluster.Role.ACTOR]
    refer_mesh = cluster_config.role_to_mesh[rl_cluster.Role.REFERENCE]

    rollo_devices = map(
        trace.create_device_timeline_id, rollo_mesh.devices.flatten().tolist()
    )
    actor_devices = map(
        trace.create_device_timeline_id, actor_mesh.devices.flatten().tolist()
    )
    refer_devices = map(
        trace.create_device_timeline_id, refer_mesh.devices.flatten().tolist()
    )

    perf_metrics_options = cluster_config.training_config.perf_metrics_options
    if (
        perf_metrics_options is not None
        and perf_metrics_options.enable_trace_writer
    ):
      # Setting export_dir to None will cause the trace writer to use a
      # default directory.
      export_dir = perf_metrics_options.trace_dir or None
      trace_writer = PerfettoTraceWriter(export_dir)
    else:
      trace_writer = None

    return PerfMetricsExport.from_role_to_devices(
        role_to_devices={
            "rollout": list(rollo_devices),
            "actor": list(actor_devices),
            "refer": list(refer_devices),
        },
        trace_writer=trace_writer,
        log_rollout_time_at_micro_batch_level=log_rollout_time_at_micro_batch_level,
        log_actor_train_time_at_micro_batch_level=log_actor_train_time_at_micro_batch_level,
    )

  # TODO(yangmu): DEPRECATED: remove after all users use the new API.
  @staticmethod
  def create_metrics_export_fn(
      cluster_config: ClusterConfig,
  ) -> MetricsExportFn:
    # [FUNCTION DOC]
    # Deprecated alias of from_cluster_config with default flags.
    return PerfMetricsExport.from_cluster_config(cluster_config)

  @staticmethod
  def _grpo_metrics_colocated(
      extract_spans_fn: _GrpoExtractSpansFn,
      trace_writer: PerfettoTraceWriter | None,
      query: PerfSpanQuery,
  ) -> MetricsT:
    # [FUNCTION DOC]
    # HELPER: Metrics for the colocated layout. Calls extract_spans_fn(query=
    # query); returns {} if not ok. ValueError if there are no global step
    # groups. Uses the first (main thread) global step group; weight sync
    # span = last "weight_sync" span in it, or a zero-length span at the group
    # end if absent. Computes global step duration, weight sync duration,
    # and per-item durations of rollout spans, reference inference spans,
    # actor train groups and actor train step spans. Logs the trace via the
    # writer if present. Output keys: perf/global_step_time,
    # perf/weight_sync_time, and perf/sum/* and perf/mean/* for rollout_time,
    # refer_inference_time, actor_train_time, actor_train_step_time (numpy
    # sum/mean; mean of empty is NaN with a runtime warning).
    """GRPO workflow: rollout, actor and reference are colocated on the same mesh.

    Args:
      extract_spans_fn: A callable to extract spans and span groups.
      trace_writer: A PerfettoTraceWriter to log performance traces.
      query: The PerfSpanQuery object to extract spans from.

    Returns:
      A dictionary of performance metrics.
    """
    # Step 1: gather spans and span groups

    (
        ok,
        global_step_groups,
        rollout_spans,
        refer_inference_spans,
        actor_train_groups,
        actor_train_step_spans,
    ) = extract_spans_fn(query=query)
    if not ok:
      return {}

    if not global_step_groups:
      raise ValueError("global_step_groups is empty")
    global_step_group = global_step_groups[0]
    weight_sync_span = global_step_group.find_last_inner_span("weight_sync")
    # If weight sync is skipped (due to shared model), create a zero duration
    # span for metrics computation.
    if weight_sync_span is None:
      weight_sync_span = Span("weight_sync", global_step_group.end)
      weight_sync_span.end = global_step_group.end

    # Step 2: compute metrics from spans and span groups

    global_step_time: float = global_step_group.duration
    weight_sync_time: float = weight_sync_span.duration

    rollout_time: list[float] = [span.duration for span in rollout_spans]

    refer_inference_time: list[float] = [
        span.duration for span in refer_inference_spans
    ]

    # train time includes gradient update and eval
    actor_train_time: list[float] = [
        group.duration for group in actor_train_groups
    ]
    actor_train_step_time: list[float] = [
        span.duration for span in actor_train_step_spans
    ]

    if trace_writer is not None:
      trace_writer.log_trace(
          global_step_groups,
          rollout_spans,
          refer_inference_spans,
          actor_train_groups,
      )

    # pyformat: disable
    return {
        "perf/global_step_time": (global_step_time, None),
        "perf/weight_sync_time": (weight_sync_time, None),
        "perf/sum/rollout_time": (np.sum(rollout_time), None),
        "perf/sum/refer_inference_time": (np.sum(refer_inference_time), None),
        "perf/sum/actor_train_time": (np.sum(actor_train_time), None),
        "perf/sum/actor_train_step_time": (np.sum(actor_train_step_time), None),
        "perf/mean/rollout_time": (np.mean(rollout_time), None),
        "perf/mean/refer_inference_time": (np.mean(refer_inference_time), None),
        "perf/mean/actor_train_time": (np.mean(actor_train_time), None),
        "perf/mean/actor_train_step_time": (np.mean(actor_train_step_time), None),
    }
    # pyformat: enable

  @staticmethod
  def _grpo_metrics_rollout_1_actor_2_reference_2(
      extract_spans_fn: _GrpoExtractSpansFn,
      trace_writer: PerfettoTraceWriter | None,
      query: PerfSpanQuery,
  ) -> MetricsT:
    # [FUNCTION DOC]
    # HELPER: Metrics when rollout is on its own mesh and actor+reference share
    # one. Same extraction and weight sync handling as colocated. Adds:
    # rollout_idle_time = weight sync begin - end of the last rollout span;
    # first_micro_batch_rollout_time = end of first rollout span - global step
    # begin; between_micro_batch_gap_time = for consecutive pairs, begin of
    # reference span i+1 minus end of actor train group i, plus a trailing 0.0.
    # Output keys: as colocated plus perf/rollout_idle_time,
    # perf/first_micro_batch_rollout_time and sum/mean of
    # between_micro_batch_gap_time. Exceptions: IndexError if no rollout spans.
    # KNOWN ISSUE: docstring lists a role_to_devices argument; the actual
    # first argument is extract_spans_fn.
    """GRPO workflow: actor and reference are on the same mesh,rollout is on a different mesh.

    Args:
      role_to_devices: A dictionary mapping role names to a list of device
        identifiers.
      trace_writer: A PerfettoTraceWriter to log performance traces.
      query: The PerfSpanQuery object to extract spans from.

    Returns:
      A dictionary of performance metrics.
    """
    # Step 1: gather spans and span groups

    (
        ok,
        global_step_groups,
        rollout_spans,
        refer_inference_spans,
        actor_train_groups,
        actor_train_step_spans,
    ) = extract_spans_fn(query=query)
    if not ok:
      return {}

    if not global_step_groups:
      raise ValueError("global_step_groups is empty")
    global_step_group = global_step_groups[0]
    weight_sync_span = global_step_group.find_last_inner_span("weight_sync")
    # If weight sync is skipped (due to shared model), create a zero duration
    # span for metrics computation.
    if weight_sync_span is None:
      weight_sync_span = Span("weight_sync", global_step_group.end)
      weight_sync_span.end = global_step_group.end

    # Step 2: compute metrics from spans and span groups

    global_step_time: float = global_step_group.duration
    weight_sync_time: float = weight_sync_span.duration

    rollout_time: list[float] = [span.duration for span in rollout_spans]
    rollout_idle_time: float = weight_sync_span.begin - rollout_spans[-1].end

    refer_inference_time: list[float] = [
        span.duration for span in refer_inference_spans
    ]

    # train time includes gradient update and eval
    actor_train_time: list[float] = [
        group.duration for group in actor_train_groups
    ]
    actor_train_step_time: list[float] = [
        span.duration for span in actor_train_step_spans
    ]

    first_micro_batch_rollout_time: float = (
        rollout_spans[0].end - global_step_group.begin
    )

    # append [0.0] to make size equal to micro batch
    between_micro_batch_gap_time: list[float] = [
        b.begin - a.end
        for a, b in zip(actor_train_groups[:-1], refer_inference_spans[1:])
    ] + [0.0]

    if trace_writer is not None:
      trace_writer.log_trace(
          global_step_groups,
          rollout_spans,
          refer_inference_spans,
          actor_train_groups,
      )

    # pyformat: disable
    return {
        "perf/global_step_time": (global_step_time, None),
        "perf/weight_sync_time": (weight_sync_time, None),
        "perf/rollout_idle_time": (rollout_idle_time, None),
        "perf/first_micro_batch_rollout_time": (first_micro_batch_rollout_time, None),
        "perf/sum/rollout_time": (np.sum(rollout_time), None),
        "perf/sum/refer_inference_time": (np.sum(refer_inference_time), None),
        "perf/sum/actor_train_time": (np.sum(actor_train_time), None),
        "perf/sum/actor_train_step_time": (np.sum(actor_train_step_time), None),
        "perf/sum/between_micro_batch_gap_time": (np.sum(between_micro_batch_gap_time), None),
        "perf/mean/rollout_time": (np.mean(rollout_time), None),
        "perf/mean/refer_inference_time": (np.mean(refer_inference_time), None),
        "perf/mean/actor_train_time": (np.mean(actor_train_time), None),
        "perf/mean/actor_train_step_time": (np.mean(actor_train_step_time), None),
        "perf/mean/between_micro_batch_gap_time": (np.mean(between_micro_batch_gap_time), None),
    }
    # pyformat: enable

  @staticmethod
  def _grpo_metrics_fully_disaggregated(
      extract_spans_fn: _GrpoExtractSpansFn,
      trace_writer: PerfettoTraceWriter | None,
      query: PerfSpanQuery,
  ) -> MetricsT:
    # [FUNCTION DOC]
    # HELPER: Metrics for the fully disaggregated layout. Same extraction; if
    # no weight_sync span exists, logs a warning and returns {}. Computes
    # global step / weight sync durations, rollout idle time and first
    # micro-batch rollout time as in the 2-mesh layout, plus refer_gap_time and
    # actor_gap_time lists computed for consecutive pairs as (end of next -
    # begin of previous), each with a trailing 0.0. Output keys: global step,
    # weight sync, rollout idle, first micro batch rollout, and sum/mean of
    # rollout_time, refer_inference_time, refer_gap_time, actor_train_time,
    # actor_train_step_time, actor_gap_time.
    # KNOWN ISSUE: the "gap" values are spans covering both items (next end -
    # previous begin), not the idle gap between them; docstring lists a
    # role_to_devices argument that does not exist.
    """GRPO workflow: rollout, actor and reference are all on different meshes.

    Args:
      role_to_devices: A dictionary mapping role names to a list of device
        identifiers.
      trace_writer: A PerfettoTraceWriter to log performance traces.
      query: The PerfSpanQuery object to extract spans from.

    Returns:
      A dictionary of performance metrics.
    """
    # Step 1: gather spans and span groups

    (
        ok,
        global_step_groups,
        rollout_spans,
        refer_inference_spans,
        actor_train_groups,
        actor_train_step_spans,
    ) = extract_spans_fn(query=query)
    if not ok:
      return {}

    if not global_step_groups:
      raise ValueError("global_step_groups is empty")
    global_step_group = global_step_groups[0]
    weight_sync_span = global_step_group.find_last_inner_span("weight_sync")
    if weight_sync_span is None:
      logging.warning("weight_sync is None")
      return {}

    # Step 2: compute metrics from spans and span groups

    global_step_time: float = global_step_group.duration
    weight_sync_time: float = weight_sync_span.duration

    rollout_time: list[float] = [span.duration for span in rollout_spans]
    rollout_idle_time: float = weight_sync_span.begin - rollout_spans[-1].end

    refer_inference_time: list[float] = [
        span.duration for span in refer_inference_spans
    ]
    # append [0.0] to make size equal to micro batch
    refer_gap_time: list[float] = [
        b.end - a.begin
        for a, b in zip(refer_inference_spans[:-1], refer_inference_spans[1:])
    ] + [0.0]

    # train time includes gradient update and eval
    actor_train_time: list[float] = [
        group.duration for group in actor_train_groups
    ]
    actor_train_step_time: list[float] = [
        span.duration for span in actor_train_step_spans
    ]

    first_micro_batch_rollout_time: float = (
        rollout_spans[0].end - global_step_group.begin
    )

    # append [0.0] to make size equal to micro batch
    actor_gap_time: list[float] = [
        b.end - a.begin
        for a, b in zip(actor_train_groups[:-1], actor_train_groups[1:])
    ] + [0.0]

    if trace_writer is not None:
      trace_writer.log_trace(
          global_step_groups,
          rollout_spans,
          refer_inference_spans,
          actor_train_groups,
      )

    # pyformat: disable
    return {
        "perf/global_step_time": (global_step_time, None),
        "perf/weight_sync_time": (weight_sync_time, None),
        "perf/rollout_idle_time": (rollout_idle_time, None),
        "perf/first_micro_batch_rollout_time": (first_micro_batch_rollout_time, None),
        "perf/sum/rollout_time": (np.sum(rollout_time), None),
        "perf/sum/refer_inference_time": (np.sum(refer_inference_time), None),
        "perf/sum/refer_gap_time": (np.sum(refer_gap_time), None),
        "perf/sum/actor_train_time": (np.sum(actor_train_time), None),
        "perf/sum/actor_train_step_time": (np.sum(actor_train_step_time), None),
        "perf/sum/actor_gap_time": (np.sum(actor_gap_time), None),
        "perf/mean/rollout_time": (np.mean(rollout_time), None),
        "perf/mean/refer_inference_time": (np.mean(refer_inference_time), None),
        "perf/mean/refer_gap_time": (np.mean(refer_gap_time), None),
        "perf/mean/actor_train_time": (np.mean(actor_train_time), None),
        "perf/mean/actor_train_step_time": (np.mean(actor_train_step_time), None),
        "perf/mean/actor_gap_time": (np.mean(actor_gap_time), None),
    }
    # pyformat: enable

  @staticmethod
  def _grpo_extract_spans_and_groups(
      role_to_devices: dict[str, list[str]],
      *,  # force keyword arguments
      log_rollout_time_at_micro_batch_level: bool,
      log_actor_train_time_at_micro_batch_level: bool,
      query: PerfSpanQuery,
  ) -> tuple[
      bool, list[SpanGroup], list[Span], list[Span], list[SpanGroup], list[Span]
  ]:
    # [FUNCTION DOC]
    # HELPER: Extracts GRPO spans for the LAST global step.
    # Inputs: role_to_devices; keyword-only flags for micro-batch logging and
    #   the query.
    # Behavior: host timelines = main thread id then other "thread-" ids
    #   sorted; collect the last "global_step" group from each (main first).
    #   None found -> warning and (False, [empty-named group], [], [], [], []).
    #   Build a selector last global_step -> all mini_batch_step -> all
    #   micro_batch_steps and evaluate it on the main thread and on the first
    #   device id of rollout, refer and actor roles. If any device result is
    #   empty -> warning and the same failure tuple. Collect: "rollout" spans
    #   from rollout groups; "refer_inference" spans directly in refer groups
    #   and inside their "actor_training" groups; "actor_training" groups from
    #   actor groups; "peft_train_step" spans inside main thread
    #   "actor_training" groups. If either logging flag is set, logs the number
    #   of global steps on the main thread, then per-item actor train and/or
    #   rollout durations.
    # Output: (True, global step groups, rollout spans, refer spans, actor
    #   train groups, actor train step spans). Exceptions: KeyError/IndexError
    #   for missing roles or empty device lists; ValueError from query.get for
    #   unknown timeline ids.
    """Extracts spans and span groups of the last global step for GRPO workflow."""

    # Get all global step groups from all host threads. The first
    # global step group is the one from the main thread.
    global_step_groups: list[SpanGroup] = []
    main_thread_id = query.get_main_thread_id()
    host_timelines = [main_thread_id] + sorted(
        tid
        for tid in query.get_timeline_ids()
        if tid != main_thread_id and tid.startswith("thread-")
    )

    for tid in host_timelines:
      gs = query().timeline(tid).last_group("global_step").get()
      if gs:
        global_step_groups.extend(gs)

    if not global_step_groups:
      logging.warning("global_step is None")
      return (False, [SpanGroup("")], [], [], [], [])

    micro_batch: PerfSpanQuery = (
        query()
        .last_group("global_step")
        .all_groups("mini_batch_step")
        .all_groups("micro_batch_steps")
    )
    main_groups = micro_batch.main().get()
    rollout_groups = micro_batch.timeline(role_to_devices["rollout"][0]).get()
    refer_groups = micro_batch.timeline(role_to_devices["refer"][0]).get()
    actor_groups = micro_batch.timeline(role_to_devices["actor"][0]).get()

    if not rollout_groups or not refer_groups or not actor_groups:
      logging.warning("rollout_group or refer_group or actor_group is None")
      return (False, [SpanGroup("")], [], [], [], [])

    rollout_span: list[Span] = []
    refer_inference_span: list[Span] = []
    actor_train_groups: list[SpanGroup] = []
    actor_train_step_span: list[Span] = []

    for group in rollout_groups:
      rollout_span.extend(group.find_all_inner_spans("rollout"))
    for group in refer_groups:
      refer_inference_span.extend(group.find_all_inner_spans("refer_inference"))
      for group1 in group.find_all_inner_groups("actor_training"):
        refer_inference_span.extend(
            group1.find_all_inner_spans("refer_inference")
        )
    for group in actor_groups:
      actor_train_groups.extend(group.find_all_inner_groups("actor_training"))
    # TODO(yangmu) rewrite this after peft_train_step is attached to device
    # timeline. Note that peft_train_step records the correct device timespan.
    for group in main_groups:
      for actor_train_group in group.find_all_inner_groups("actor_training"):
        actor_train_step_span.extend(
            actor_train_group.find_all_inner_spans("peft_train_step")
        )

    if (
        log_actor_train_time_at_micro_batch_level
        or log_rollout_time_at_micro_batch_level
    ):
      global_step_index: int = len(
          query().main().all_groups("global_step").get()
      )
      logging.info("global step [%s]", global_step_index)
    if log_actor_train_time_at_micro_batch_level:
      for i, time in enumerate(
          [group.duration for group in actor_train_groups]
      ):
        logging.info("actor train time [%s] = %s sec", i, time)
    if log_rollout_time_at_micro_batch_level:
      for i, time in enumerate([span.duration for span in rollout_span]):
        logging.info("rollout time [%s] = %s sec", i, time)

    return (
        True,
        global_step_groups,
        rollout_span,
        refer_inference_span,
        actor_train_groups,
        actor_train_step_span,
    )
