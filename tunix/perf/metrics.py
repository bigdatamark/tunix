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
# Configuration and query types for performance metrics in RL workflows.
# Defines: MetricsT (dict metric name -> (value(s) or string, optional
# aggregation function)); MetricsBuffer (per-step buffered metrics);
# PerfMetricsOptions (validated user options); PerfMetricsConfig (custom
# export functions for v1 and v2); PerfSpanQuery (fluent selector API over
# tracer timelines returning SpanGroups). Aliases: ArrayLike, Timeline (Any;
# a tracer timeline exposing a "root" SpanGroup), Span, SpanGroup.
"""APIs of performance metrics for RL workflows.

Config API:

    from tunix import PerfMetricsConfig
    from tunix import PerfMetricsExport

    # 1. Create a PerfMetricsConfig object.

    perf_config = PerfMetricsConfig()

    # 2. Create a metrics export function.

    # Let PerfMetricsExport create a metrics export function based on the mesh
    # topology in cluster config. See PerfMetricsExport for more details.
    perf_config.custom_export_fn = (
      PerfMetricsExport.from_cluster_config(cluster_config)
    )

    # Write your own custom export function.
    def my_custom_export_fn(query: PerfMetricsQuery) -> MetricsT:
      ...
    perf_config.custom_export_fn = my_custom_export_fn

    # 3. Pass the PerfMetricsConfig when creating rl cluster.

    rl_cluster.RLCluster(..., perf_config=perf_config)
"""

from __future__ import annotations

import dataclasses
from typing import Any
from typing import Callable, Dict, Mapping, Tuple

from absl import logging
from jax import typing
from tunix.perf import span


ArrayLike = typing.ArrayLike
Timeline = Any  # tunix.perf.span.Timeline
Span = span.Span
SpanGroup = span.SpanGroup

MetricsT = Dict[
    str, Tuple[ArrayLike | str, Callable[[ArrayLike], ArrayLike] | None]
]  # Metrics to be buffered: name -> (values, optional agg_fn)


@dataclasses.dataclass(slots=True)
class MetricsBuffer:
  # [CLASS DOC]
  # Slotted dataclass accumulating metrics for one global step: global_steps
  # (step number), metrics (name -> (list of buffered values, optional
  # aggregation function); default empty), mode (default "train").
  global_steps: int
  # Metrics to be buffered: name -> (list of (values), optional agg_fn)
  metrics: dict[
      str, tuple[list[ArrayLike | str], Callable[[ArrayLike], ArrayLike] | None]
  ] = dataclasses.field(default_factory=dict)
  mode: str = "train"


@dataclasses.dataclass(frozen=True)
class PerfMetricsOptions:
  # [CLASS DOC]
  # Frozen dataclass of perf options: enable_perf_v1 (default True),
  # enable_perf_v2 (False), custom_export_fn_path and custom_export_fn_path_v2
  # (module paths of custom export functions; empty = unset),
  # enable_trace_writer (True), trace_dir (""). Validated at construction.
  """Options for configuring performance metrics.

  Attributes:
    enable_perf_v1: Whether to enable perf metrics v1. Default is True.
    enable_perf_v2: Whether to enable perf metrics v2. Default is False.
    custom_export_fn_path: Path to the custom export function for Perf metrics
      v1. If set, the custom export function will be loaded from the path
      instead of being created by PerfMetricsExport.
    custom_export_fn_path_v2: Path to the custom export function for Perf
      metrics v2. If set, the custom export function will be loaded from the
      path instead of being created by PerfMetricsExport. This is currently
      experimental and planned to replace the v1 version.
    enable_trace_writer: Whether to enable the trace writer. By default, it is
      enabled when perf metrics are enabled. If False, the trace will not be
      written out.
    trace_dir: Directory the trace writer writes the raw metrics/events to.
  """

  enable_perf_v1: bool = True
  enable_perf_v2: bool = False
  custom_export_fn_path: str = ""
  custom_export_fn_path_v2: str = ""
  enable_trace_writer: bool = True
  trace_dir: str = ""

  def __post_init__(self):
    # [FUNCTION DOC]
    # Validation: ValueError if a v1 custom path is set while v1 is disabled,
    # if a v2 custom path is set while v2 is disabled, or if the trace writer
    # is enabled while both v1 and v2 are disabled. Logs a warning if
    # trace_dir is set while the trace writer is disabled.
    if self.custom_export_fn_path and not self.enable_perf_v1:
      raise ValueError(
          "custom_export_fn_path is set but enable_perf_v1 is False."
      )
    if self.custom_export_fn_path_v2 and not self.enable_perf_v2:
      raise ValueError(
          "custom_export_fn_path_v2 is set but enable_perf_v2 is False."
      )
    if self.enable_trace_writer and not (
        self.enable_perf_v1 or self.enable_perf_v2
    ):
      raise ValueError(
          "enable_trace_writer is True but neither perf v1 nor v2 is enabled."
      )
    if self.trace_dir and not self.enable_trace_writer:
      logging.warning(
          "trace_dir is set to %r but enable_trace_writer is False.",
          self.trace_dir,
      )


@dataclasses.dataclass
class PerfMetricsConfig:
  # [CLASS DOC]
  # Mutable dataclass holding optional custom export functions: v1 takes a
  # PerfSpanQuery and returns MetricsT; v2 takes a mapping of timeline id to
  # timeline and returns MetricsT. Both default to None.
  # (query, epoch) -> metrics
  custom_export_fn: Callable[[PerfSpanQuery], MetricsT] | None = None
  custom_export_fn_v2: Callable[[Mapping[str, Timeline]], MetricsT] | None = (
      None
  )


class PerfSpanQuery:
  # [CLASS DOC]
  # Fluent query over timelines. Holds the timelines dict, main thread id, the
  # selected timeline id (None until chosen) and an ordered list of group
  # selectors (name, kind, index) with kind 0=first, 1=last, 2=nth, 3=all.
  # Selector methods mutate and return self; calling the object returns a
  # copy with the same selections (fresh selector list), which is how callers
  # start independent queries.
  # KNOWN ISSUE: the docstring documents a ".group(name)" selector that does
  # not exist; the methods are first_group/last_group/nth_group/all_groups.
  """Query API for PerfMetrics.

  Format:
    query().<timeline_selector>.<group_selector>.get()

  Timeline selector (required):
    .main()
    .timeline(id)

  Group selector (optional):
    .group(name)
    .group(a).group(b).group(c) # nested groups

  Examples:

    # Get the last SpanGroup with name "global_step" in the main thread
    # timeline.
    query.main().group("global_step").get()

    # Get the last SpanGroup with name "global_step", then get the last
    # SpanGroup with name "mini_batch" in the tpu0 timeline.
    query.timeline("tpu0").group("global_step").group("mini_batch").get()
  """

  def __init__(self, timelines: dict[str, Timeline], main_thread_id: str):
    # [FUNCTION DOC]
    # Stores timelines and main thread id; no timeline selected; no group
    # selectors.
    self._timelines: dict[str, Timeline] = timelines
    self._main_thread_id = main_thread_id

    self._select_timeline: str | None = None

    # (name, type, arg)
    # type is 0 - first, 1 - last, 2 - nth, 3 - all
    self._select_groups: list[tuple[str, int, int]] = []

  def __call__(self) -> PerfSpanQuery:
    # [FUNCTION DOC]
    # Returns a new query sharing the timelines dict, with copied timeline
    # selection and a copy of the selector list.
    query = PerfSpanQuery(self._timelines, self._main_thread_id)
    query._select_timeline = self._select_timeline
    query._select_groups = self._select_groups.copy()
    return query

  def get_main_thread_id(self) -> str:
    # [FUNCTION DOC]
    # Returns the main thread timeline id.
    return self._main_thread_id

  def get_timeline_ids(self) -> list[str]:
    # [FUNCTION DOC]
    # Returns a list of all timeline ids.
    return list(self._timelines.keys())

  def timeline(self, id: str) -> PerfSpanQuery:
    # [FUNCTION DOC]
    # Selects timeline by id (not validated until get); returns self.
    self._select_timeline = id
    return self

  def main(self) -> PerfSpanQuery:
    # [FUNCTION DOC]
    # Selects the main thread timeline; returns self.
    self._select_timeline = self._main_thread_id
    return self

  def first_group(self, name: str) -> PerfSpanQuery:
    # [FUNCTION DOC]
    # Appends a "first child group named name" selector; returns self.
    self._select_groups.append((name, 0, 0))
    return self

  def last_group(self, name: str) -> PerfSpanQuery:
    # [FUNCTION DOC]
    # Appends a "last child group named name" selector; returns self.
    self._select_groups.append((name, 1, 0))
    return self

  def nth_group(self, name: str, index: int) -> PerfSpanQuery:
    # [FUNCTION DOC]
    # Appends an "index-th child group named name" selector; returns self.
    self._select_groups.append((name, 2, index))
    return self

  def all_groups(self, name: str) -> PerfSpanQuery:
    # [FUNCTION DOC]
    # Appends an "all child groups named name" selector; returns self.
    self._select_groups.append((name, 3, 0))
    return self

  def get(self) -> list[SpanGroup]:
    # [FUNCTION DOC]
    # Evaluates the query. ValueError if the selected timeline id is not
    # present (including when none was selected). Starts from [root group] of
    # that timeline and applies each selector in order to the current list of
    # groups using the span module's batch queries (first/last/nth/all direct
    # children by name). Output: resulting list of SpanGroups (possibly empty).
    # AssertionError for an invalid selector kind.
    """Returns SpanGroups gathered by the given selectors."""

    if self._select_timeline not in self._timelines:
      raise ValueError(f"timeline '{self._select_timeline}' not found.")

    curr_groups: list[SpanGroup] = [self._timelines[self._select_timeline].root]
    next_groups: list[SpanGroup] = []
    for name, type, index in self._select_groups:
      assert 0 <= type <= 3, f"invalid query type: {type}"
      match type:
        case 0:  # first
          next_groups = span.span_group_batch_query_first(curr_groups, name)
        case 1:  # last
          next_groups = span.span_group_batch_query_last(curr_groups, name)
        case 2:  # nth
          next_groups = span.span_group_batch_query_nth(
              curr_groups, name, index
          )
        case 3:  # all
          next_groups = span.span_group_batch_query_all(curr_groups, name)
      curr_groups = next_groups
      next_groups = []
    return curr_groups
