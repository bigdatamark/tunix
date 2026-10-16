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
# Trace writers for the experimental (perf v2) tracer. TraceWriter is the
# abstract interface (write_timelines); NoopTraceWriter ignores input;
# PerfettoTraceWriter converts all committed spans of all timelines into a
# Perfetto protobuf trace and rewrites one file
# "perfetto_trace_v2_<unix seconds>.pb" in the trace directory (local or
# remote via etils epath) on every call.
# Track layout: parent tracks "Host - Main threads" (uuid 100000), "Host -
# Rollout threads" (100001) and one per device role cluster (100002 and up);
# each timeline gets its own track (uuid 100000 * (k + 2) for the k-th
# timeline seen) under its parent, and when its spans overlap, child lane
# tracks (timeline uuid + lane + 1) so that each lane is overlap-free.
# Span display names are enriched from tags (see _create_span_name).
"""Trace writer implementations and helper functions."""

from __future__ import annotations

import abc
from collections.abc import Iterable, Mapping
import dataclasses
import itertools
import time
from typing import Any

from absl import logging
from etils import epath
from perfetto.trace_builder.proto_builder import TraceProtoBuilder
from tunix.perf.experimental import constants as perf_constants
from tunix.perf.experimental import timeline
from tunix.perf.experimental import timeline_utils

from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TrackDescriptor
from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TrackEvent


Timeline = timeline.Timeline

_UUID_OFFSET = 100_000  # Offset for lane UUIDs.


def _create_span_name(name: str, tags: Mapping[str, Any]) -> str:
  # [FUNCTION DOC]
  # HELPER: Builds a display name "name (k=v, ...)" from tags, or just name
  # when no parts apply. Parts in order: "step=" if the step tag exists; for
  # peft_train spans "role=", "mini_batch=", "micro_batch=" when present; for
  # rollout, reference_inference, old_actor_inference, advantage_computation
  # and environment spans "group_id="; for rollout and environment spans
  # "pair_index="; for queue spans "name=" (the queued span's name).
  """Creates a descriptive name for the span based on its tags."""
  parts = []
  if perf_constants.STEP in tags:
    parts.append(f"step={tags[perf_constants.STEP]}")

  if name == perf_constants.PEFT_TRAIN:
    if perf_constants.ROLE in tags:
      parts.append(f"role={tags[perf_constants.ROLE]}")
    if perf_constants.MINI_BATCH in tags:
      parts.append(f"mini_batch={tags[perf_constants.MINI_BATCH]}")
    if perf_constants.MICRO_BATCH in tags:
      parts.append(f"micro_batch={tags[perf_constants.MICRO_BATCH]}")

  if name in [
      perf_constants.ROLLOUT,
      perf_constants.REFERENCE_INFERENCE,
      perf_constants.OLD_ACTOR_INFERENCE,
      perf_constants.ADVANTAGE_COMPUTATION,
      perf_constants.ENVIRONMENT,
  ]:
    if perf_constants.GROUP_ID in tags:
      parts.append(f"group_id={tags[perf_constants.GROUP_ID]}")

  if name in [perf_constants.ROLLOUT, perf_constants.ENVIRONMENT]:
    if perf_constants.PAIR_INDEX in tags:
      parts.append(f"pair_index={tags[perf_constants.PAIR_INDEX]}")

  if name == perf_constants.QUEUE:
    if perf_constants.NAME in tags:
      parts.append(f"name={tags[perf_constants.NAME]}")

  if parts:
    return f"{name} ({', '.join(parts)})"
  return name


def _assign_lanes(
    spans: Iterable[timeline.Span],
) -> tuple[Mapping[int, int], int]:
  # [FUNCTION DOC]
  # HELPER: Greedy interval lane assignment: sort spans by (begin, id); place
  # each in the first lane whose last end is <= its begin (updating that lane's
  # end), else open a new lane. Output: (span id -> lane index, lane count).
  # Note nested spans (child inside parent) also go to separate lanes.
  """Assigns lanes to spans to handle overlaps.

  Perfetto requires spans on the same track to be strictly nested (no arbitrary
  overlaps). This function assigns a lane index to each span such that spans
  in the same lane do not overlap.

  Args:
    spans: An iterable of spans to assign to lanes.

  Returns:
    A tuple (`lane_by_span_id`, `num_lanes`), where:
      `lane_by_span_id`: A dictionary mapping span IDs to their assigned lane
        index.
      `num_lanes`: The total number of lanes required.
  """
  sorted_spans = sorted(spans, key=lambda s: (s.begin, s.id))
  lanes_end_times = []
  lane_by_span_id = {}

  for s in sorted_spans:
    placed = False
    for lane_idx, lane_end in enumerate(lanes_end_times):
      if lane_end <= s.begin:
        lanes_end_times[lane_idx] = s.end
        lane_by_span_id[s.id] = lane_idx
        placed = True
        break
    if not placed:
      lane_by_span_id[s.id] = len(lanes_end_times)
      lanes_end_times.append(s.end)

  return lane_by_span_id, len(lanes_end_times)


class TraceWriter(abc.ABC):
  # [CLASS DOC]
  # Abstract base class for trace writers with one abstract method.
  """An abstract base class for writing traces."""

  @abc.abstractmethod
  def write_timelines(self, timelines: Mapping[str, Timeline]) -> None:
    # [FUNCTION DOC]
    # Abstract: write the given id -> timeline mapping to the trace.
    """Writes timelines to the trace."""


class NoopTraceWriter(TraceWriter):
  # [CLASS DOC]
  # Trace writer that discards everything.
  """A no-op trace writer that does nothing."""

  def write_timelines(self, timelines: Mapping[str, Timeline]) -> None:
    # [FUNCTION DOC]
    # Does nothing.
    del self, timelines  # Unused.


@dataclasses.dataclass
class TrackInfo:
  # [CLASS DOC]
  # Dataclass describing a parent track: display name and optional Perfetto
  # uuid.
  """Information about a track.

  Attributes:
    name: The name of the track.
    uuid: The unique identifier for the track in Perfetto.
  """

  name: str
  uuid: int | None = None


class PerfettoTraceWriter(TraceWriter):
  # [CLASS DOC]
  # Perfetto writer. State: trace dir, role -> device list mapping, parent
  # track infos keyed by track key, timeline id -> track key, timeline id ->
  # uuid, and the trace file path (None if initialization failed, which
  # disables writing). Not thread-safe; callers serialize writes.
  """A writer for Perfetto trace events."""

  def __init__(
      self,
      trace_dir: str,
      role_to_devices: Mapping[str, Any] | None = None,
  ):
    # [FUNCTION DOC]
    # Stores the trace dir and a copy of role_to_devices (empty if None);
    # creates the directory (parents allowed) and sets the file path
    # "perfetto_trace_v2_<unix seconds>.pb", logging it. Any exception is
    # logged and leaves the path None (best effort, never raises).
    """Initializes the instance.

    Args:
      trace_dir: The directory to export trace files to. This path can be a
        local Linux path or a remote storage path (e.g. gs://).
      role_to_devices: An optional mapping from role names to their assigned
        devices.
    """
    self._trace_dir = trace_dir
    self._role_to_devices = (
        dict(role_to_devices) if role_to_devices is not None else {}
    )
    self._track_info: dict[str, TrackInfo] = {}
    self._timeline_tracks: dict[str, str] = {}
    self._timeline_uuids: dict[str, int] = {}

    self._trace_file_path = None
    try:
      trace_dir_path = epath.Path(self._trace_dir)
      trace_dir_path.mkdir(parents=True, exist_ok=True)
      trace_file_name = f"perfetto_trace_v2_{int(time.time())}.pb"
      self._trace_file_path = trace_dir_path / trace_file_name
      logging.info(
          "Initializing perfetto trace writer at: %s", self._trace_file_path
      )
    except Exception:  # pylint: disable=broad-except
      # Catching broad exceptions to ensure that failures in trace
      # initialization (e.g., due to file system errors, permissions, etc.) do
      # not crash the application. Tracing is best-effort.
      logging.exception(
          "Failed to initialize perfetto trace writer in directory %r. Skipping"
          " trace dumping for this run.",
          self._trace_dir,
      )
      self._trace_file_path = None

  def _get_device_track_name(self, tl_id: str) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Track name for a device or queue timeline: None for host
    # timelines. Strips the "_queue" suffix, then collects (in mapping order,
    # without duplicates) every role whose device list contains a device whose
    # timeline id equals it. Returns the roles converted from snake_case to
    # CamelCase, comma-joined, followed by " Cluster"; None when no role
    # matches. Exceptions: ValueError for unsupported device entries.
    """Gets a formatted track name for a device timeline.

    Args:
      tl_id: The timeline ID.

    Returns:
      A formatted track name. Returns None if the timeline is a host timeline
      or if the device is not found in the role_to_devices mapping.
    """
    if timeline_utils.is_host_timeline(tl_id):
      return None

    base_tl_id = (
        tl_id[:-6] if timeline_utils.is_queued_timeline(tl_id) else tl_id
    )

    cluster_roles = []

    for role, devices in self._role_to_devices.items():
      for device in devices:
        device_str = timeline_utils.generate_device_timeline_id(device)
        if device_str == base_tl_id and role not in cluster_roles:

          cluster_roles.append(role)

    if cluster_roles:
      camel_roles = [
          "".join(word.capitalize() for word in role.split("_"))
          for role in cluster_roles
      ]
      return f"{', '.join(camel_roles)} Cluster"
    return None

  def write(self, builder: TraceProtoBuilder) -> None:
    # [FUNCTION DOC]
    # Overwrites the trace file with builder's serialized bytes; no-op without
    # a path; exceptions are logged, not raised.
    """Writes the built trace to the file."""
    if self._trace_file_path is None:
      return

    try:
      # TODO: b/480134569 -  see if file writing is a bottleneck and explore
      # faster alternatives (e.g., keeping in memory and writing at the end).
      self._trace_file_path.write_bytes(builder.serialize())
    except Exception:  # pylint: disable=broad-except
      # Catching broad exceptions to ensure that failures in trace
      # serialization or writing do not crash the application. Tracing is
      # best-effort.
      logging.exception(
          "Failed to write to trace file: %s", self._trace_file_path
      )

  def _init_tracks(
      self, timelines: Mapping[str, Timeline]
  ) -> None:
    # [FUNCTION DOC]
    # HELPER: For each timeline id (sorted) not yet assigned to a parent track
    # and having at least one non-empty committed step: host timelines whose
    # committed spans are all "rollout" go under "host_rollout", other host
    # timelines under "host_main" (creating/refreshing those track infos);
    # device/queue timelines go under their role cluster track (created with
    # uuid 100000 + 2 + current number of track infos) or, if no role matches,
    # a warning is logged and the timeline gets no parent.
    # KNOWN ISSUE: the classification is made once; a host timeline first seen
    # with only rollout spans stays under the rollout track forever.
    """Initializes track info for timelines.

    Args:
      timelines: A mapping of timeline IDs to timelines.
    """

    for tl_id in sorted(timelines):
      if tl_id in self._timeline_tracks:
        continue

      tl = timelines[tl_id]
      if not any(tl.committed_steps):
        continue

      if timeline_utils.is_host_timeline(tl_id):
        # Rollout only timelines threads.
        if timeline_utils.is_timeline_only_of_allowed_type(
            tl, [perf_constants.ROLLOUT], include_cur_step=False
        ):
          self._track_info["host_rollout"] = TrackInfo(
              name="Host - Rollout threads",
              uuid=_UUID_OFFSET + 1,
          )
          self._timeline_tracks[tl_id] = "host_rollout"
        # Main timelines threads.
        else:
          self._track_info["host_main"] = TrackInfo(
              name="Host - Main threads",
              uuid=_UUID_OFFSET,
          )
          self._timeline_tracks[tl_id] = "host_main"
      else:
        track_name = self._get_device_track_name(tl_id)
        if track_name:
          if not self._track_info.get(track_name):
            self._track_info[track_name] = TrackInfo(
                name=track_name,
                uuid=_UUID_OFFSET + (2 + len(self._track_info)),
            )
          self._timeline_tracks[tl_id] = track_name
        else:
          logging.warning("Failed to get track name for timeline ID: %s", tl_id)

  def write_timelines(self, timelines: Mapping[str, Timeline]) -> None:
    # [FUNCTION DOC]
    # Writes the full trace. Returns early if timelines is empty or no
    # timeline has a non-empty committed step. Builds a TraceProtoBuilder:
    # initializes tracks; emits descriptors for all parent tracks; for each
    # timeline (sorted id) with committed spans, assigns/reuses its uuid,
    # emits its descriptor (with parent when known), assigns lanes over ALL
    # committed spans (emitting lane descriptors with empty names when more
    # than one lane), and creates slice-begin events (timestamp in ns relative
    # to the timeline's born time, enriched name) and slice-end events for
    # ended spans. All events are sorted by timestamp with ends before begins
    # at equal timestamps, emitted with sequence id 1, then written (file
    # overwritten).
    # Side effects: file write; mutates track/uuid bookkeeping.
    # KNOWN ISSUE: every call reprocesses all historical steps, so cost grows
    # with run length.
    """Writes timelines to the trace file."""
    if not timelines:
      return

    if not any(any(tl.committed_steps) for tl in timelines.values()):
      return

    builder = TraceProtoBuilder()

    self._init_tracks(timelines)

    # Write track descriptors for parent tracks.
    for track_info in self._track_info.values():
      packet = builder.add_packet()
      packet.track_descriptor.uuid = track_info.uuid
      packet.track_descriptor.name = track_info.name

    # Sort timelines by ID to ensure consistent track ordering.
    sorted_ids = sorted(timelines)

    events = []

    for tl_id in sorted_ids:
      tl = timelines[tl_id]

      if not any(tl.committed_steps):
        continue

      # Assign a UUID to the timeline if it hasn't been assigned one yet. Offset
      # by 2 to account for track descriptor UUIDs.
      if tl_id not in self._timeline_uuids:
        self._timeline_uuids[tl_id] = _UUID_OFFSET * (
            len(self._timeline_uuids) + 2
        )
      tl_uuid = self._timeline_uuids[tl_id]

      packet = builder.add_packet()
      packet.track_descriptor.uuid = tl_uuid
      packet.track_descriptor.name = tl_id

      if tl_id in self._timeline_tracks:
        track_info = self._track_info[self._timeline_tracks[tl_id]]
        packet.track_descriptor.parent_uuid = track_info.uuid

      # TODO: noghabi -  limit processing to last steps. we don't need to start
      # from the beginning every time.
      all_spans = list(
          itertools.chain.from_iterable(
              step.values() for step in tl.committed_steps
          )
      )
      lane_by_span_id, num_lanes = _assign_lanes(all_spans)

      if num_lanes > 1:
        # Emit track descriptors for each lane so they group under the timeline
        for lane_idx in range(num_lanes):
          lane_uuid = tl_uuid + lane_idx + 1
          packet = builder.add_packet()
          packet.track_descriptor.uuid = lane_uuid
          packet.track_descriptor.parent_uuid = tl_uuid
          packet.track_descriptor.name = ""  # empty name for lanes

      for s in all_spans:
        lane_idx = lane_by_span_id[s.id]
        lane_uuid = tl_uuid if num_lanes <= 1 else (tl_uuid + lane_idx + 1)

        # Timestamp in nanoseconds, relative to timeline creation (born).
        start_ns = int((s.begin - tl.born) * 1e9)
        events.append({
            "timestamp": start_ns,
            "type": TrackEvent.Type.TYPE_SLICE_BEGIN,
            "uuid": lane_uuid,
            "name": _create_span_name(s.name, s.tags),
        })

        if s.ended:
          end_ns = int((s.end - tl.born) * 1e9)
          events.append({
              "timestamp": end_ns,
              "type": TrackEvent.Type.TYPE_SLICE_END,
              "uuid": lane_uuid,
              "name": None,
          })

    # Perfetto trace processor requires events within a sequence to be strictly
    # sorted by timestamp. Out-of-order events can cause spans to be rendered
    # incorrectly (e.g., stretched end times).
    events.sort(
        key=lambda e: (
            e["timestamp"],
            0 if e["type"] == TrackEvent.Type.TYPE_SLICE_END else 1,
        )
    )

    for e in events:
      packet = builder.add_packet()
      packet.timestamp = e["timestamp"]
      packet.trusted_packet_sequence_id = 1
      packet.track_event.type = e["type"]
      packet.track_event.track_uuid = e["uuid"]
      if e["name"] is not None:
        packet.track_event.name = e["name"]

    self.write(builder)
