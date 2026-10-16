# Copyright 2026 Google LLC
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
# Unit tests for tunix.utils.mesh, the device-allocation and mesh-creation
# utilities: host/slice identification of devices (device_host_key,
# device_slice_id, Pathways repr parsing), allocation-policy normalization
# (COMPACT default or PERFORMANCE), grouping devices by slice and host,
# coordinate topology construction (bounding box, optional core_on_chip axis),
# whole-chip checks, logging summaries, box-shape enumeration and scoring,
# coordinate-region splitting, coordinate-based allocation of contiguous
# device boxes (with remaining-region tracking, policy-dependent region
# choice, single-slice preference and whole-slice cross-slice allocation),
# incremental allocation state, allocate_named_mesh_device_slices, and
# create_mesh. Devices are simulated with small per-test FakeDevice classes
# exposing only the attributes under test (id, coords, core_on_chip,
# process_index, slice_index, device_kind).
from unittest import mock

from absl.testing import absltest
import jax
from tunix.utils import mesh


class MeshUtilsTest(absltest.TestCase):
  # [CLASS DOC]
  # Test case for tunix.utils.mesh helpers and allocation logic.

  def test_device_host_key_prefers_slice_and_process_metadata(self):
    # [FUNCTION DOC]
    # Verifies device_host_key returns (slice_index, process_index) = (4, 7).
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake device with slice_index 4 and process_index 7.

      def __init__(self):
        # [FUNCTION DOC]
        # HELPER: sets the fixed attributes.
        self.slice_index = 4
        self.process_index = 7

    self.assertEqual(mesh.device_host_key(FakeDevice()), (4, 7))

  def test_device_host_key_uses_process_index_without_slice_index(self):
    # [FUNCTION DOC]
    # Verifies a device without slice_index yields (None, process_index).
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake device with only process_index 9.

      def __init__(self):
        # [FUNCTION DOC]
        # HELPER: sets process_index.
        self.process_index = 9

    self.assertEqual(mesh.device_host_key(FakeDevice()), (None, 9))

  def test_device_host_key_ignores_task_id_attribute(self):
    # [FUNCTION DOC]
    # Verifies task_id is ignored; host is identified by process_index only.
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake device with slice_index, process_index and task_id.

      def __init__(self):
        # [FUNCTION DOC]
        # HELPER: sets the fixed attributes.
        self.slice_index = 4
        self.process_index = 7
        self.task_id = 9  # Must be ignored; only process_index identifies host.

    self.assertEqual(mesh.device_host_key(FakeDevice()), (4, 7))

  def test_device_host_key_parses_slice_from_attr_logical_task_from_repr(self):
    # [FUNCTION DOC]
    # Verifies on a Pathways backend (patched) the slice comes from the
    # slice_index attribute while the host comes from logical_task parsed from
    # the device repr: result (4, 11).
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake Pathways device whose attributes differ from its repr fields.

      def __init__(self):
        # [FUNCTION DOC]
        # HELPER: sets slice_index 4, process_index 0, task_id 0.
        self.slice_index = 4  # Keep it different from repr on purpose
        self.process_index = 0
        self.task_id = 0

      def __repr__(self):
        # [FUNCTION DOC]
        # HELPER: returns a Pathways-style repr with logical_task=11 and slice=3.
        return (
            "device(0,TPU_DEVICE,coords=[0,0,0,0],logical_task=11,"
            "slice=3,default_mem=device,mem_spaces=3)"
        )

    with mock.patch.object(
        mesh.topology, "_is_pathways_backend_used", return_value=True
    ):
      self.assertEqual(mesh.device_host_key(FakeDevice()), (4, 11))

  def test_device_host_key_returns_none_without_task_metadata(self):
    # [FUNCTION DOC]
    # Verifies device_host_key returns None when no host metadata exists.
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake device with no attributes.
      pass

    self.assertIsNone(mesh.device_host_key(FakeDevice()))

  def test_device_slice_id_uses_slice_index_only(self):
    # [FUNCTION DOC]
    # Verifies device_slice_id returns slice_index or None if absent.
    class SliceIndexDevice:
      # [CLASS DOC]
      # HELPER: fake device with slice_index 4.

      def __init__(self):
        # [FUNCTION DOC]
        # HELPER: sets slice_index.
        self.slice_index = 4

    self.assertEqual(mesh.device_slice_id(SliceIndexDevice()), 4)
    self.assertIsNone(mesh.device_slice_id(object()))

  def test_normalize_allocation_policy_validates_and_uppercases(self):
    # [FUNCTION DOC]
    # Verifies normalize_allocation_policy: None -> 'COMPACT', case-insensitive
    # upper-casing, and ValueError 'allocation_policy must be one of' otherwise.
    self.assertEqual(mesh.normalize_allocation_policy(None), "COMPACT")
    self.assertEqual(mesh.normalize_allocation_policy("compact"), "COMPACT")
    self.assertEqual(
        mesh.normalize_allocation_policy("performance"), "PERFORMANCE"
    )
    with self.assertRaisesRegex(ValueError, "allocation_policy must be one of"):
      mesh.normalize_allocation_policy("bogus")

  def test_group_devices_by_slice_preserves_first_seen_order(self):
    # [FUNCTION DOC]
    # Verifies group_devices_by_slice groups by slice in first-seen order.
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake device with id and slice_index.

      def __init__(self, device_id, slice_index):
        # [FUNCTION DOC]
        # HELPER: stores id and slice_index.
        self.id = device_id
        self.slice_index = slice_index

    grouped = mesh.group_devices_by_slice([
        FakeDevice(0, 2),
        FakeDevice(1, 2),
        FakeDevice(2, 1),
        FakeDevice(3, 1),
    ])

    self.assertEqual(
        [[device.id for device in group] for group in grouped], [[0, 1], [2, 3]]
    )

  def test_group_devices_by_slice_treats_missing_metadata_as_one_slice(self):
    # [FUNCTION DOC]
    # Verifies devices lacking slice metadata form a single group.
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake device with only id.

      def __init__(self, device_id):
        # [FUNCTION DOC]
        # HELPER: stores id.
        self.id = device_id

    grouped = mesh.group_devices_by_slice([
        FakeDevice(0),
        FakeDevice(1),
    ])

    self.assertEqual(
        [[device.id for device in group] for group in grouped], [[0, 1]]
    )

  def test_partition_devices_by_host_groups_and_sorts_by_slice_then_host(self):
    # [FUNCTION DOC]
    # Verifies _partition_devices_by_host groups by (slice, host) sorted by
    # slice then process index.
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake device with id, slice_index and process_index.

      def __init__(self, device_id, slice_index, process_index):
        # [FUNCTION DOC]
        # HELPER: stores the three attributes.
        self.id = device_id
        self.slice_index = slice_index
        self.process_index = process_index

    groups = mesh._partition_devices_by_host([
        FakeDevice(3, 1, 1),
        FakeDevice(2, 1, 0),
        FakeDevice(1, 0, 1),
        FakeDevice(0, 0, 0),
    ])

    self.assertEqual(
        [[device.id for device in group] for group in groups],
        [[0], [1], [2], [3]],
    )

  def test_partition_devices_by_host_returns_none_without_metadata(self):
    # [FUNCTION DOC]
    # Verifies _partition_devices_by_host returns None without host metadata.
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake device with no attributes.
      pass

    self.assertIsNone(mesh._partition_devices_by_host([FakeDevice()]))

  def test_candidate_uses_whole_chips_requires_all_cores(self):
    # [FUNCTION DOC]
    # Verifies candidate_uses_whole_chips is False when a candidate set of 4D
    # coords (x,y,z,core) covers only some cores of a chip and True when all
    # cores of each touched chip are included.
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake device with id, coords and core_on_chip.

      def __init__(self, device_id, coords, core_on_chip):
        # [FUNCTION DOC]
        # HELPER: stores id, coords and core_on_chip.
        self.id = device_id
        self.coords = coords
        self.core_on_chip = core_on_chip

    topology = mesh.get_coord_topology([
        FakeDevice(0, (0, 0, 0), 0),
        FakeDevice(1, (0, 0, 0), 1),
        FakeDevice(2, (1, 0, 0), 0),
        FakeDevice(3, (1, 0, 0), 1),
    ])

    self.assertFalse(
        mesh.candidate_uses_whole_chips(
            topology,
            [(0, 0, 0, 0), (1, 0, 0, 0)],
        )
    )
    self.assertTrue(
        mesh.candidate_uses_whole_chips(
            topology,
            [(0, 0, 0, 0), (0, 0, 0, 1), (1, 0, 0, 0), (1, 0, 0, 1)],
        )
    )

  def test_candidate_uses_whole_chips_ignores_plain_coords_without_core_axis(
      self,
  ):
    # [FUNCTION DOC]
    # Verifies candidate_uses_whole_chips is True for topologies without a
    # core-on-chip axis.
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake device with id and coords.

      def __init__(self, device_id, coords):
        # [FUNCTION DOC]
        # HELPER: stores id and coords.
        self.id = device_id
        self.coords = coords

    topology = mesh.get_coord_topology([
        FakeDevice(0, (0, 0, 0)),
        FakeDevice(1, (0, 0, 1)),
    ])

    self.assertTrue(
        mesh.candidate_uses_whole_chips(
            topology,
            [(0, 0, 0)],
        )
    )

  def test_infer_core_on_chip_count_counts_cores_per_chip(self):
    # [FUNCTION DOC]
    # Verifies infer_core_on_chip_count returns 2 for chips exposing cores 0, 1.
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake device with coords and core_on_chip.

      def __init__(self, coords, core_on_chip):
        # [FUNCTION DOC]
        # HELPER: stores coords and core_on_chip.
        self.coords = coords
        self.core_on_chip = core_on_chip

    # Two chips, each exposing cores 0 and 1 -> 2 cores per chip.
    devices = [
        FakeDevice((0, 0, 0), 0),
        FakeDevice((0, 0, 0), 1),
        FakeDevice((1, 0, 0), 0),
        FakeDevice((1, 0, 0), 1),
    ]
    self.assertEqual(mesh.infer_core_on_chip_count(devices), 2)

  def test_infer_core_on_chip_count_returns_one_for_single_core(self):
    # [FUNCTION DOC]
    # Verifies a single core per chip yields count 1.
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake device with coords and core_on_chip 0.

      def __init__(self, coords):
        # [FUNCTION DOC]
        # HELPER: stores coords and core 0.
        self.coords = coords
        self.core_on_chip = 0

    self.assertEqual(mesh.infer_core_on_chip_count([FakeDevice((0, 0, 0))]), 1)

  def test_infer_core_on_chip_count_returns_none_without_core_metadata(self):
    # [FUNCTION DOC]
    # Verifies None when devices lack core_on_chip.
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake device with only coords.

      def __init__(self, coords):
        # [FUNCTION DOC]
        # HELPER: stores coords.
        self.coords = coords

    self.assertIsNone(mesh.infer_core_on_chip_count([FakeDevice((0, 0, 0))]))

  def test_get_coord_topology_builds_bounding_box(self):
    # [FUNCTION DOC]
    # Verifies get_coord_topology computes num_dims 3, bounding-box max_shape
    # (2, 2, 1) from offset coords, and keeps all_coords in device order.
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake device with id and coords.

      def __init__(self, device_id, coords):
        # [FUNCTION DOC]
        # HELPER: stores id and coords.
        self.id = device_id
        self.coords = coords

    fake_devices = [
        FakeDevice(0, (2, 1, 0)),
        FakeDevice(1, (3, 1, 0)),
        FakeDevice(2, (2, 2, 0)),
        FakeDevice(3, (3, 2, 0)),
    ]

    topology = mesh.get_coord_topology(fake_devices)

    self.assertIsNotNone(topology)
    self.assertEqual(topology.num_dims, 3)
    self.assertEqual(topology.max_shape, (2, 2, 1))
    self.assertEqual(
        topology.all_coords, ((2, 1, 0), (3, 1, 0), (2, 2, 0), (3, 2, 0))
    )

  def test_get_coord_topology_rejects_duplicate_coords(self):
    # [FUNCTION DOC]
    # Verifies duplicate coords (no core axis) make get_coord_topology None.
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake device with only coords.

      def __init__(self, coords):
        # [FUNCTION DOC]
        # HELPER: stores coords.
        self.coords = coords

    fake_devices = [FakeDevice((0, 0, 0)), FakeDevice((0, 0, 0))]

    self.assertIsNone(mesh.get_coord_topology(fake_devices))

  def test_get_coord_topology_uses_core_on_chip_to_disambiguate_devices(self):
    # [FUNCTION DOC]
    # Verifies core_on_chip is appended as a 4th coordinate to disambiguate
    # devices sharing a chip, setting has_core_on_chip_dimension.
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake device with coords and core_on_chip.

      def __init__(self, coords, core_on_chip):
        # [FUNCTION DOC]
        # HELPER: stores coords and core_on_chip.
        self.coords = coords
        self.core_on_chip = core_on_chip

    fake_devices = [
        FakeDevice((0, 0, 0), 0),
        FakeDevice((0, 0, 0), 1),
    ]

    topology = mesh.get_coord_topology(fake_devices)

    self.assertIsNotNone(topology)
    self.assertEqual(topology.all_coords, ((0, 0, 0, 0), (0, 0, 0, 1)))
    self.assertTrue(topology.has_core_on_chip_dimension)

  def test_get_coord_topology_marks_plain_coords_without_core_axis(self):
    # [FUNCTION DOC]
    # Verifies plain 3D coords give has_core_on_chip_dimension False.
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake device with only coords.

      def __init__(self, coords):
        # [FUNCTION DOC]
        # HELPER: stores coords.
        self.coords = coords

    topology = mesh.get_coord_topology(
        [FakeDevice((0, 0, 0)), FakeDevice((0, 0, 1))]
    )

    self.assertIsNotNone(topology)
    self.assertFalse(topology.has_core_on_chip_dimension)

  def test_get_coord_topology_rejects_empty_device_list(self):
    # [FUNCTION DOC]
    # Verifies an empty device list yields None.
    self.assertIsNone(mesh.get_coord_topology([]))

  def test_get_coord_topology_rejects_mismatched_coord_dimensions(self):
    # [FUNCTION DOC]
    # Verifies coords of differing lengths yield None.
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake device with only coords.

      def __init__(self, coords):
        # [FUNCTION DOC]
        # HELPER: stores coords.
        self.coords = coords

    fake_devices = [FakeDevice((0, 0, 0)), FakeDevice((0, 0, 0, 1))]

    self.assertIsNone(mesh.get_coord_topology(fake_devices))

  def test_summarize_devices_for_logging_includes_id_coords_and_host(self):
    # [FUNCTION DOC]
    # Verifies summarize_devices_for_logging emits dicts with id, coords and host
    # (slice_index, process_index).
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake device with id, coords, process_index and slice_index.

      def __init__(self, device_id, coords, process_index, slice_index):
        # [FUNCTION DOC]
        # HELPER: stores the four attributes.
        self.id = device_id
        self.coords = coords
        self.process_index = process_index
        self.slice_index = slice_index

    self.assertEqual(
        mesh.summarize_devices_for_logging([FakeDevice(11, (1, 2, 0), 5, 6)]),
        [{"id": 11, "coords": (1, 2, 0), "host": (6, 5)}],
    )

  def test_summarize_devices_for_logging_debug_includes_raw_core_on_chip(self):
    # [FUNCTION DOC]
    # Verifies debug mode additionally includes the raw core_on_chip value.
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake device with fixed id, coords, core_on_chip and host metadata.

      def __init__(self):
        # [FUNCTION DOC]
        # HELPER: sets the fixed attributes.
        self.id = 11
        self.coords = (1, 2, 0)
        self.core_on_chip = 1
        self.process_index = 5
        self.slice_index = 7

    self.assertEqual(
        mesh.summarize_devices_for_logging([FakeDevice()], debug=True),
        [{"id": 11, "coords": (1, 2, 0), "core_on_chip": 1, "host": (7, 5)}],
    )

  def test_summarize_devices_for_logging_respects_limit(self):
    # [FUNCTION DOC]
    # Verifies the limit argument truncates the summary to the first N devices.
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake device with id and coords (id, 0, 0).

      def __init__(self, device_id):
        # [FUNCTION DOC]
        # HELPER: stores id and derived coords.
        self.id = device_id
        self.coords = (device_id, 0, 0)

    summaries = mesh.summarize_devices_for_logging(
        [FakeDevice(0), FakeDevice(1), FakeDevice(2)], limit=2
    )
    self.assertEqual([s["id"] for s in summaries], [0, 1])

  def test_allocate_named_mesh_device_slices_ignores_host_groups_when_coord_allocation_fails(
      self,
  ):
    # [FUNCTION DOC]
    # Verifies that when coordinate allocation fails (patched
    # _allocate_devices_by_coords returns (None, None)) on a v7 coord topology,
    # allocate_named_mesh_device_slices raises ValueError 'coord-based
    # allocation could not construct a valid box' instead of falling back to
    # host groups.
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake TPU v7 device with id, process_index, coords, core_on_chip.

      def __init__(self, device_id, process_index, coords, core_on_chip):
        # [FUNCTION DOC]
        # HELPER: stores attributes and sets device_kind 'TPU v7'.
        self.id = device_id
        self.process_index = process_index
        self.coords = coords
        self.core_on_chip = core_on_chip
        self.device_kind = "TPU v7"

    fake_devices = []
    device_id = 0
    for process_index, z in ((0, 0), (1, 1)):
      for x in range(2):
        for y in range(2):
          for core_on_chip in (0, 1):
            fake_devices.append(
                FakeDevice(device_id, process_index, (x, y, z), core_on_chip)
            )
            device_id += 1

    with self.assertRaisesRegex(
        ValueError,
        "coord-based allocation could not construct a valid box",
    ):
      with mock.patch.object(
          mesh, "_allocate_devices_by_coords", return_value=(None, None)
      ):
        mesh.allocate_named_mesh_device_slices(
            [("actor", 8)],
            devices=fake_devices,
        )

  def test_divisors_returns_sorted_unique_factors(self):
    # [FUNCTION DOC]
    # Verifies _divisors(12) returns sorted factors [1, 2, 3, 4, 6, 12].
    self.assertEqual(mesh._divisors(12), [1, 2, 3, 4, 6, 12])

  def test_enumerate_box_shapes_returns_shapes_with_requested_volume(self):
    # [FUNCTION DOC]
    # Verifies _enumerate_box_shapes returns all shapes with the requested volume
    # that fit within the max shape, sorted.
    self.assertEqual(
        mesh._enumerate_box_shapes(4, (4, 2, 2)),
        [(1, 2, 2), (2, 1, 2), (2, 2, 1), (4, 1, 1)],
    )

  def test_coord_box_score_prefers_more_compact_shapes(self):
    # [FUNCTION DOC]
    # Verifies _coord_box_score gives a lower (better) score to a compact
    # (2,2,1) box than a stretched (1,4,1) box at the same start.
    compact_score = mesh._coord_box_score((0, 0, 0), (2, 2, 1))
    stretched_score = mesh._coord_box_score((0, 0, 0), (1, 4, 1))

    self.assertLess(compact_score, stretched_score)

  def test_split_coord_region_returns_x_y_z_remainders(self):
    # [FUNCTION DOC]
    # Verifies _split_coord_region for a box at the region origin returns the
    # x, y and z remainder regions (start, shape) in that order.
    region = mesh.CoordRegion((0, 0, 0), (16, 16, 16))

    self.assertEqual(
        mesh._split_coord_region(region, (0, 0, 0), (4, 4, 8)),
        (
            mesh.CoordRegion((4, 0, 0), (12, 16, 16)),
            mesh.CoordRegion((0, 4, 0), (4, 12, 16)),
            mesh.CoordRegion((0, 0, 8), (4, 4, 8)),
        ),
    )

  def test_coord_box_score_prioritizes_x_then_y_then_z_start_order(self):
    # [FUNCTION DOC]
    # Verifies at equal shape the score orders starts by x first, then y, then z.
    first = mesh._coord_box_score((0, 0, 0, 0), (2, 1, 1, 2))
    next_in_x = mesh._coord_box_score((2, 0, 0, 0), (2, 1, 1, 2))
    next_in_y = mesh._coord_box_score((0, 1, 0, 0), (2, 1, 1, 2))
    next_in_z = mesh._coord_box_score((0, 0, 1, 0), (2, 1, 1, 2))

    self.assertLess(first, next_in_x)
    self.assertLess(next_in_x, next_in_y)
    self.assertLess(next_in_y, next_in_z)

  def test_split_coord_region_accounts_for_non_origin_allocated_start(self):
    # [FUNCTION DOC]
    # Verifies _split_coord_region for a box at a non-origin start returns six
    # regions: before/after along x, then y, then z.
    region = mesh.CoordRegion((0, 0, 0), (16, 16, 16))

    self.assertEqual(
        mesh._split_coord_region(region, (4, 4, 4), (4, 4, 8)),
        (
            mesh.CoordRegion((0, 0, 0), (4, 16, 16)),
            mesh.CoordRegion((8, 0, 0), (8, 16, 16)),
            mesh.CoordRegion((4, 0, 0), (4, 4, 16)),
            mesh.CoordRegion((4, 8, 0), (4, 8, 16)),
            mesh.CoordRegion((4, 4, 0), (4, 4, 4)),
            mesh.CoordRegion((4, 4, 12), (4, 4, 4)),
        ),
    )

  def test_device_mesh_coords_appends_core_on_chip_when_present(self):
    # [FUNCTION DOC]
    # Verifies device_mesh_coords appends core_on_chip to coords when present.
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake device with coords (1,2,0) and core_on_chip 1.

      def __init__(self):
        # [FUNCTION DOC]
        # HELPER: sets the fixed attributes.
        self.coords = (1, 2, 0)
        self.core_on_chip = 1

    self.assertEqual(
        mesh.device_mesh_coords(FakeDevice()),
        (1, 2, 0, 1),
    )

  def test_device_mesh_coords_returns_none_without_coords(self):
    # [FUNCTION DOC]
    # Verifies device_mesh_coords returns None for a device without coords.
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake device with no attributes.
      pass

    self.assertIsNone(mesh.device_mesh_coords(FakeDevice()))

  def test_allocate_devices_by_coords_supports_edge_family_3d_coords_with_trailing_singleton(
      self,
  ):
    # [FUNCTION DOC]
    # Verifies coordinate allocation on an edge family (TPU v5 lite) 4x4 grid
    # with trailing singleton z picks 8 devices forming a 2x4 box (ids
    # 0,1,4,5,8,9,12,13).
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake v5e device with id, coords, process_index and core 0.

      def __init__(self, device_id, coords, process_index):
        # [FUNCTION DOC]
        # HELPER: stores attributes and sets device_kind 'TPU v5 lite'.
        self.id = device_id
        self.coords = coords
        self.process_index = process_index
        self.core_on_chip = 0
        self.device_kind = "TPU v5 lite"

    fake_devices = []
    device_id = 0
    for y in range(4):
      for x in range(4):
        fake_devices.append(FakeDevice(device_id, (x, y, 0), 0))
        device_id += 1

    allocated, _ = mesh._allocate_devices_by_coords(fake_devices, 8)

    self.assertEqual(
        [device.id for device in allocated], [0, 1, 4, 5, 8, 9, 12, 13]
    )

  def test_allocate_named_mesh_device_slices_prefers_coord_boxes(self):
    # [FUNCTION DOC]
    # Verifies allocate_named_mesh_device_slices uses a contiguous coord box
    # (including both cores per chip) for an 8-device 'actor' mesh, returning
    # devices in box order [0,1,8,9,2,3,10,11].
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake device with id and 4D coords.

      def __init__(self, device_id, coords):
        # [FUNCTION DOC]
        # HELPER: stores id and coords.
        self.id = device_id
        self.coords = coords

    fake_devices = [
        FakeDevice(0, (0, 0, 0, 0)),
        FakeDevice(1, (0, 0, 0, 1)),
        FakeDevice(2, (1, 0, 0, 0)),
        FakeDevice(3, (1, 0, 0, 1)),
        FakeDevice(4, (2, 0, 0, 0)),
        FakeDevice(5, (2, 0, 0, 1)),
        FakeDevice(6, (3, 0, 0, 0)),
        FakeDevice(7, (3, 0, 0, 1)),
        FakeDevice(8, (0, 1, 0, 0)),
        FakeDevice(9, (0, 1, 0, 1)),
        FakeDevice(10, (1, 1, 0, 0)),
        FakeDevice(11, (1, 1, 0, 1)),
        FakeDevice(12, (2, 1, 0, 0)),
        FakeDevice(13, (2, 1, 0, 1)),
        FakeDevice(14, (3, 1, 0, 0)),
        FakeDevice(15, (3, 1, 0, 1)),
    ]

    allocated = mesh.allocate_named_mesh_device_slices(
        [("actor", 8)],
        devices=fake_devices,
    )

    self.assertEqual(
        [device.id for device in allocated["actor"]],
        [0, 1, 8, 9, 2, 3, 10, 11],
    )

  def test_allocate_devices_by_coords_uses_core_on_chip_dimension(self):
    # [FUNCTION DOC]
    # Verifies v7 coordinate allocation treats core_on_chip as a 4th dimension,
    # choosing 8 devices [0,1,4,5,16,17,20,21] (2x2x1 chips x 2 cores).
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake v7 device with id, coords and core_on_chip.

      def __init__(self, device_id, coords, core_on_chip):
        # [FUNCTION DOC]
        # HELPER: stores attributes and sets device_kind 'TPU v7'.
        self.id = device_id
        self.coords = coords
        self.core_on_chip = core_on_chip
        self.device_kind = "TPU v7"

    fake_devices = []
    device_id = 0
    for x in range(4):
      for y in range(4):
        for z in range(2):
          for core_on_chip in (0, 1):
            fake_devices.append(FakeDevice(device_id, (x, y, z), core_on_chip))
            device_id += 1

    allocated, _ = mesh._allocate_devices_by_coords(fake_devices, 8)

    self.assertEqual(
        [device.id for device in allocated],
        [0, 1, 4, 5, 16, 17, 20, 21],
    )

  def test_allocate_devices_by_coords_returns_none_without_coord_topology(self):
    # [FUNCTION DOC]
    # Verifies _allocate_devices_by_coords returns (None, None) without coords.
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake device with only process_index.

      def __init__(self, process_index):
        # [FUNCTION DOC]
        # HELPER: stores process_index.
        self.process_index = process_index

    self.assertEqual(
        mesh._allocate_devices_by_coords([FakeDevice(0), FakeDevice(0)], 2),
        (None, None),
    )

  def test_allocate_devices_by_coords_returns_best_contiguous_box(self):
    # [FUNCTION DOC]
    # Verifies the best contiguous 2x2 box at the origin is chosen (ids 0..3).
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake device with id, coords and process_index.

      def __init__(self, device_id, coords, process_index):
        # [FUNCTION DOC]
        # HELPER: stores the attributes.
        self.id = device_id
        self.coords = coords
        self.process_index = process_index

    fake_devices = [
        FakeDevice(0, (0, 0, 0), 0),
        FakeDevice(1, (1, 0, 0), 0),
        FakeDevice(2, (0, 1, 0), 0),
        FakeDevice(3, (1, 1, 0), 0),
        FakeDevice(4, (2, 0, 0), 1),
        FakeDevice(5, (3, 0, 0), 1),
        FakeDevice(6, (2, 1, 0), 1),
        FakeDevice(7, (3, 1, 0), 1),
    ]

    allocated, _ = mesh._allocate_devices_by_coords(fake_devices, 4)

    self.assertEqual([device.id for device in allocated], [0, 1, 2, 3])

  def test_allocate_devices_by_coords_prefers_more_cubical_supported_shape(
      self,
  ):
    # [FUNCTION DOC]
    # Verifies for 256 v7 chips in a 4x8x16 grid the most cubical supported
    # shape (4, 8, 8) at the origin is chosen.
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake v7 device with id and coords.

      def __init__(self, device_id, coords):
        # [FUNCTION DOC]
        # HELPER: stores id, coords and device_kind 'TPU v7'.
        self.id = device_id
        self.coords = coords
        self.device_kind = "TPU v7"

    fake_devices = []
    device_id = 0
    for x in range(4):
      for y in range(8):
        for z in range(16):
          fake_devices.append(FakeDevice(device_id, (x, y, z)))
          device_id += 1

    allocated, _ = mesh._allocate_devices_by_coords(fake_devices, 256)

    allocated_coords = [device.coords for device in allocated]
    mins = tuple(
        min(coords[dim] for coords in allocated_coords) for dim in range(3)
    )
    maxs = tuple(
        max(coords[dim] for coords in allocated_coords) for dim in range(3)
    )

    self.assertLen(allocated, 256)
    self.assertEqual(mins, (0, 0, 0))
    self.assertEqual(maxs, (3, 7, 7))

  def test_allocate_devices_tracks_remaining_coord_regions(self):
    # [FUNCTION DOC]
    # Verifies allocate_devices with return_state on a 16^3 v7 grid allocates a
    # 4x4x8 box for 128 chips at the origin and records the x/y/z remainder
    # regions under slice key None in the returned state.
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake v7 device with id and coords.

      def __init__(self, device_id, coords):
        # [FUNCTION DOC]
        # HELPER: stores id, coords and device_kind 'TPU v7'.
        self.id = device_id
        self.coords = coords
        self.device_kind = "TPU v7"

    fake_devices = []
    device_id = 0
    for x in range(16):
      for y in range(16):
        for z in range(16):
          fake_devices.append(FakeDevice(device_id, (x, y, z)))
          device_id += 1

    allocated, next_state = mesh.allocate_devices(
        128,
        devices=fake_devices,
        return_state=True,
    )

    allocated_coords = [device.coords for device in allocated]

    self.assertEqual(
        next_state.remaining_coord_regions_by_slice,
        {
            None: (
                mesh.CoordRegion((4, 0, 0), (12, 16, 16)),
                mesh.CoordRegion((0, 4, 0), (4, 12, 16)),
                mesh.CoordRegion((0, 0, 8), (4, 4, 8)),
            )
        },
    )
    self.assertEqual(
        (
            tuple(
                min(coords[dim] for coords in allocated_coords)
                for dim in range(3)
            ),
            tuple(
                max(coords[dim] for coords in allocated_coords)
                for dim in range(3)
            ),
        ),
        ((0, 0, 0), (3, 3, 7)),
    )

  def test_allocate_devices_small_tpu7_request_stays_within_one_host(self):
    # [FUNCTION DOC]
    # Verifies small v7 requests (4 devices = 2 chips x 2 cores) stay within one
    # host: the first allocation takes ids 0..3 and leaves a matching remaining
    # region; a second incremental allocation takes ids 8..11 from that region,
    # which is then consumed.
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake v7 device with id, coords, core_on_chip, process_index and
      # slice_index 0.

      def __init__(self, device_id, coords, core_on_chip, process_index):
        # [FUNCTION DOC]
        # HELPER: stores attributes, slice_index 0 and device_kind 'TPU v7'.
        self.id = device_id
        self.coords = coords
        self.core_on_chip = core_on_chip
        self.process_index = process_index
        self.slice_index = 0
        self.device_kind = "TPU v7"

    fake_devices = []
    device_id = 0
    for z in range(4):
      for y in range(4):
        for x in range(4):
          process_index = x // 2 + 2 * (y // 2) + 4 * z
          for core_on_chip in (0, 1):
            fake_devices.append(
                FakeDevice(device_id, (x, y, z), core_on_chip, process_index)
            )
            device_id += 1

    actor_devices, next_state = mesh.allocate_devices(
        4,
        devices=fake_devices,
        mesh_name="actor_model_config",
        return_state=True,
    )

    self.assertEqual([device.id for device in actor_devices], [0, 1, 2, 3])
    self.assertEqual(
        next_state.remaining_coord_regions_by_slice[0][0],
        mesh.CoordRegion((0, 1, 0, 0), (2, 1, 1, 2)),
    )

    rollout_devices, next_state = mesh.allocate_devices(
        4,
        allocation_state=next_state,
        mesh_name="rollout_model_config",
        return_state=True,
    )

    self.assertEqual([device.id for device in rollout_devices], [8, 9, 10, 11])
    self.assertNotIn(
        mesh.CoordRegion((0, 1, 0, 0), (2, 1, 1, 2)),
        next_state.remaining_coord_regions_by_slice[0],
    )

  def test_allocate_devices_prefers_smallest_remaining_coord_region_first(self):
    # [FUNCTION DOC]
    # Verifies a second 128-chip allocation uses the smallest remaining region
    # (z remainder), spanning (0,0,8)-(3,3,15).
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake v7 device with id and coords.

      def __init__(self, device_id, coords):
        # [FUNCTION DOC]
        # HELPER: stores id, coords and device_kind 'TPU v7'.
        self.id = device_id
        self.coords = coords
        self.device_kind = "TPU v7"

    fake_devices = []
    device_id = 0
    for x in range(16):
      for y in range(16):
        for z in range(16):
          fake_devices.append(FakeDevice(device_id, (x, y, z)))
          device_id += 1

    _, next_state = mesh.allocate_devices(
        128,
        devices=fake_devices,
        return_state=True,
    )
    allocated = mesh.allocate_devices(128, allocation_state=next_state)

    allocated_coords = [device.coords for device in allocated]
    self.assertEqual(
        (
            tuple(
                min(coords[dim] for coords in allocated_coords)
                for dim in range(3)
            ),
            tuple(
                max(coords[dim] for coords in allocated_coords)
                for dim in range(3)
            ),
        ),
        ((0, 0, 8), (3, 3, 15)),
    )

  def test_allocate_devices_matches_required_count_to_smallest_fitting_remaining_region(
      self,
  ):
    # [FUNCTION DOC]
    # Verifies a 576-chip follow-up allocation goes to the smallest region that
    # fits it (y remainder), spanning (0,4,0)-(3,15,11).
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake v7 device with id and coords.

      def __init__(self, device_id, coords):
        # [FUNCTION DOC]
        # HELPER: stores id, coords and device_kind 'TPU v7'.
        self.id = device_id
        self.coords = coords
        self.device_kind = "TPU v7"

    fake_devices = []
    device_id = 0
    for x in range(16):
      for y in range(16):
        for z in range(16):
          fake_devices.append(FakeDevice(device_id, (x, y, z)))
          device_id += 1

    _, next_state = mesh.allocate_devices(
        128,
        devices=fake_devices,
        return_state=True,
    )
    allocated = mesh.allocate_devices(576, allocation_state=next_state)

    allocated_coords = [device.coords for device in allocated]
    self.assertEqual(
        (
            tuple(
                min(coords[dim] for coords in allocated_coords)
                for dim in range(3)
            ),
            tuple(
                max(coords[dim] for coords in allocated_coords)
                for dim in range(3)
            ),
        ),
        ((0, 4, 0), (3, 15, 11)),
    )

  def test_allocate_devices_performance_policy_prefers_more_cubical_shape(self):
    # [FUNCTION DOC]
    # Verifies under PERFORMANCE policy (kept in the state) a 512-chip follow-up
    # picks the more cubical 8x8x8 box (4,0,0)-(11,7,7).
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake v7 device with id and coords.

      def __init__(self, device_id, coords):
        # [FUNCTION DOC]
        # HELPER: stores id, coords and device_kind 'TPU v7'.
        self.id = device_id
        self.coords = coords
        self.device_kind = "TPU v7"

    fake_devices = []
    device_id = 0
    for x in range(16):
      for y in range(16):
        for z in range(16):
          fake_devices.append(FakeDevice(device_id, (x, y, z)))
          device_id += 1

    _, next_state = mesh.allocate_devices(
        128,
        devices=fake_devices,
        allocation_policy="PERFORMANCE",
        return_state=True,
    )
    allocated = mesh.allocate_devices(512, allocation_state=next_state)

    allocated_coords = [device.coords for device in allocated]
    self.assertEqual(
        (
            tuple(
                min(coords[dim] for coords in allocated_coords)
                for dim in range(3)
            ),
            tuple(
                max(coords[dim] for coords in allocated_coords)
                for dim in range(3)
            ),
        ),
        ((4, 0, 0), (11, 7, 7)),
    )

  def test_allocate_devices_compact_policy_prefers_smallest_fitting_region(
      self,
  ):
    # [FUNCTION DOC]
    # Verifies under COMPACT policy a 512-chip follow-up goes to the smallest
    # fitting remaining region (expected bounds asserted below).
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake v7 device with id and coords.

      def __init__(self, device_id, coords):
        # [FUNCTION DOC]
        # HELPER: stores id, coords and device_kind 'TPU v7'.
        self.id = device_id
        self.coords = coords
        self.device_kind = "TPU v7"

    fake_devices = []
    device_id = 0
    for x in range(16):
      for y in range(16):
        for z in range(16):
          fake_devices.append(FakeDevice(device_id, (x, y, z)))
          device_id += 1

    _, next_state = mesh.allocate_devices(
        128,
        devices=fake_devices,
        allocation_policy="COMPACT",
        return_state=True,
    )
    allocated = mesh.allocate_devices(512, allocation_state=next_state)

    allocated_coords = [device.coords for device in allocated]
    self.assertEqual(
        (
            tuple(
                min(coords[dim] for coords in allocated_coords)
                for dim in range(3)
            ),
            tuple(
                max(coords[dim] for coords in allocated_coords)
                for dim in range(3)
            ),
        ),
        ((0, 4, 0), (3, 11, 15)),
    )

  def test_allocate_devices_rejects_mismatched_policy_for_existing_state(self):
    # [FUNCTION DOC]
    # Verifies reusing a COMPACT allocation state with allocation_policy
    # PERFORMANCE raises ValueError 'allocation_policy must match
    # allocation_state.allocation_policy' (_allocate_devices_by_coords patched).
    fake_devices = [object(), object()]

    with mock.patch.object(
        mesh,
        "_allocate_devices_by_coords",
        return_value=([fake_devices[0]], None),
    ):
      _, state = mesh.allocate_devices(
          1,
          devices=fake_devices,
          allocation_policy="COMPACT",
          return_state=True,
      )

    with self.assertRaisesRegex(
        ValueError,
        "allocation_policy must match allocation_state.allocation_policy",
    ):
      mesh.allocate_devices(
          1,
          allocation_state=state,
          allocation_policy="PERFORMANCE",
      )

  def test_allocate_devices_allocates_single_mesh(self):
    # [FUNCTION DOC]
    # Verifies allocate_devices delegates to _allocate_devices_by_coords with
    # (devices, count, mesh_name None, 'COMPACT') and returns its device list.
    fake_devices = [object(), object()]

    with mock.patch.object(
        mesh,
        "_allocate_devices_by_coords",
        return_value=(fake_devices, None),
    ) as allocate_mock:
      allocated = mesh.allocate_devices(2, devices=fake_devices)

    allocate_mock.assert_called_once_with(
        fake_devices,
        2,
        None,
        "COMPACT",
    )
    self.assertIs(allocated, fake_devices)

  def test_allocate_devices_prefers_coords_over_host_groups(self):
    # [FUNCTION DOC]
    # Verifies v7 coordinate allocation selects the 2x2 box on host 0 even when
    # host grouping is possible, returned in coordinate box order [0, 2, 1, 3].
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake v7 device with id, coords and process_index.

      def __init__(self, device_id, coords, process_index):
        # [FUNCTION DOC]
        # HELPER: stores attributes and device_kind 'TPU v7'.
        self.id = device_id
        self.coords = coords
        self.process_index = process_index
        self.device_kind = "TPU v7"

    fake_devices = [
        FakeDevice(0, (0, 0, 0), 0),
        FakeDevice(1, (1, 0, 0), 0),
        FakeDevice(2, (0, 1, 0), 0),
        FakeDevice(3, (1, 1, 0), 0),
        FakeDevice(4, (2, 0, 0), 1),
        FakeDevice(5, (3, 0, 0), 1),
        FakeDevice(6, (2, 1, 0), 1),
        FakeDevice(7, (3, 1, 0), 1),
    ]

    allocated = mesh.allocate_devices(4, devices=fake_devices)

    self.assertEqual([device.id for device in allocated], [0, 2, 1, 3])

  def test_allocate_devices_returns_updated_state_for_incremental_use(self):
    # [FUNCTION DOC]
    # Verifies incremental allocation with a patched coordinate allocator: the
    # returned state's remaining_devices excludes allocated ones, a follow-up
    # allocation consumes the rest, and used_device_count reaches 3.
    fake_devices = [object(), object(), object()]

    with mock.patch.object(
        mesh,
        "_allocate_devices_by_coords",
        side_effect=[(fake_devices[:1], None), (fake_devices[1:], None)],
    ):
      assigned_devices, next_state = mesh.allocate_devices(
          1,
          devices=fake_devices,
          return_state=True,
      )
      remaining_devices = list(next_state.remaining_devices)
      assigned_devices_2, final_state = mesh.allocate_devices(
          2,
          allocation_state=next_state,
          return_state=True,
      )

    self.assertEqual(assigned_devices, fake_devices[:1])
    self.assertEqual(remaining_devices, fake_devices[1:])
    self.assertEqual(assigned_devices_2, fake_devices[1:])
    self.assertEqual(list(final_state.remaining_devices), [])
    self.assertEqual(final_state.used_device_count, 3)

  def test_allocate_devices_raises_when_incremental_state_is_exhausted(self):
    # [FUNCTION DOC]
    # Verifies ValueError 'but only 0 remain available' for an exhausted state.
    allocation_state = mesh.DeviceAllocationState(
        remaining_devices=(),
        total_device_count=0,
    )

    with self.assertRaisesRegex(ValueError, "but only 0 remain available"):
      mesh.allocate_devices(1, allocation_state=allocation_state)

  def test_allocate_devices_rejects_devices_and_state_together(self):
    # [FUNCTION DOC]
    # Verifies passing both devices and allocation_state raises ValueError.
    fake_devices = [object()]
    allocation_state = mesh.DeviceAllocationState(
        remaining_devices=tuple(fake_devices),
        total_device_count=1,
    )

    with self.assertRaisesRegex(
        ValueError,
        "Pass either devices or allocation_state to allocate_devices, not both",
    ):
      mesh.allocate_devices(
          1,
          devices=fake_devices,
          allocation_state=allocation_state,
      )

  def test_allocate_devices_prefers_single_slice_before_cross_slice(self):
    # [FUNCTION DOC]
    # Verifies a request that fits in one slice is satisfied from the first
    # slice only (ids 0..3).
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake device with id, slice_index and coords.

      def __init__(self, device_id, slice_index, coords):
        # [FUNCTION DOC]
        # HELPER: stores the attributes.
        self.id = device_id
        self.slice_index = slice_index
        self.coords = coords

    fake_devices = [
        FakeDevice(0, 0, (0, 0, 0)),
        FakeDevice(1, 0, (1, 0, 0)),
        FakeDevice(2, 0, (2, 0, 0)),
        FakeDevice(3, 0, (3, 0, 0)),
        FakeDevice(4, 1, (4, 0, 0)),
        FakeDevice(5, 1, (5, 0, 0)),
        FakeDevice(6, 1, (6, 0, 0)),
        FakeDevice(7, 1, (7, 0, 0)),
    ]

    allocated = mesh.allocate_devices(4, devices=fake_devices)

    self.assertEqual([device.id for device in allocated], [0, 1, 2, 3])

  def test_allocate_devices_prefers_single_slice_before_other_slices(self):
    # [FUNCTION DOC]
    # Verifies a 2-device v7 request is taken from slice 0 (ids 0, 1).
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake v7 device with id, slice_index, process_index and coords.

      def __init__(self, device_id, slice_index, process_index, coords):
        # [FUNCTION DOC]
        # HELPER: stores attributes and device_kind 'TPU v7'.
        self.id = device_id
        self.slice_index = slice_index
        self.process_index = process_index
        self.coords = coords
        self.device_kind = "TPU v7"

    fake_devices = [
        FakeDevice(0, 0, 0, (0, 0, 0)),
        FakeDevice(1, 0, 0, (1, 0, 0)),
        FakeDevice(2, 0, 0, (0, 1, 0)),
        FakeDevice(3, 0, 0, (1, 1, 0)),
        FakeDevice(4, 1, 1, (2, 0, 0)),
        FakeDevice(5, 1, 1, (3, 0, 0)),
        FakeDevice(6, 1, 1, (2, 1, 0)),
        FakeDevice(7, 1, 1, (3, 1, 0)),
    ]

    allocated = mesh.allocate_devices(2, devices=fake_devices)

    self.assertEqual([device.id for device in allocated], [0, 1])

  def test_allocate_devices_raises_when_cross_slice_request_needs_partial_slice(
      self,
  ):
    # [FUNCTION DOC]
    # Verifies a request larger than one slice but not a whole number of slices
    # (6 of 2x4) raises ValueError 'cross-slice allocation only supports whole
    # slices'.
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake device with id, slice_index and coords (interleaved x).

      def __init__(self, device_id, slice_index, coords):
        # [FUNCTION DOC]
        # HELPER: stores the attributes.
        self.id = device_id
        self.slice_index = slice_index
        self.coords = coords

    fake_devices = [
        FakeDevice(0, 0, (0, 0, 0)),
        FakeDevice(1, 0, (2, 0, 0)),
        FakeDevice(2, 0, (4, 0, 0)),
        FakeDevice(3, 0, (6, 0, 0)),
        FakeDevice(4, 1, (1, 0, 0)),
        FakeDevice(5, 1, (3, 0, 0)),
        FakeDevice(6, 1, (5, 0, 0)),
        FakeDevice(7, 1, (7, 0, 0)),
    ]

    with self.assertRaisesRegex(
        ValueError,
        "cross-slice allocation only supports whole slices",
    ):
      mesh.allocate_devices(6, devices=fake_devices)

  def test_allocate_devices_consumes_whole_slices_in_order_when_cross_slice(
      self,
  ):
    # [FUNCTION DOC]
    # Verifies a request equal to all slices consumes whole slices in slice
    # order (ids 0..7), regardless of coordinate interleaving.
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake device with id, slice_index and coords.

      def __init__(self, device_id, slice_index, coords):
        # [FUNCTION DOC]
        # HELPER: stores the attributes.
        self.id = device_id
        self.slice_index = slice_index
        self.coords = coords

    fake_devices = [
        FakeDevice(0, 0, (0, 0, 0)),
        FakeDevice(1, 0, (2, 0, 0)),
        FakeDevice(2, 0, (4, 0, 0)),
        FakeDevice(3, 0, (6, 0, 0)),
        FakeDevice(4, 1, (1, 0, 0)),
        FakeDevice(5, 1, (3, 0, 0)),
        FakeDevice(6, 1, (5, 0, 0)),
        FakeDevice(7, 1, (7, 0, 0)),
    ]

    allocated = mesh.allocate_devices(8, devices=fake_devices)

    self.assertEqual(
        [device.id for device in allocated], [0, 1, 2, 3, 4, 5, 6, 7]
    )

  def test_allocate_devices_skips_partial_slice_for_cross_slice_request(self):
    # [FUNCTION DOC]
    # Verifies after a 2-device allocation from slice 0, an 8-device
    # cross-slice request skips the partially used slice 0 and takes whole
    # slices 1 and 2 (ids 4..11).
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake device with id, slice_index and coords.

      def __init__(self, device_id, slice_index, coords):
        # [FUNCTION DOC]
        # HELPER: stores the attributes.
        self.id = device_id
        self.slice_index = slice_index
        self.coords = coords

    fake_devices = [
        FakeDevice(0, 0, (0, 0, 0)),
        FakeDevice(1, 0, (1, 0, 0)),
        FakeDevice(2, 0, (2, 0, 0)),
        FakeDevice(3, 0, (3, 0, 0)),
        FakeDevice(4, 1, (4, 0, 0)),
        FakeDevice(5, 1, (5, 0, 0)),
        FakeDevice(6, 1, (6, 0, 0)),
        FakeDevice(7, 1, (7, 0, 0)),
        FakeDevice(8, 2, (8, 0, 0)),
        FakeDevice(9, 2, (9, 0, 0)),
        FakeDevice(10, 2, (10, 0, 0)),
        FakeDevice(11, 2, (11, 0, 0)),
    ]

    assigned_devices, next_state = mesh.allocate_devices(
        2,
        devices=fake_devices,
        return_state=True,
    )
    self.assertEqual([device.id for device in assigned_devices], [0, 1])

    allocated = mesh.allocate_devices(8, allocation_state=next_state)

    self.assertEqual(
        [device.id for device in allocated], [4, 5, 6, 7, 8, 9, 10, 11]
    )

  def test_allocate_named_mesh_device_slices_calls_allocate_devices_in_loop(
      self,
  ):
    # [FUNCTION DOC]
    # Verifies allocate_named_mesh_device_slices creates one initial state with
    # COMPACT policy, calls allocate_devices once per (name, count) in order
    # threading the returned state, logs no warning, and returns a dict of mesh
    # name to device list (allocate_devices and state creation patched).
    fake_devices = [object(), object(), object()]
    state_0 = mesh.DeviceAllocationState(
        remaining_devices=tuple(fake_devices),
        total_device_count=3,
        used_device_count=0,
    )
    state_1 = mesh.DeviceAllocationState(
        remaining_devices=tuple(fake_devices[1:]),
        total_device_count=3,
        used_device_count=1,
    )
    state_2 = mesh.DeviceAllocationState(
        remaining_devices=(),
        total_device_count=3,
        used_device_count=3,
    )

    with mock.patch.object(
        mesh,
        "allocate_devices",
        side_effect=[
            ([fake_devices[0]], state_1),
            ([fake_devices[1], fake_devices[2]], state_2),
        ],
    ) as allocate_mock, mock.patch.object(
        mesh,
        "_create_device_allocation_state",
        return_value=state_0,
    ) as state_mock, mock.patch.object(
        mesh.logging,
        "warning",
    ) as warning_mock:
      allocated = mesh.allocate_named_mesh_device_slices(
          [("mesh1", 1), ("mesh2", 2)],
          devices=fake_devices,
      )

    state_mock.assert_called_once_with(
        fake_devices,
        allocation_policy="COMPACT",
    )
    self.assertEqual(allocate_mock.call_count, 2)
    self.assertEqual(
        allocate_mock.call_args_list,
        [
            mock.call(
                1,
                mesh_name="mesh1",
                allocation_state=state_0,
                return_state=True,
            ),
            mock.call(
                2,
                mesh_name="mesh2",
                allocation_state=state_1,
                return_state=True,
            ),
        ],
    )
    warning_mock.assert_not_called()
    self.assertEqual(
        allocated,
        {
            "mesh1": [fake_devices[0]],
            "mesh2": [fake_devices[1], fake_devices[2]],
        },
    )

  @mock.patch.object(jax, "device_count")
  def test_create_mesh_uses_jax_make_mesh_without_assigned_devices(
      self, mock_device_count_fn
  ):
    # [FUNCTION DOC]
    # Verifies create_mesh without explicit devices calls jax.make_mesh with the
    # shape, axis names and Auto axis types, returning its result. Input: the
    # patched jax.device_count mock (returns 4).
    mock_device_count_fn.return_value = 4
    expected_mesh = object()

    with mock.patch.object(
        jax, "make_mesh", return_value=expected_mesh
    ) as make_mesh_mock:
      created_mesh = mesh.create_mesh((2, 2), ("x", "y"))

    make_mesh_mock.assert_called_once_with(
        (2, 2),
        ("x", "y"),
        axis_types=(jax.sharding.AxisType.Auto,) * 2,
    )
    self.assertIs(created_mesh, expected_mesh)

  def test_create_mesh_uses_assigned_devices(self):
    # [FUNCTION DOC]
    # Verifies create_mesh with explicit devices reshapes them to the mesh shape
    # and constructs jax.sharding.Mesh (patched with FakeMesh) with axis names.
    assigned_devices = ["d0", "d1", "d2", "d3"]

    class FakeMesh:
      # [CLASS DOC]
      # HELPER: stand-in for jax.sharding.Mesh recording constructor arguments.

      def __init__(self, devices, axis_names, axis_types=None):
        # [FUNCTION DOC]
        # HELPER: stores devices, axis_names and axis_types.
        self.devices = devices
        self.axis_names = axis_names
        self.axis_types = axis_types

    with mock.patch.object(jax.sharding, "Mesh", side_effect=FakeMesh):
      created_mesh = mesh.create_mesh(
          (2, 2),
          ("x", "y"),
          devices=assigned_devices,
      )

    self.assertEqual(created_mesh.devices.shape, (2, 2))
    self.assertEqual(
        created_mesh.devices.flatten().tolist(),
        assigned_devices,
    )
    self.assertEqual(created_mesh.axis_names, ("x", "y"))

  def test_allocate_named_mesh_device_slices_uses_jax_devices_by_default(self):
    # [FUNCTION DOC]
    # Verifies allocate_named_mesh_device_slices defaults to jax.devices()
    # (patched) when no devices are given.
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake device with id and coords (id, 0, 0).

      def __init__(self, device_id):
        # [FUNCTION DOC]
        # HELPER: stores id and coords.
        self.id = device_id
        self.coords = (device_id, 0, 0)

    fake_devices = [FakeDevice(0), FakeDevice(1)]

    with mock.patch.object(mesh.jax, "devices", return_value=fake_devices):
      allocated = mesh.allocate_named_mesh_device_slices([("trainer", 2)])

    self.assertEqual([device.id for device in allocated["trainer"]], [0, 1])

  def test_allocate_named_mesh_device_slices_raises_when_coord_allocation_fails(
      self,
  ):
    # [FUNCTION DOC]
    # Verifies a failed coordinate allocation for multiple named meshes raises
    # ValueError 'coord-based allocation could not construct a valid box'.
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake v7 device with id, process_index and coords.

      def __init__(self, device_id, process_index, coords):
        # [FUNCTION DOC]
        # HELPER: stores attributes and device_kind 'TPU v7'.
        self.id = device_id
        self.process_index = process_index
        self.coords = coords
        self.device_kind = "TPU v7"

    fake_devices = [
        FakeDevice(0, 0, (0, 0, 0)),
        FakeDevice(1, 0, (1, 0, 0)),
        FakeDevice(2, 0, (0, 1, 0)),
        FakeDevice(3, 0, (1, 1, 0)),
        FakeDevice(4, 1, (0, 0, 1)),
        FakeDevice(5, 1, (1, 0, 1)),
        FakeDevice(6, 1, (0, 1, 1)),
        FakeDevice(7, 1, (1, 1, 1)),
    ]

    with self.assertRaisesRegex(
        ValueError,
        "coord-based allocation could not construct a valid box",
    ):
      with mock.patch.object(
          mesh, "_allocate_devices_by_coords", return_value=(None, None)
      ):
        mesh.allocate_named_mesh_device_slices(
            [("trainer", 4), ("rollout", 4)],
            devices=fake_devices,
        )

  def test_allocate_named_mesh_device_slices_raises_when_not_enough_devices(
      self,
  ):
    # [FUNCTION DOC]
    # Verifies requesting more devices than exist raises ValueError
    # 'but only 2 remain available'.
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake device with only id.

      def __init__(self, device_id):
        # [FUNCTION DOC]
        # HELPER: stores id.
        self.id = device_id

    fake_devices = [FakeDevice(0), FakeDevice(1)]

    with self.assertRaisesRegex(ValueError, "but only 2 remain available"):
      mesh.allocate_named_mesh_device_slices(
          [("trainer", 3)],
          devices=fake_devices,
      )


if __name__ == "__main__":
  absltest.main()
