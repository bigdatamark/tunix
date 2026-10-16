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
# Unit tests for tunix.utils.topology: TPU device-kind normalization to
# family keys (v4, v5e, v5p, v6e, v7x), choosing best slice topology shapes
# for a chip count (2D/3D edge shapes for non-v7 families; for v7x 'fish'
# shapes made of 4x4x4 cubes for multiples of 64 chips, or unique sub-cube
# shapes below one cube, constrained by an available chip shape), and
# device attribute helpers including parsing Pathways device repr strings.
from absl.testing import absltest
from tunix.utils import topology


class TopologyTest(absltest.TestCase):
  # [CLASS DOC]
  # Test case for topology helpers, using small fake device classes.

  def test_normalize_device_kind_recognizes_supported_families(self):
    # [FUNCTION DOC]
    # Verifies _normalize_device_kind maps raw kinds ('TPU v7', 'TPU v6 lite',
    # 'TPU v5 lite', ...) to family keys and returns None for 'gpu'.
    self.assertEqual(topology._normalize_device_kind("TPU v7"), "v7x")
    self.assertEqual(topology._normalize_device_kind("TPU v6e"), "v6e")
    self.assertEqual(topology._normalize_device_kind("TPU v6 lite"), "v6e")
    self.assertEqual(topology._normalize_device_kind("TPU v5e"), "v5e")
    self.assertEqual(topology._normalize_device_kind("TPU v5 lite"), "v5e")
    self.assertEqual(topology._normalize_device_kind("TPU v5p"), "v5p")
    self.assertEqual(topology._normalize_device_kind("TPU v4"), "v4")
    self.assertIsNone(topology._normalize_device_kind("gpu"))

  def test_best_topology_shapes_for_chip_count_returns_unique_edge_shape(self):
    # [FUNCTION DOC]
    # Verifies v6e with 8 chips yields [(2, 4)] for rank 2 and [(2, 4, 1)] for
    # rank 3.
    self.assertEqual(
        topology.best_topology_shapes_for_chip_count("TPU v6e", 8, chip_rank=2),
        [(2, 4)],
    )
    self.assertEqual(
        topology.best_topology_shapes_for_chip_count("TPU v6e", 8, chip_rank=3),
        [(2, 4, 1)],
    )

  def test_best_topology_shapes_for_chip_count_returns_empty_for_unsupported_edge_count(
      self,
  ):
    # [FUNCTION DOC]
    # Verifies an unsupported chip count (3 on v6e) yields an empty list.
    # 3 chips is not a supported edge shape, so there is no candidate.
    self.assertEqual(
        topology.best_topology_shapes_for_chip_count("TPU v6e", 3),
        [],
    )

  def test_best_topology_shapes_for_chip_count_prefers_most_cubical_fish_shape(
      self,
  ):
    # [FUNCTION DOC]
    # Verifies 256 v7 chips choose the most cubical arrangement (4, 8, 8).
    self.assertEqual(
        topology.best_topology_shapes_for_chip_count("TPU v7", 256),
        [(4, 8, 8)],
    )

  def test_best_topology_shapes_for_chip_count_supports_single_host_fish_subslice(
      self,
  ):
    # [FUNCTION DOC]
    # Verifies 2 v7 chips within an available (4,4,4) region yield [(2, 1, 1)].
    self.assertEqual(
        topology.best_topology_shapes_for_chip_count(
            "TPU v7",
            2,
            available_chip_shape=(4, 4, 4),
        ),
        [(2, 1, 1)],
    )

  def test_best_topology_shapes_for_chip_count_returns_unique_fish_sub_cube(
      self,
  ):
    # [FUNCTION DOC]
    # Verifies below one cube each v7 volume maps to one sub-cube: 16 ->
    # (2, 2, 4), 32 -> (2, 4, 4).
    # Below the first full cube each volume maps to exactly one sub-cube shape.
    self.assertEqual(
        topology.best_topology_shapes_for_chip_count("TPU v7", 16),
        [(2, 2, 4)],
    )
    self.assertEqual(
        topology.best_topology_shapes_for_chip_count("TPU v7", 32),
        [(2, 4, 4)],
    )

  def test_best_topology_shapes_for_chip_count_fish_sub_cube_respects_available_shape(
      self,
  ):
    # [FUNCTION DOC]
    # Verifies a sub-cube that does not fit the available shape (2,2,1) yields [].
    # The 16-chip sub-cube (2, 2, 4) cannot fit when z is limited to 1.
    self.assertEqual(
        topology.best_topology_shapes_for_chip_count(
            "TPU v7", 16, available_chip_shape=(2, 2, 1)
        ),
        [],
    )

  def test_best_topology_shapes_for_chip_count_filters_by_available_shape(self):
    # [FUNCTION DOC]
    # Verifies a shape not fitting the available (1, 8) region is filtered out.
    self.assertEqual(
        topology.best_topology_shapes_for_chip_count(
            "TPU v6e",
            8,
            chip_rank=2,
            available_chip_shape=(1, 8),
        ),
        [],
    )

  def test_best_topology_shapes_for_chip_count_derives_shape_within_remaining_region(
      self,
  ):
    # [FUNCTION DOC]
    # Verifies 576 v7 chips within (4, 12, 16) yield [(4, 12, 12)].
    self.assertEqual(
        topology.best_topology_shapes_for_chip_count(
            "TPU v7",
            576,
            available_chip_shape=(4, 12, 16),
        ),
        [(4, 12, 12)],
    )

  def test_best_topology_shapes_for_chip_count_rejects_non_cube_multiple(self):
    # [FUNCTION DOC]
    # Verifies ValueError 'must be divisible by 64 chips' for 96 v7 chips.
    with self.assertRaisesRegex(ValueError, "must be divisible by 64 chips"):
      topology.best_topology_shapes_for_chip_count("TPU v7", 96)

  def test_best_fish_cube_shape_returns_single_cube(self):
    # [FUNCTION DOC]
    # Verifies _best_fish_cube_shape(64, None) is (4, 4, 4).
    self.assertEqual(topology._best_fish_cube_shape(64, None), (4, 4, 4))

  def test_best_fish_cube_shape_prefers_most_cubical_arrangement(self):
    # [FUNCTION DOC]
    # Verifies 256 chips prefer (4, 8, 8) over (4, 4, 16).
    # 256 chips = 4 cubes; (4, 8, 8) is more cubical than (4, 4, 16).
    self.assertEqual(topology._best_fish_cube_shape(256, None), (4, 8, 8))

  def test_best_fish_cube_shape_respects_available_shape(self):
    # [FUNCTION DOC]
    # Verifies the available shape bound is honoured (576 in (4,12,16)).
    self.assertEqual(
        topology._best_fish_cube_shape(576, (4, 12, 16)), (4, 12, 12)
    )

  def test_best_fish_cube_shape_returns_none_below_one_cube(self):
    # [FUNCTION DOC]
    # Verifies None is returned below 64 chips.
    self.assertIsNone(topology._best_fish_cube_shape(32, None))

  def test_best_fish_cube_shape_returns_none_when_no_arrangement_fits(self):
    # [FUNCTION DOC]
    # Verifies None when the bound cannot fit the required cubes.
    # 128 chips = 2 cubes, but a 4x4x4 bound leaves room for only one cube.
    self.assertIsNone(topology._best_fish_cube_shape(128, (4, 4, 4)))

  def test_best_fish_cube_shape_raises_on_non_cube_multiple(self):
    # [FUNCTION DOC]
    # Verifies ValueError for a non-multiple of 64.
    with self.assertRaisesRegex(ValueError, "must be divisible by 64 chips"):
      topology._best_fish_cube_shape(96, None)

  def test_device_family_resolves_from_first_device_kind(self):
    # [FUNCTION DOC]
    # Verifies _device_family uses the first device's device_kind.
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake device exposing a device_kind attribute.

      def __init__(self, device_kind):
        # [FUNCTION DOC]
        # HELPER: stores device_kind.
        self.device_kind = device_kind

    self.assertEqual(topology._device_family([FakeDevice("TPU v6e")]), "v6e")
    self.assertEqual(topology._device_family([FakeDevice("TPU v7")]), "v7x")

  def test_device_family_returns_none_without_device_kind(self):
    # [FUNCTION DOC]
    # Verifies _device_family returns None for an empty list or a device lacking
    # device_kind.
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake device with no attributes.
      pass

    self.assertIsNone(topology._device_family([]))
    self.assertIsNone(topology._device_family([FakeDevice()]))

  def test_resolve_family_accepts_raw_kind_and_family_key(self):
    # [FUNCTION DOC]
    # Verifies _resolve_family accepts raw kinds and already-normalized keys,
    # returning None for unknown kinds.
    self.assertEqual(topology._resolve_family("TPU v5 lite"), "v5e")
    self.assertEqual(topology._resolve_family("v5e"), "v5e")  # already a key
    self.assertIsNone(topology._resolve_family("gpu"))

  def test_device_attr_calls_callable_attributes(self):
    # [FUNCTION DOC]
    # Verifies _device_attr calls callable attributes, returns plain attributes,
    # and returns default (None unless given) for missing ones.
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake device with coords attribute and device_kind method.
      coords = (1, 2, 3)

      def device_kind(self):  # exposed lazily as a method
        # [FUNCTION DOC]
        # HELPER: returns 'TPU v7'.
        return "TPU v7"

    device = FakeDevice()
    self.assertEqual(topology._device_attr(device, "device_kind"), "TPU v7")
    self.assertEqual(topology._device_attr(device, "coords"), (1, 2, 3))
    self.assertIsNone(topology._device_attr(device, "missing"))
    self.assertEqual(topology._device_attr(device, "missing", default=5), 5)

  def test_pathways_device_host_attr_parses_int_and_list_values(self):
    # [FUNCTION DOC]
    # Verifies _pathways_device_host_attr parses 'name=value' fields from the
    # device repr into ints or int lists.
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake device whose repr contains coords and logical_task fields.

      def __repr__(self):
        # [FUNCTION DOC]
        # HELPER: returns the fixed Pathways-style repr string.
        return "device(0,TPU,coords=[1,2,3],logical_task=11,slice=3)"

    device = FakeDevice()
    self.assertEqual(
        topology._pathways_device_host_attr(device, "logical_task"), 11
    )
    self.assertEqual(
        topology._pathways_device_host_attr(device, "coords"), [1, 2, 3]
    )

  def test_pathways_device_host_attr_returns_none_when_absent(self):
    # [FUNCTION DOC]
    # Verifies None when the field is absent from the repr.
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake device whose repr lacks logical_task.

      def __repr__(self):
        # [FUNCTION DOC]
        # HELPER: returns the fixed repr string.
        return "device(0,TPU,slice=3)"

    self.assertIsNone(
        topology._pathways_device_host_attr(FakeDevice(), "logical_task")
    )

  def test_pathways_device_host_attr_anchors_on_delimiter(self):
    # [FUNCTION DOC]
    # Verifies field matching is anchored on a delimiter so 'xlogical_task' does
    # not match 'logical_task'.
    # A field that merely ends with the attr name must not match.
    class FakeDevice:
      # [CLASS DOC]
      # HELPER: fake device with a confusable field name in its repr.

      def __repr__(self):
        # [FUNCTION DOC]
        # HELPER: returns the fixed repr string.
        return "device(0,TPU,xlogical_task=99,logical_task=7)"

    self.assertEqual(
        topology._pathways_device_host_attr(FakeDevice(), "logical_task"), 7
    )


if __name__ == "__main__":
  absltest.main()
