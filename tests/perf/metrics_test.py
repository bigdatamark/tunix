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
# Unit tests for tunix.perf.metrics.PerfMetricsOptions: default values
# (perf v1 on, v2 off, empty export paths, trace writer on, empty trace_dir)
# and construction-time validation raising ValueError when an export path is
# set for a disabled perf version or the trace writer is enabled while both
# perf versions are disabled.
from absl.testing import absltest
from absl.testing import parameterized
from tunix.perf import metrics


class MetricsTest(parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for PerfMetricsOptions.

  def test_perf_metrics_options_defaults(self):
    # [FUNCTION DOC]
    # Verifies the default field values listed in the module doc.
    options = metrics.PerfMetricsOptions()
    self.assertTrue(options.enable_perf_v1)
    self.assertFalse(options.enable_perf_v2)
    self.assertEqual(options.custom_export_fn_path, "")
    self.assertEqual(options.custom_export_fn_path_v2, "")
    self.assertTrue(options.enable_trace_writer)
    self.assertEqual(options.trace_dir, "")

  @parameterized.named_parameters(
      dict(
          testcase_name="v1_disabled_but_path_set",
          enable_perf_v1=False,
          custom_export_fn_path="some.path",
          enable_perf_v2=False,
          custom_export_fn_path_v2="",
          expected_regex=(
              "custom_export_fn_path is set but enable_perf_v1 is False"
          ),
      ),
      dict(
          testcase_name="v2_disabled_but_path_set",
          enable_perf_v1=True,
          custom_export_fn_path="",
          enable_perf_v2=False,
          custom_export_fn_path_v2="some.path",
          expected_regex=(
              "custom_export_fn_path_v2 is set but enable_perf_v2 is False"
          ),
      ),
      dict(
          testcase_name="trace_writer_enabled_but_no_perf",
          enable_perf_v1=False,
          custom_export_fn_path="",
          enable_perf_v2=False,
          custom_export_fn_path_v2="",
          expected_regex=(
              "enable_trace_writer is True but neither perf v1 nor v2 is"
              " enabled."
          ),
      ),
  )
  def test_perf_metrics_options_validation_error(
      self,
      enable_perf_v1,
      custom_export_fn_path,
      enable_perf_v2,
      custom_export_fn_path_v2,
      expected_regex,
  ):
    # [FUNCTION DOC]
    # Verifies ValueError with the expected message for each inconsistent
    # combination (trace writer left at its default True).
    # Inputs: enable flags, export paths and expected_regex.
    with self.assertRaisesRegex(ValueError, expected_regex):
      metrics.PerfMetricsOptions(
          enable_perf_v1=enable_perf_v1,
          custom_export_fn_path=custom_export_fn_path,
          enable_perf_v2=enable_perf_v2,
          custom_export_fn_path_v2=custom_export_fn_path_v2,
      )

  def test_perf_metrics_options_validation_success(self):
    # [FUNCTION DOC]
    # Verifies a fully consistent configuration constructs without raising.
    # Should not raise exception
    metrics.PerfMetricsOptions(
        enable_perf_v1=True,
        custom_export_fn_path="some.path",
        enable_perf_v2=True,
        custom_export_fn_path_v2="some.other.path",
        enable_trace_writer=True,
    )


if __name__ == "__main__":
  absltest.main()
