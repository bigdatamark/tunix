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
# Unit tests for tunix.perf.experimental.export (perf v2): PerfMetricsExport
# is a context manager that owns a trace writer (PerfettoTraceWriter with
# trace_dir, defaulting to DEFAULT_TRACE_DIR when None or empty, and a
# role_to_devices map; or NoopTraceWriter when disabled) and a background
# ThreadPoolExecutor; export_metrics schedules writes in the background
# (warning if already shut down), _safe_write logs errors instead of raising,
# shutdown(wait) shuts the executor down and clears it, and
# from_cluster_config derives role_to_devices (role value to flattened mesh
# devices; empty when role_to_mesh is absent). log_metric_export_fn logs the
# timelines. Side effects: writes a trace file into a temp dir in one test.
import os
import tempfile
import time
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from tunix.perf.experimental import export
from tunix.perf.experimental import tracer
import jax
from tunix.rl import rl_cluster


class ExportTest(parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for the perf v2 exporter.

  def test_perf_metrics_export(self):
    # [FUNCTION DOC]
    # Verifies end-to-end: a PerfTracer exporting through PerfMetricsExport
    # writes exactly one file named 'perfetto_trace_v2_*' into trace_dir.
    # Backward compatibility check
    with tempfile.TemporaryDirectory() as tmp_dir:
      with export.PerfMetricsExport(trace_dir=tmp_dir) as exporter:
        # Create dummy timeline
        t = tracer.PerfTracer(export_fn=exporter.export_metrics)
        with t.span("test_span"):
          time.sleep(0.001)
        t.export()

      files = os.listdir(tmp_dir)
      self.assertLen(files, 1)
      self.assertStartsWith(files[0], "perfetto_trace_v2_")

  def test_basic_metrics_export(self):
    # [FUNCTION DOC]
    # Verifies log_metric_export_fn logs '=== Exporting Timelines ===' at INFO.
    with self.assertLogs(level="INFO") as logs:
      export.log_metric_export_fn({})
    self.assertTrue(
        any("=== Exporting Timelines ===" in log for log in logs.output)
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="none_dir",
          trace_dir=None,
          expected_dir=export.DEFAULT_TRACE_DIR,
      ),
      dict(
          testcase_name="empty_dir",
          trace_dir="",
          expected_dir=export.DEFAULT_TRACE_DIR,
      ),
      dict(
          testcase_name="custom_dir",
          trace_dir="/my/custom/path",
          expected_dir="/my/custom/path",
      ),
  )
  @mock.patch.object(
      export.trace_writer_lib,
      "PerfettoTraceWriter",
      autospec=True,
      spec_set=True,
  )
  def test_perf_metrics_export_initialization_with_trace_writer_enabled(
      self, mock_writer_cls, trace_dir, expected_dir
  ):
    # [FUNCTION DOC]
    # Verifies an enabled trace writer is constructed with the expected dir
    # (default for None/empty) and role_to_devices None, and that export_metrics
    # does not crash. Inputs: patched writer class, trace_dir, expected_dir.
    with export.PerfMetricsExport(
        enable_trace_writer=True, trace_dir=trace_dir
    ) as exporter:
      mock_writer_cls.assert_called_once_with(expected_dir, role_to_devices=None)
      # export_metrics shouldn't crash
      exporter.export_metrics({})

  @mock.patch.object(
      export.trace_writer_lib, "NoopTraceWriter", autospec=True, spec_set=True
  )
  def test_perf_metrics_export_initialization_with_trace_writer_disabled(
      self, mock_noop_cls
  ):
    # [FUNCTION DOC]
    # Verifies disabling the trace writer constructs NoopTraceWriter with no
    # args and stores it as _writer.
    with export.PerfMetricsExport(enable_trace_writer=False) as exporter:
      # export_metrics shouldn't crash
      exporter.export_metrics({})
      mock_noop_cls.assert_called_once_with()
      # Test that the writer is actually set to the NoopTraceWriter instance
      self.assertEqual(exporter._writer, mock_noop_cls.return_value)

  @mock.patch.object(
      export.concurrent.futures, "ThreadPoolExecutor", autospec=True
  )
  def test_perf_metrics_export_shutdown_waits_for_executor(
      self, mock_executor_cls
  ):
    # [FUNCTION DOC]
    # Verifies leaving the context calls executor.shutdown(wait=True).
    mock_executor_instance = mock_executor_cls.return_value
    with export.PerfMetricsExport(enable_trace_writer=True):
      pass
    mock_executor_instance.shutdown.assert_called_once_with(wait=True)

  @mock.patch.object(
      export.concurrent.futures, "ThreadPoolExecutor", autospec=True
  )
  def test_perf_metrics_export_shutdown_can_be_called_manually(
      self, mock_executor_cls
  ):
    # [FUNCTION DOC]
    # Verifies manual shutdown(wait=False) forwards wait and sets _executor None.
    mock_executor_instance = mock_executor_cls.return_value
    exporter = export.PerfMetricsExport(enable_trace_writer=True)
    exporter.shutdown(wait=False)
    mock_executor_instance.shutdown.assert_called_once_with(wait=False)
    self.assertIsNone(exporter._executor)

  @mock.patch.object(
      export.trace_writer_lib,
      "PerfettoTraceWriter",
      autospec=True,
  )
  def test_from_cluster_config(self, mock_writer_cls):
    # [FUNCTION DOC]
    # Verifies from_cluster_config maps roles to flattened mesh device lists
    # ({'actor': tpu0..3, 'rollout': tpu4..7}) passed to PerfettoTraceWriter.
    import numpy as np
    mock_mesh_1 = mock.create_autospec(jax.sharding.Mesh, instance=True)
    mock_mesh_1.devices = np.array([["tpu0", "tpu1"], ["tpu2", "tpu3"]])

    mock_mesh_2 = mock.create_autospec(jax.sharding.Mesh, instance=True)
    mock_mesh_2.devices = np.array([["tpu4", "tpu5"], ["tpu6", "tpu7"]])

    mock_cluster_config = mock.create_autospec(
        rl_cluster.ClusterConfig, instance=True
    )
    mock_cluster_config.role_to_mesh = {
        rl_cluster.Role.ACTOR: mock_mesh_1,
        rl_cluster.Role.ROLLOUT: mock_mesh_2,
    }

    exporter = export.PerfMetricsExport.from_cluster_config(
        mock_cluster_config,
        enable_trace_writer=True,
        trace_dir="/test/dir",
    )

    expected_role_to_devices = {
        "actor": ["tpu0", "tpu1", "tpu2", "tpu3"],
        "rollout": ["tpu4", "tpu5", "tpu6", "tpu7"],
    }
    mock_writer_cls.assert_called_once_with(
        "/test/dir", role_to_devices=expected_role_to_devices
    )
    self.assertIs(exporter._writer, mock_writer_cls.return_value)

  def test_safe_write_exception(self):
    # [FUNCTION DOC]
    # Verifies _safe_write logs ERROR 'Background trace export failed.' when the
    # writer raises.
    with export.PerfMetricsExport(enable_trace_writer=False) as exporter:
      with mock.patch.object(exporter, "_writer", autospec=True) as mock_writer:
        mock_writer.write_timelines.side_effect = ValueError("test error")
        with self.assertLogs(level="ERROR") as logs:
          exporter._safe_write({})
        self.assertTrue(
            any("Background trace export failed." in log for log in logs.output)
        )

  def test_export_metrics_after_shutdown(self):
    # [FUNCTION DOC]
    # Verifies export_metrics after shutdown logs a WARNING that the background
    # worker has been shut down.
    with export.PerfMetricsExport(enable_trace_writer=True) as exporter:
      exporter.shutdown(wait=False)
      with self.assertLogs(level="WARNING") as logs:
        exporter.export_metrics({})
      self.assertTrue(
          any(
              "PerfMetricsExport background worker has been shut down." in log
              for log in logs.output
          )
      )

  @mock.patch.object(
      export.trace_writer_lib,
      "PerfettoTraceWriter",
      autospec=True,
      spec_set=True,
  )
  def test_from_cluster_config_no_role_to_mesh(self, mock_writer_cls):
    # [FUNCTION DOC]
    # Verifies a cluster config lacking role_to_mesh yields role_to_devices {}.
    mock_cluster_config = mock.create_autospec(
        rl_cluster.ClusterConfig, instance=True, spec_set=True
    )
    del mock_cluster_config.role_to_mesh

    exporter = export.PerfMetricsExport.from_cluster_config(
        mock_cluster_config,
        enable_trace_writer=True,
        trace_dir="/test/dir",
    )

    mock_writer_cls.assert_called_once_with("/test/dir", role_to_devices={})
    self.assertIs(exporter._writer, mock_writer_cls.return_value)

if __name__ == "__main__":
  absltest.main()
