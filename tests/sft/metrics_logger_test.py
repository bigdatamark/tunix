# [MODULE DOC]
# Unit tests for tunix.sft.metrics_logger: MetricsLoggerOptions backend
# creation (default Tensorboard + W&B backends externally, CLU internally,
# custom backend factories via backend_kwargs['custom_backend'], per-backend
# kwargs) and MetricsLogger (registers each backend's log_scalar with
# jax.monitoring, records scalars as '<prefix>/<mode>/<name>', aggregates
# metric means / geometric mean for perplexity, keeps history, creates no
# backends on non-zero process index, closes backends on close()).
# Side effects: temp directories; backend classes are mocked.
"""Metrics logger unittest."""

import copy
import os
import tempfile
from unittest import mock

from absl.testing import absltest
import jax
import metrax.logging as metrax_logging
from tunix.sft import metrics_logger
from tunix.utils import env_utils


class MetricLoggerTest(absltest.TestCase):
  # [CLASS DOC]
  # Test case for MetricsLogger. Fixture patches the default backend classes
  # (CluBackend if internal env, else TensorboardBackend and WandbBackend) and
  # stores the mocks in self.mock_backends in that order.

  def setUp(self):
    # [FUNCTION DOC]
    # HELPER: fixture. Creates a TemporaryDirectory (kept on self so it lives for
    # the test), sets self.log_dir, and patches backend classes via enter_context
    # so patches are undone at test teardown.
    super().setUp()
    self._temp_dir_obj = tempfile.TemporaryDirectory()
    self.log_dir = self._temp_dir_obj.name
    self.mock_backends = []

    if env_utils.is_internal_env():
      self.mock_backends.append(
          self.enter_context(
              mock.patch.object(metrics_logger, "CluBackend")
          )
      )
    else:
      self.mock_backends.append(
          self.enter_context(
              mock.patch.object(metrics_logger, "TensorboardBackend")
          )
      )
      self.mock_backends.append(
          self.enter_context(
              mock.patch.object(metrics_logger, "WandbBackend")
          )
      )

  @mock.patch.object(jax.monitoring, "register_scalar_listener")
  def test_custom_backends_override_defaults(self, mock_register):
    # [FUNCTION DOC]
    # Verifies a 'custom_backend' factory list replaces defaults: no default
    # backend constructed, factory called once, only its instance in
    # logger._backends, its log_scalar registered as listener, and close()
    # closes it. Input: patched register_scalar_listener.
    """Tests that providing a 'backends' list overrides the defaults."""
    mock_backend_instance = mock.Mock(spec=metrax_logging.LoggingBackend)
    mock_factory = mock.Mock(return_value=mock_backend_instance)
    options = metrics_logger.MetricsLoggerOptions(
        log_dir=self.log_dir,
        backend_kwargs={"custom_backend": [mock_factory]},
    )

    logger = metrics_logger.MetricsLogger(metrics_logger_options=options)
    for backend_mock in self.mock_backends:
      backend_mock.assert_not_called()
    mock_factory.assert_called_once()
    self.assertIn(mock_backend_instance, logger._backends)
    self.assertLen(logger._backends, 1)
    mock_register.assert_called_once_with(mock_backend_instance.log_scalar)

    logger.close()
    mock_backend_instance.close.assert_called_once()

  @mock.patch.object(jax.monitoring, "register_scalar_listener")
  def test_defaults_are_used_when_no_backends_provided(self, mock_register):
    # [FUNCTION DOC]
    # Verifies default backends are each constructed once, registered as
    # listeners, stored in logger._backends, and closed by close().
    """Tests that defaults are created when 'backends' is None."""
    options = metrics_logger.MetricsLoggerOptions(log_dir=self.log_dir)

    logger = metrics_logger.MetricsLogger(metrics_logger_options=options)

    for backend_mock in self.mock_backends:
      backend_mock.assert_called_once()
      self.assertIn(backend_mock.return_value, logger._backends)

    self.assertLen(logger._backends, len(self.mock_backends))
    self.assertEqual(mock_register.call_count, len(self.mock_backends))

    logger.close()
    for backend_mock in self.mock_backends:
      backend_mock.return_value.close.assert_called_once()

  @mock.patch.object(jax.monitoring, "register_scalar_listener")
  def test_logger_handles_missing_wandb_gracefully(self, mock_register):
    # [FUNCTION DOC]
    # Verifies (external env only; returns early internally) that an ImportError
    # from the W&B backend constructor is tolerated: only Tensorboard remains.
    """Tests that the logger doesn't crash if wandb is not installed."""
    # wandb is not supported in internal environment.
    if env_utils.is_internal_env():
      return

    mock_tensorboard_instance = self.mock_backends[0].return_value
    mock_wandb_backend = self.mock_backends[1]
    mock_wandb_backend.side_effect = ImportError("W&B not installed")
    options = metrics_logger.MetricsLoggerOptions(log_dir=self.log_dir)

    logger = metrics_logger.MetricsLogger(metrics_logger_options=options)
    self.assertIn(mock_tensorboard_instance, logger._backends)
    self.assertLen(logger._backends, 1)
    mock_register.assert_called_once_with(mock_tensorboard_instance.log_scalar)

    logger.close()
    mock_tensorboard_instance.close.assert_called_once()

  def test_options_deepcopy_safety(self):
    # [FUNCTION DOC]
    # Verifies deep-copying options and changing log_dir produces a second logger
    # whose backends are constructed again with the new log_dir (Tensorboard also
    # with flush_every_n_steps=100).
    """Tests that deepcopying options and creating new loggers is safe."""
    options1 = metrics_logger.MetricsLoggerOptions(log_dir=self.log_dir)
    logger1 = metrics_logger.MetricsLogger(metrics_logger_options=options1)
    for backend_mock in self.mock_backends:
      self.assertEqual(backend_mock.call_count, 1)

    options2 = copy.deepcopy(options1)
    new_log_dir = os.path.join(self.log_dir, "critic")
    options2.log_dir = new_log_dir
    logger2 = metrics_logger.MetricsLogger(metrics_logger_options=options2)

    for backend_mock in self.mock_backends:
      self.assertEqual(backend_mock.call_count, 2)

    if env_utils.is_internal_env():
      self.mock_backends[0].assert_called_with(log_dir=new_log_dir)
    else:
      self.mock_backends[0].assert_called_with(
          log_dir=new_log_dir,
          flush_every_n_steps=100,
      )

    logger1.close()
    logger2.close()

  @mock.patch.object(jax.monitoring, "record_scalar")
  def test_log_metrics(self, mock_record_scalar):
    # [FUNCTION DOC]
    # Verifies log() forwards to jax.monitoring.record_scalar with name
    # 'test_prefix/train/loss' and step, and that metric_exists, get_metric
    # (mean 0.075 of 0.1 and 0.05) and get_metric_history work. No backends.
    options = metrics_logger.MetricsLoggerOptions(
        log_dir=self.log_dir, backend_kwargs={"custom_backend": []}
    )
    logger = metrics_logger.MetricsLogger(metrics_logger_options=options)
    logger.log("test_prefix", "loss", 0.1, metrics_logger.Mode.TRAIN, 1)
    logger.log("test_prefix", "loss", 0.05, metrics_logger.Mode.TRAIN, 2)
    mock_record_scalar.assert_has_calls([
        mock.call("test_prefix/train/loss", 0.1, step=1),
        mock.call("test_prefix/train/loss", 0.05, step=2),
    ])
    self.assertTrue(logger.metric_exists("test_prefix", "loss", "train"))
    self.assertAlmostEqual(
        logger.get_metric("test_prefix", "loss", "train"), 0.075
    )
    history = logger.get_metric_history("test_prefix", "loss", "train")
    self.assertLen(history, 2)
    self.assertAlmostEqual(history[0], 0.1)
    self.assertAlmostEqual(history[1], 0.05)

  @mock.patch.object(jax.monitoring, "record_scalar")
  def test_log_perplexity(self, mock_record_scalar):
    # [FUNCTION DOC]
    # Verifies perplexity aggregation uses the geometric mean
    # (sqrt(10*100) = 31.6227766) rather than the arithmetic mean.
    options = metrics_logger.MetricsLoggerOptions(
        log_dir=self.log_dir, backend_kwargs={"custom_backend": []}
    )
    logger = metrics_logger.MetricsLogger(metrics_logger_options=options)
    logger.log("test_prefix", "perplexity", 10.0, metrics_logger.Mode.EVAL, 1)
    logger.log("test_prefix", "perplexity", 100.0, metrics_logger.Mode.EVAL, 2)
    mock_record_scalar.assert_has_calls([
        mock.call("test_prefix/eval/perplexity", 10.0, step=1),
        mock.call("test_prefix/eval/perplexity", 100.0, step=2),
    ])
    self.assertTrue(logger.metric_exists("test_prefix", "perplexity", "eval"))
    self.assertAlmostEqual(
        logger.get_metric("test_prefix", "perplexity", "eval"), 31.6227766
    )

  @mock.patch.object(jax, "process_index", return_value=1)
  @mock.patch.object(jax.monitoring, "register_scalar_listener")
  def test_no_backends_on_secondary_process(
      self,
      mock_register,
      mock_jax_process_index,
  ):
    # [FUNCTION DOC]
    # Verifies that with jax.process_index patched to 1 no backends are created
    # or registered and close() still works.
    del mock_jax_process_index
    options = metrics_logger.MetricsLoggerOptions(log_dir=self.log_dir)
    logger = metrics_logger.MetricsLogger(metrics_logger_options=options)
    for backend_mock in self.mock_backends:
      backend_mock.assert_not_called()
    self.assertEmpty(logger._backends)
    mock_register.assert_not_called()
    logger.close()

  @mock.patch.object(env_utils, "is_internal_env", return_value=True)
  def test_raises_when_clu_backend_missing_in_internal_env(
      self, mock_is_internal_env
  ):
    # [FUNCTION DOC]
    # Verifies create_backends raises ImportError ('Internal environment
    # detected, but CluBackend not available.') when is_internal_env is True and
    # CluBackend is None.
    del mock_is_internal_env
    # We need to patch CluBackend to be None in the module.
    with mock.patch.object(metrics_logger, "CluBackend", new=None):
      options = metrics_logger.MetricsLoggerOptions(log_dir=self.log_dir)
      with self.assertRaisesRegex(
          ImportError,
          "Internal environment detected, but CluBackend not available.",
      ):
        options.create_backends()

  def test_backend_kwargs_are_passed_to_backends(self):
    # [FUNCTION DOC]
    # Verifies backend_kwargs are forwarded: internally CLU gets log_dir plus
    # 'clu' kwargs; externally Tensorboard gets log_dir, flush_every_n_steps=100
    # and 'tensorboard' kwargs, and W&B gets project 'tunix', name '' and the
    # 'wandb' kwargs.
    """Tests that backend_kwargs are passed to the backends initialization."""
    mock_wandb = mock.Mock()
    options = metrics_logger.MetricsLoggerOptions(
        log_dir=self.log_dir,
        backend_kwargs={
            "wandb": {
                "resume": "must",
                "id": "12345",
                "settings": mock_wandb.Settings(console="off"),
            },
            "tensorboard": {"flush_interval_s": 10.0},
            "clu": {"some_arg": "value"},
        },
    )

    # Trigger backend creation
    _ = metrics_logger.MetricsLogger(metrics_logger_options=options)

    if env_utils.is_internal_env():
      self.mock_backends[0].assert_called_once_with(
          log_dir=self.log_dir, **options.backend_kwargs["clu"]
      )
    else:
      self.mock_backends[0].assert_called_once_with(
          log_dir=self.log_dir,
          flush_every_n_steps=100,
          **options.backend_kwargs["tensorboard"],
      )
      self.mock_backends[1].assert_called_once_with(
          project="tunix",
          name="",
          resume="must",
          id="12345",
          settings=mock_wandb.Settings.return_value,
      )


if __name__ == "__main__":
  absltest.main()
