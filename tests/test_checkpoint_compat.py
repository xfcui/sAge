"""Regression checks for Orbax's zero-duration checkpoint I/O metric."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model'))
import orbax_compat as checkpoint_compat


class CheckpointMetricTests(unittest.TestCase):
    def test_unmeasurable_rates_are_skipped(self):
        for now in (100.0, 99.0, float('nan'), float('inf')):
            with self.subTest(now=now), \
                    patch.object(checkpoint_compat.time, 'time', return_value=now), \
                    patch.object(checkpoint_compat.jax.monitoring,
                                 'record_event_duration_secs') as record:
                checkpoint_compat._log_io_per_sec_metric('save', 1024, 100.0)
                record.assert_not_called()

    def test_valid_rate_is_recorded(self):
        with patch.object(checkpoint_compat.time, 'time', return_value=102.0), \
                patch.object(checkpoint_compat.jax.monitoring,
                             'record_event_duration_secs') as record:
            checkpoint_compat._log_io_per_sec_metric('save', 1024, 100.0)
            record.assert_called_once_with('save', 512.0)

    def test_pinned_handler_is_patched(self):
        from orbax.checkpoint._src.handlers import base_pytree_checkpoint_handler
        checkpoint_compat.install_checkpoint_compat()
        self.assertIs(base_pytree_checkpoint_handler._log_io_per_sec_metric,
                      checkpoint_compat._log_io_per_sec_metric)


if __name__ == '__main__':
    unittest.main()
