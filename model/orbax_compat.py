"""Compatibility for throughput logging in the pinned Orbax release."""

from importlib.metadata import version
import math
import time

from absl import logging
import jax


def _log_io_per_sec_metric(name, size, start_time):
    # Orbax 0.6.4 formats NaN with humanize when the wall clock has not
    # advanced. On Windows this can abort an otherwise valid checkpoint save.
    # An unmeasurable rate is omitted; serialization errors are never caught.
    elapsed = time.time() - start_time
    if not math.isfinite(elapsed) or elapsed <= 0:
        return
    rate = float(size) / elapsed
    if not math.isfinite(rate):
        return
    logging.info('%s: %.3f bytes/s (total bytes: %s, elapsed: %.6f s)',
                 name, rate, size, elapsed)
    jax.monitoring.record_event_duration_secs(name, rate)


def install_checkpoint_compat():
    """Change only the affected diagnostic logger, not the checkpoint format."""
    if version('orbax-checkpoint') == '0.6.4':
        from orbax.checkpoint._src.handlers import base_pytree_checkpoint_handler
        base_pytree_checkpoint_handler._log_io_per_sec_metric = _log_io_per_sec_metric
