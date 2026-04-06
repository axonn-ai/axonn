"""
CUDA event-based timers for collective and pruning operations.

Environment variable (read once at import time, zero overhead when disabled):
    AXONN_TIME_OPS=1   — enable timing (default: 0)

Design
------
Each timer is either a ``_CudaOpTimer`` instance or ``None``.  Call sites
receive the timer as an ``Optional[_CudaOpTimer]`` argument and guard all
accesses with ``if timer is not None``, so when timing is off there is no
per-call overhead beyond a single ``None`` check.

Global named timers are pre-created here and registered in ``TIMERS`` so
that training_log can iterate over them to flush and log all at once.

Adding a new timer
------------------
    my_timer = _make_timer("my_timer")   # None when AXONN_TIME_OPS=0

Then pass ``my_timer`` into the call site as an argument.

Flushing at iteration boundaries
---------------------------------
    from axonn.op_timers import flush_all_and_get_ms
    name_to_ms = flush_all_and_get_ms()   # {name: ms_from_prev_iter}

The deferred read is safe because by the time training_log runs the GPU has
moved past all events from the prior iteration.
"""
import os
from typing import Dict, Optional

import torch

# Read once at import — no per-call overhead.
_ENABLED: bool = os.environ.get("AXONN_TIME_OPS", "0") == "1"

# Registry: name -> timer (populated only when _ENABLED).
TIMERS: Dict[str, "_CudaOpTimer"] = {}


class _CudaOpTimer:
    """
    Accumulates CUDA (start, end) event pairs across one iteration.

    Call ``flush_and_get_ms()`` at the iteration boundary to read the
    total elapsed ms from the *previous* iteration and rotate the buffers.
    """

    __slots__ = ("_pending", "_current", "_cur_start")

    def __init__(self):
        self._pending: list = []   # event pairs from last iteration
        self._current: list = []   # event pairs accumulating this iteration
        self._cur_start = None

    def start(self, stream=None):
        """Record a timing-start event on *stream* (default: current stream)."""
        e = torch.cuda.Event(enable_timing=True)
        e.record(stream)
        self._cur_start = e

    def stop(self, stream=None):
        """Record a timing-end event on *stream* (default: current stream)."""
        e = torch.cuda.Event(enable_timing=True)
        e.record(stream)
        self._current.append((self._cur_start, e))
        self._cur_start = None

    def flush_and_get_ms(self) -> float:
        """
        Return total elapsed ms for the *previous* iteration and rotate
        buffers so current-iteration pairs become pending.
        """
        total = 0.0
        for s, e in self._pending:
            try:
                total += s.elapsed_time(e)
            except RuntimeError:
                pass  # should not happen at iter boundary, skip gracefully
        self._pending = self._current
        self._current = []
        return total

    def reset(self):
        """Discard all accumulated events."""
        self._pending = []
        self._current = []
        self._cur_start = None


def _make_timer(name: str) -> Optional[_CudaOpTimer]:
    """Return a new _CudaOpTimer registered under *name*, or None if disabled."""
    if not _ENABLED:
        return None
    t = _CudaOpTimer()
    TIMERS[name] = t
    return t


def flush_all_and_get_ms() -> Dict[str, float]:
    """Flush every registered timer and return {name: ms} for the previous iteration."""
    return {name: t.flush_and_get_ms() for name, t in TIMERS.items()}


# ---------------------------------------------------------------------------
# Named global timers — None when AXONN_TIME_OPS is not set.
# ---------------------------------------------------------------------------

# Time spent in the DP all-reduce collective (sparse or dense).
allreduce_timer: Optional[_CudaOpTimer] = _make_timer("allreduce_timer")

# Time spent pruning gradients before the DP all-reduce.
dp_prune_timer: Optional[_CudaOpTimer] = _make_timer("dp_prune_timer")
