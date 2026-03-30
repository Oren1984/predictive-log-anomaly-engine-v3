# src/runtime/sequence_buffer.py

# Purpose: Define the SequenceBuffer class, which maintains a rolling window of 
# 
# log events for each stream key (service:session_id) during runtime inference. 
# The SequenceBuffer allows the InferenceEngine to determine when to emit 
# a window of events for scoring and to construct the Sequence objects 
# that are fed into the anomaly detection models.

# Input: - window_size: The number of events that constitute a full window for scoring.

# Output: - SequenceBuffer: An instance of the SequenceBuffer class that can be used to ingest log events, 
# determine when to emit windows for scoring, and construct Sequence objects for the InferenceEngine.

# Used by: The SequenceBuffer class is used by the InferenceEngine in the runtime stage 
# to maintain rolling windows of log events for each stream key (service:session_id). 
# It allows the InferenceEngine to determine when to emit a window of events for scoring 
# and to construct the Sequence objects that are fed into the anomaly detection models.

"""Stage 5 — Runtime: rolling token-window buffer."""
from __future__ import annotations

from collections import deque
from typing import Optional

from ..data_layer.models import LogEvent
from ..sequencing.models import Sequence


class SequenceBuffer:
    """
    Maintain a rolling token window per stream key (service:session_id).

    Parameters
    ----------
    window_size     : number of events per window (tokens in the Sequence)
    stride          : emit a window every `stride` new events once the
                      window is full (stride=1 → emit on every new event)
    max_stream_keys : LRU cap; oldest key is evicted when limit is reached
    """

    def __init__(
        self,
        window_size: int,
        stride: int = 1,
        max_stream_keys: int = 5000,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.window_size = window_size
        self.stride = stride
        self.max_stream_keys = max_stream_keys

        # insertion-ordered dicts act as LRU when we pop the first key
        self._buffers: dict[str, deque] = {}
        self._total_counts: dict[str, int] = {}   # total events ingested per key
        self._emit_counts: dict[str, int] = {}    # windows emitted per key

    # ------------------------------------------------------------------
    # Key derivation
    # ------------------------------------------------------------------

    def stream_key_for(self, event: LogEvent | dict) -> str:
        """Return the canonical stream key for *event*."""
        if isinstance(event, LogEvent):
            svc = event.service or "unknown"
            sid = event.meta.get("session_id", "")
        else:
            svc = event.get("service", "unknown") or "unknown"
            sid = event.get("session_id", "") or ""
        return f"{svc}:{sid}"

    # ------------------------------------------------------------------
    # Core public API
    # ------------------------------------------------------------------

    def ingest(self, event: LogEvent | dict) -> str:
        """
        Add *event* to the rolling buffer.

        Returns
        -------
        stream_key : the key that was updated (use with should_emit / get_window)
        """
        key = self.stream_key_for(event)

        if key not in self._buffers:
            if len(self._buffers) >= self.max_stream_keys:
                self._evict_oldest()
            self._buffers[key] = deque(maxlen=self.window_size)
            self._total_counts[key] = 0
            self._emit_counts[key] = 0

        self._buffers[key].append(event)
        self._total_counts[key] += 1
        return key

    def should_emit(self, stream_key: str) -> bool:
        """
        Return True when the window for *stream_key* is full and a
        stride boundary has been reached.

        Emission schedule (window_size W, stride S):
          first emit at event #W, then every S events: W, W+S, W+2S, …
        """
        n = self._total_counts.get(stream_key, 0)
        if n < self.window_size:
            return False
        return (n - self.window_size) % self.stride == 0

    def get_window(self, stream_key: str) -> Sequence:
        """
        Build and return a Sequence from the current window contents.

        Side-effect: increments the emit counter for *stream_key*.
        """
        buf = list(self._buffers[stream_key])
        tokens: list[int] = []
        timestamps: list[float] = []
        labels: list[int] = []

        for ev in buf:
            if isinstance(ev, LogEvent):
                tok = ev.meta.get("token_id", 1)
                ts  = ev.timestamp if ev.timestamp is not None else 0.0
                lbl = ev.label
            else:
                tok = ev.get("token_id", 1)
                ts  = ev.get("timestamp") or 0.0
                lbl = ev.get("label")

            tokens.append(int(tok))
            try:
                timestamps.append(float(ts))
            except (TypeError, ValueError):
                timestamps.append(0.0)
            if lbl is not None:
                try:
                    labels.append(int(lbl))
                except (TypeError, ValueError):
                    pass

        label = max(labels) if labels else None
        seq_id = f"{stream_key}_w{self._emit_counts[stream_key]}"
        self._emit_counts[stream_key] += 1

        return Sequence(
            sequence_id=seq_id,
            tokens=tokens,
            timestamps=timestamps,
            label=label,
        )

    def reset(self, stream_key: str) -> None:
        """Clear buffer for a single stream key (keeps key registered)."""
        if stream_key in self._buffers:
            self._buffers[stream_key].clear()
            self._total_counts[stream_key] = 0

    def clear(self) -> None:
        """Remove all stream keys and their buffers."""
        self._buffers.clear()
        self._total_counts.clear()
        self._emit_counts.clear()

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def active_keys(self) -> list[str]:
        return list(self._buffers.keys())

    def buffer_length(self, stream_key: str) -> int:
        return len(self._buffers.get(stream_key, []))

    def __len__(self) -> int:
        return len(self._buffers)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_oldest(self) -> None:
        """Remove the key that was inserted first (FIFO eviction)."""
        if not self._buffers:
            return
        oldest = next(iter(self._buffers))
        self._buffers.pop(oldest, None)
        self._total_counts.pop(oldest, None)
        self._emit_counts.pop(oldest, None)
