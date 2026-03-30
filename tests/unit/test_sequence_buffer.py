# test/unit/test_sequence_buffer.py

# Purpose: Unit tests for the SequenceBuffer class to verify correct buffering, windowing, and emission logic.

# Input: None (test code only)

# Output: Test results (pass/fail) when run with pytest.

# Used by: N/A (these are unit tests for the SequenceBuffer class, 
# indirectly used by the runtime calibration script and 
# any real-time inference components that consume sequences from the buffer)

"""Unit tests for SequenceBuffer."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.runtime.sequence_buffer import SequenceBuffer
from src.data_layer.models import LogEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dict_event(i: int, service: str = "bgl", session: str = "s1") -> dict:
    return {
        "timestamp":   float(i),
        "service":     service,
        "session_id":  session,
        "token_id":    (i % 100) + 2,   # ensure token_id >= 2
        "label":       0,
    }


def _make_log_event(i: int, service: str = "bgl", session: str = "s1") -> LogEvent:
    return LogEvent(
        timestamp=float(i),
        service=service,
        level="INFO",
        message=f"msg {i}",
        meta={"session_id": session, "token_id": (i % 100) + 2},
        label=0,
    )


# ---------------------------------------------------------------------------
# stream_key_for
# ---------------------------------------------------------------------------

class TestStreamKeyFor:
    def test_dict_event_key(self):
        buf = SequenceBuffer(window_size=5)
        ev = _make_dict_event(0)
        assert buf.stream_key_for(ev) == "bgl:s1"

    def test_log_event_key(self):
        buf = SequenceBuffer(window_size=5)
        ev = _make_log_event(0, service="hdfs", session="blk_123")
        assert buf.stream_key_for(ev) == "hdfs:blk_123"

    def test_missing_fields_default(self):
        buf = SequenceBuffer(window_size=5)
        assert buf.stream_key_for({}) == "unknown:"


# ---------------------------------------------------------------------------
# No emit before window is full
# ---------------------------------------------------------------------------

class TestNoEmitBeforeFull:
    def test_no_emit_below_window_size(self):
        buf = SequenceBuffer(window_size=10, stride=1)
        key = None
        for i in range(9):
            key = buf.ingest(_make_dict_event(i))
        assert key is not None
        assert not buf.should_emit(key)

    def test_buffer_length_grows(self):
        buf = SequenceBuffer(window_size=10, stride=5)
        key = None
        for i in range(7):
            key = buf.ingest(_make_dict_event(i))
        assert buf.buffer_length(key) == 7


# ---------------------------------------------------------------------------
# Emit at window fill
# ---------------------------------------------------------------------------

class TestEmitAtFill:
    def test_first_emit_at_window_size(self):
        W = 10
        buf = SequenceBuffer(window_size=W, stride=5)
        key = None
        for i in range(W):
            key = buf.ingest(_make_dict_event(i))
        assert buf.should_emit(key)

    def test_sequence_has_correct_length(self):
        W = 8
        buf = SequenceBuffer(window_size=W, stride=4)
        key = None
        for i in range(W):
            key = buf.ingest(_make_dict_event(i))
        seq = buf.get_window(key)
        assert len(seq.tokens) == W

    def test_tokens_are_ints(self):
        buf = SequenceBuffer(window_size=5, stride=1)
        key = None
        for i in range(5):
            key = buf.ingest(_make_dict_event(i))
        seq = buf.get_window(key)
        assert all(isinstance(t, int) for t in seq.tokens)

    def test_sequence_id_contains_key(self):
        buf = SequenceBuffer(window_size=5, stride=1)
        key = None
        for i in range(5):
            key = buf.ingest(_make_dict_event(i))
        seq = buf.get_window(key)
        assert seq.sequence_id.startswith(key)

    def test_timestamps_populated(self):
        buf = SequenceBuffer(window_size=5, stride=5)
        key = None
        for i in range(5):
            key = buf.ingest(_make_dict_event(i))
        seq = buf.get_window(key)
        assert len(seq.timestamps) == 5
        assert seq.timestamps[0] == 0.0
        assert seq.timestamps[-1] == 4.0


# ---------------------------------------------------------------------------
# Stride behaviour
# ---------------------------------------------------------------------------

class TestStrideBehaviour:
    def test_no_extra_emit_within_stride(self):
        W, S = 10, 5
        buf = SequenceBuffer(window_size=W, stride=S)
        key = None
        emits = 0
        for i in range(W + S - 1):   # W + S - 1 events: should emit exactly once
            key = buf.ingest(_make_dict_event(i))
            if buf.should_emit(key):
                emits += 1
                buf.get_window(key)   # consume the window
        assert emits == 1

    def test_second_emit_at_W_plus_S(self):
        W, S = 10, 5
        buf = SequenceBuffer(window_size=W, stride=S)
        key = None
        emits = 0
        for i in range(W + S):       # exactly W + S events: should emit twice
            key = buf.ingest(_make_dict_event(i))
            if buf.should_emit(key):
                emits += 1
                buf.get_window(key)
        assert emits == 2

    def test_emit_count_increments(self):
        W, S = 5, 5
        buf = SequenceBuffer(window_size=W, stride=S)
        key = None
        for i in range(W * 3):
            key = buf.ingest(_make_dict_event(i))
        # Should have emitted at events 5, 10, 15  → 3 times
        assert buf._emit_counts.get(key, 0) == 0  # not consumed yet
        # Manually consume all pending emissions
        consumed = 0
        for _ in range(3):
            if buf.should_emit(key):
                buf.get_window(key)
                consumed += 1
        # After W*3 events with stride S=5: emit at 5,10,15 → 3 emits
        assert buf._emit_counts.get(key, 0) == 3


# ---------------------------------------------------------------------------
# Multiple independent keys
# ---------------------------------------------------------------------------

class TestMultipleKeys:
    def test_keys_do_not_interfere(self):
        buf = SequenceBuffer(window_size=5, stride=1)
        for i in range(5):
            buf.ingest(_make_dict_event(i, session="A"))
        for j in range(3):
            buf.ingest(_make_dict_event(j, session="B"))

        key_a = buf.stream_key_for(_make_dict_event(0, session="A"))
        key_b = buf.stream_key_for(_make_dict_event(0, session="B"))

        assert buf.should_emit(key_a)
        assert not buf.should_emit(key_b)

    def test_two_different_services(self):
        buf = SequenceBuffer(window_size=4, stride=2)
        for i in range(4):
            buf.ingest(_make_dict_event(i, service="hdfs", session="blk_1"))
        for j in range(2):
            buf.ingest(_make_dict_event(j, service="bgl",  session="win_0"))

        k_hdfs = buf.stream_key_for(_make_dict_event(0, service="hdfs", session="blk_1"))
        k_bgl  = buf.stream_key_for(_make_dict_event(0, service="bgl",  session="win_0"))

        assert buf.should_emit(k_hdfs)
        assert not buf.should_emit(k_bgl)


# ---------------------------------------------------------------------------
# Label aggregation
# ---------------------------------------------------------------------------

class TestLabelAggregation:
    def test_all_normal_label_is_0(self):
        buf = SequenceBuffer(window_size=5, stride=5)
        key = None
        for i in range(5):
            ev = _make_dict_event(i)
            ev["label"] = 0
            key = buf.ingest(ev)
        seq = buf.get_window(key)
        assert seq.label == 0

    def test_one_anomaly_makes_label_1(self):
        buf = SequenceBuffer(window_size=5, stride=5)
        key = None
        for i in range(5):
            ev = _make_dict_event(i)
            ev["label"] = 1 if i == 3 else 0
            key = buf.ingest(ev)
        seq = buf.get_window(key)
        assert seq.label == 1

    def test_no_label_is_none(self):
        buf = SequenceBuffer(window_size=3, stride=3)
        key = None
        for i in range(3):
            ev = {"service": "bgl", "session_id": "x", "token_id": 5}
            key = buf.ingest(ev)
        seq = buf.get_window(key)
        assert seq.label is None


# ---------------------------------------------------------------------------
# LogEvent compatibility
# ---------------------------------------------------------------------------

class TestLogEventCompatibility:
    def test_ingest_log_events(self):
        buf = SequenceBuffer(window_size=5, stride=5)
        key = None
        for i in range(5):
            key = buf.ingest(_make_log_event(i))
        assert buf.should_emit(key)
        seq = buf.get_window(key)
        assert len(seq.tokens) == 5

    def test_token_from_meta(self):
        buf = SequenceBuffer(window_size=3, stride=3)
        key = None
        for i in range(3):
            ev = LogEvent(
                timestamp=float(i), service="bgl", level="",
                message="m", meta={"session_id": "s", "token_id": 42 + i},
            )
            key = buf.ingest(ev)
        seq = buf.get_window(key)
        assert seq.tokens[0] == 42


# ---------------------------------------------------------------------------
# reset / clear
# ---------------------------------------------------------------------------

class TestResetClear:
    def test_reset_single_key(self):
        buf = SequenceBuffer(window_size=5, stride=1)
        key = None
        for i in range(5):
            key = buf.ingest(_make_dict_event(i))
        assert buf.should_emit(key)
        buf.reset(key)
        assert not buf.should_emit(key)
        assert buf.buffer_length(key) == 0

    def test_clear_removes_all_keys(self):
        buf = SequenceBuffer(window_size=3, stride=1)
        for i in range(3):
            buf.ingest(_make_dict_event(i, session="A"))
        for i in range(3):
            buf.ingest(_make_dict_event(i, session="B"))
        assert len(buf) == 2
        buf.clear()
        assert len(buf) == 0


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------

class TestMaxStreamKeys:
    def test_eviction_at_capacity(self):
        buf = SequenceBuffer(window_size=3, stride=1, max_stream_keys=3)
        # Fill 3 keys
        for sid in ["A", "B", "C"]:
            for i in range(2):
                buf.ingest(_make_dict_event(i, session=sid))
        assert len(buf) == 3
        # Adding a 4th key should evict "A" (oldest)
        buf.ingest(_make_dict_event(0, session="D"))
        assert len(buf) == 3
        remaining = buf.active_keys()
        assert not any(":A" in k for k in remaining), f"A should be evicted, got {remaining}"
