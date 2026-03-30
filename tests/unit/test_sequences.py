# test/unit/test_sequences.py

# Purpose: Unit tests for the sequence building logic, 
# including SlidingWindowSequenceBuilder, 
# SessionSequenceBuilder, and DatasetSplitter.

# Input: None (test code only)

# Output: Test results (pass/fail) when run with pytest.

# Used by: N/A (these are unit tests for the sequence building logic, 
# indirectly used by the sequence building scripts 
# and any downstream models that consume the sequences).


"""Unit tests for Sequence builders and DatasetSplitter."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.sequencing import (
    Sequence,
    SlidingWindowSequenceBuilder,
    SessionSequenceBuilder,
    DatasetSplitter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_events_df():
    """20 events with sequential template_ids (0..19) and mixed labels."""
    return pd.DataFrame({
        "template_id": list(range(20)),
        "label":       [0] * 15 + [1] * 5,
        "timestamp":   [float(i) for i in range(20)],
    })


@pytest.fixture()
def session_csv_df():
    """Simulates a loaded session_sequences_v2.csv."""
    return pd.DataFrame({
        "session_id":                 ["s1", "s2", "s3"],
        "ordered_template_sequence": ["1,2,3", "4,5", ""],
        "label":                     [0, 1, 0],
    })


# ---------------------------------------------------------------------------
# SlidingWindowSequenceBuilder
# ---------------------------------------------------------------------------

class TestSlidingWindowSequenceBuilder:
    def test_basic_count(self, small_events_df):
        builder = SlidingWindowSequenceBuilder(window=5, stride=5)
        seqs = builder.build(small_events_df)
        assert len(seqs) == 4   # windows at 0,5,10,15

    def test_window_length(self, small_events_df):
        builder = SlidingWindowSequenceBuilder(window=5, stride=5)
        seqs = builder.build(small_events_df)
        assert all(len(s.tokens) == 5 for s in seqs)

    def test_stride_one(self, small_events_df):
        builder = SlidingWindowSequenceBuilder(window=5, stride=1)
        seqs = builder.build(small_events_df)
        # 20 - 5 + 1 = 16 windows
        assert len(seqs) == 16

    def test_sequence_ids(self, small_events_df):
        builder = SlidingWindowSequenceBuilder(window=5, stride=5)
        seqs = builder.build(small_events_df)
        assert seqs[0].sequence_id == "window_0"
        assert seqs[1].sequence_id == "window_5"

    def test_label_is_max(self, small_events_df):
        builder = SlidingWindowSequenceBuilder(window=5, stride=5,
                                               label_col="label")
        seqs = builder.build(small_events_df)
        # window 3 (rows 15-19) contains labels [0,0,0,1,1] -> max=1
        assert seqs[3].label == 1
        assert seqs[0].label == 0

    def test_tokens_are_ints(self, small_events_df):
        builder = SlidingWindowSequenceBuilder(window=5, stride=5)
        seqs = builder.build(small_events_df)
        assert all(isinstance(t, int) for t in seqs[0].tokens)

    def test_iter_build(self, small_events_df):
        builder = SlidingWindowSequenceBuilder(window=5, stride=5)
        seqs_list = builder.build(small_events_df)
        seqs_iter = list(builder.iter_build(small_events_df))
        assert len(seqs_list) == len(seqs_iter)


# ---------------------------------------------------------------------------
# SessionSequenceBuilder
# ---------------------------------------------------------------------------

class TestSessionSequenceBuilder:
    def test_build_basic(self, session_csv_df):
        builder = SessionSequenceBuilder()
        seqs = builder.build(session_csv_df)
        assert len(seqs) == 3

    def test_tokens_parsed(self, session_csv_df):
        builder = SessionSequenceBuilder()
        seqs = builder.build(session_csv_df)
        assert seqs[0].tokens == [1, 2, 3]
        assert seqs[1].tokens == [4, 5]

    def test_empty_sequence(self, session_csv_df):
        builder = SessionSequenceBuilder()
        seqs = builder.build(session_csv_df)
        assert seqs[2].tokens == []

    def test_labels(self, session_csv_df):
        builder = SessionSequenceBuilder()
        seqs = builder.build(session_csv_df)
        assert seqs[0].label == 0
        assert seqs[1].label == 1

    def test_tokenizer_applied(self, session_csv_df):
        from src.parsing import EventTokenizer
        tok = EventTokenizer()
        tok._tid_to_text = {1: "a", 2: "b", 3: "c", 4: "d", 5: "e"}
        tok._text_to_tid = {v: k for k, v in tok._tid_to_text.items()}
        tok._sorted_tids = [1, 2, 3, 4, 5]
        builder = SessionSequenceBuilder(tokenizer=tok)
        seqs = builder.build(session_csv_df)
        # tid 1 -> token 3, tid 2 -> 4, tid 3 -> 5
        assert seqs[0].tokens == [3, 4, 5]


# ---------------------------------------------------------------------------
# DatasetSplitter
# ---------------------------------------------------------------------------

class TestDatasetSplitter:
    @pytest.fixture()
    def mixed_sequences(self):
        seqs = []
        for i in range(80):
            seqs.append(Sequence(f"s{i}", [i], label=0))
        for i in range(20):
            seqs.append(Sequence(f"a{i}", [i + 100], label=1))
        return seqs

    def test_stratified_total(self, mixed_sequences):
        splitter = DatasetSplitter(val_ratio=0.1, test_ratio=0.1)
        train, val, test = splitter.split_stratified(mixed_sequences)
        assert len(train) + len(val) + len(test) == len(mixed_sequences)

    def test_stratified_label_balance(self, mixed_sequences):
        splitter = DatasetSplitter(val_ratio=0.1, test_ratio=0.1)
        train, val, test = splitter.split_stratified(mixed_sequences)
        # test should have ~10% of each label
        test_labels = [s.label for s in test]
        assert 1 in test_labels   # anomalies present in test

    def test_time_based_total(self, mixed_sequences):
        splitter = DatasetSplitter(val_ratio=0.1, test_ratio=0.1)
        train, val, test = splitter.split_time_based(mixed_sequences)
        assert len(train) + len(val) + len(test) == len(mixed_sequences)

    def test_time_based_order(self, mixed_sequences):
        splitter = DatasetSplitter(val_ratio=0.1, test_ratio=0.1)
        train, val, test = splitter.split_time_based(mixed_sequences)
        # First element of train should be the first sequence
        assert train[0].sequence_id == mixed_sequences[0].sequence_id

    def test_invalid_ratios(self):
        with pytest.raises(ValueError):
            DatasetSplitter(val_ratio=0.6, test_ratio=0.6)
