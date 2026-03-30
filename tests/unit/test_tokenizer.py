# test/unit/test_tokenizer.py

# Purpose: Unit tests for EventTokenizer to verify
# correct encoding/decoding behavior, vocab handling, and edge cases.

# Input: None (test code only)

# Output: Test results (pass/fail) when run with pytest.

# Used by: N/A (these are unit tests for the EventTokenizer, 
# indirectly used by the tokenization script and
# any downstream models that consume tokenized data)


"""Unit tests for EventTokenizer."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.parsing import EventTokenizer


@pytest.fixture()
def tok_from_dict():
    """Build a tokenizer without CSV by directly populating internals."""
    tok = EventTokenizer()
    # Simulate three templates: tid 1, 2, 3
    tok._tid_to_text = {1: "template A", 2: "template B", 3: "template C"}
    tok._text_to_tid = {v: k for k, v in tok._tid_to_text.items()}
    tok._sorted_tids = [1, 2, 3]
    return tok


class TestConstants:
    def test_pad_id(self):
        assert EventTokenizer.PAD_ID == 0

    def test_unk_id(self):
        assert EventTokenizer.UNK_ID == 1

    def test_offset(self):
        assert EventTokenizer._OFFSET == 2


class TestEncode:
    def test_known_template(self, tok_from_dict):
        # template_id=1 -> token_id = 1+2 = 3
        assert tok_from_dict.encode([1]) == [3]

    def test_multiple(self, tok_from_dict):
        assert tok_from_dict.encode([1, 2, 3]) == [3, 4, 5]

    def test_unknown_maps_to_unk(self, tok_from_dict):
        assert tok_from_dict.encode([999]) == [EventTokenizer.UNK_ID]

    def test_empty(self, tok_from_dict):
        assert tok_from_dict.encode([]) == []


class TestDecode:
    def test_known_token(self, tok_from_dict):
        # token 3 -> tid 1 -> "template A"
        assert tok_from_dict.decode([3]) == ["template A"]

    def test_pad(self, tok_from_dict):
        assert tok_from_dict.decode([0]) == ["<PAD>"]

    def test_unk(self, tok_from_dict):
        assert tok_from_dict.decode([1]) == ["<UNK>"]

    def test_unknown_token_shows_unk(self, tok_from_dict):
        # token 999 not in vocab
        assert tok_from_dict.decode([999]) == ["<UNK>"]

    def test_roundtrip(self, tok_from_dict):
        original = [1, 2, 3]
        tokens = tok_from_dict.encode(original)
        texts = tok_from_dict.decode(tokens)
        assert texts == ["template A", "template B", "template C"]


class TestVocabSize:
    def test_vocab_size(self, tok_from_dict):
        # 3 templates + 2 (PAD, UNK) = 5
        assert tok_from_dict.vocab_size == 5

    def test_empty_vocab_size(self):
        tok = EventTokenizer()
        assert tok.vocab_size == 2   # only PAD + UNK offset


class TestVocabDict:
    def test_contains_pad_unk(self, tok_from_dict):
        d = tok_from_dict.to_vocab_dict()
        assert d["0"] == "<PAD>"
        assert d["1"] == "<UNK>"

    def test_contains_all_templates(self, tok_from_dict):
        d = tok_from_dict.to_vocab_dict()
        # tid 1 -> key "3", tid 2 -> key "4", tid 3 -> key "5"
        assert d["3"] == "template A"
        assert d["4"] == "template B"
        assert d["5"] == "template C"
