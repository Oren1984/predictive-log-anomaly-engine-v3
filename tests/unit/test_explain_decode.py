# test/unit/test_explain_decode.py

# Purpose: Unit tests for the token_id -> template string decoding logic used in InferenceEngine.explain().

# Input: None (test code only)

# Output: Test results (pass/fail) when run with pytest.

# Used by: N/A (these are unit tests for the decoding logic, indirectly used by InferenceEngine.explain()).

"""Tests for token_id -> template string decoding via artifacts/vocab.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

VOCAB_PATH     = ROOT / "artifacts" / "vocab.json"
TEMPLATES_PATH = ROOT / "artifacts" / "templates.json"

_vocab_present     = VOCAB_PATH.exists()
_templates_present = TEMPLATES_PATH.exists()

needs_vocab     = pytest.mark.skipif(not _vocab_present,     reason="artifacts/vocab.json not found")
needs_templates = pytest.mark.skipif(not _templates_present, reason="artifacts/templates.json not found")


# ---------------------------------------------------------------------------
# vocab.json structure
# ---------------------------------------------------------------------------

class TestVocabJson:
    @needs_vocab
    def test_vocab_loads_as_dict(self):
        with open(VOCAB_PATH, encoding="utf-8") as fh:
            vocab = json.load(fh)
        assert isinstance(vocab, dict)
        assert len(vocab) > 0

    @needs_vocab
    def test_vocab_has_pad_token(self):
        with open(VOCAB_PATH, encoding="utf-8") as fh:
            vocab = json.load(fh)
        assert "0" in vocab
        assert vocab["0"] == "<PAD>"

    @needs_vocab
    def test_vocab_has_unk_token(self):
        with open(VOCAB_PATH, encoding="utf-8") as fh:
            vocab = json.load(fh)
        assert "1" in vocab
        assert vocab["1"] == "<UNK>"

    @needs_vocab
    def test_vocab_has_real_template_tokens(self):
        """Some token_ids >= 2 (real templates, offset from template_id) must exist."""
        with open(VOCAB_PATH, encoding="utf-8") as fh:
            vocab = json.load(fh)
        real_tokens = [k for k in vocab if int(k) >= 2]
        assert len(real_tokens) > 0
        # Each real token text must be non-empty
        for k in real_tokens[:5]:
            text = vocab[k]
            assert isinstance(text, str) and len(text) > 0

    @needs_vocab
    def test_vocab_keys_are_strings(self):
        with open(VOCAB_PATH, encoding="utf-8") as fh:
            vocab = json.load(fh)
        # All keys should be strings (JSON keys are always strings)
        assert all(isinstance(k, str) for k in vocab.keys())

    @needs_vocab
    def test_vocab_values_are_strings(self):
        with open(VOCAB_PATH, encoding="utf-8") as fh:
            vocab = json.load(fh)
        assert all(isinstance(v, str) for v in vocab.values())

    @needs_vocab
    def test_vocab_size_matches_templates_plus_2(self):
        with open(VOCAB_PATH, encoding="utf-8") as fh:
            vocab = json.load(fh)
        # vocab_size = num_templates + 2  (PAD + UNK)
        # Templates are entries with token_id >= 2
        n_templates = sum(1 for k in vocab if int(k) >= 2)
        assert len(vocab) == n_templates + 2


# ---------------------------------------------------------------------------
# templates.json structure
# ---------------------------------------------------------------------------

class TestTemplatesJson:
    @needs_templates
    def test_templates_loads_as_dict(self):
        with open(TEMPLATES_PATH, encoding="utf-8") as fh:
            templates = json.load(fh)
        assert isinstance(templates, dict)
        assert len(templates) > 0

    @needs_templates
    def test_templates_keys_are_string_ints(self):
        with open(TEMPLATES_PATH, encoding="utf-8") as fh:
            templates = json.load(fh)
        # All keys should be parseable as ints
        for k in templates:
            assert k.isdigit() or (k.startswith("-") and k[1:].isdigit()), \
                f"Non-integer key found: {k!r}"

    @needs_templates
    def test_templates_values_are_nonempty_strings(self):
        with open(TEMPLATES_PATH, encoding="utf-8") as fh:
            templates = json.load(fh)
        for v in templates.values():
            assert isinstance(v, str) and len(v) > 0


# ---------------------------------------------------------------------------
# Decode offset: token_id = template_id + 2
# ---------------------------------------------------------------------------

class TestDecodeOffset:
    @needs_vocab
    @needs_templates
    def test_vocab_and_templates_consistent(self):
        """For every token_id >= 2, vocab[str(tid)] should match templates[str(tid-2)]."""
        with open(VOCAB_PATH, encoding="utf-8") as fh:
            vocab = json.load(fh)
        with open(TEMPLATES_PATH, encoding="utf-8") as fh:
            templates = json.load(fh)

        # Check a sample of entries (up to 50)
        checked = 0
        for tok_str, text_v in vocab.items():
            tok_id = int(tok_str)
            if tok_id < 2:
                continue
            template_id = tok_id - 2
            tid_str = str(template_id)
            if tid_str in templates:
                assert vocab[tok_str] == templates[tid_str], (
                    f"Mismatch at token_id={tok_id}: "
                    f"vocab={vocab[tok_str]!r} vs templates={templates[tid_str]!r}"
                )
            checked += 1
            if checked >= 50:
                break

        assert checked > 0, "No template entries checked"

    @needs_vocab
    def test_decode_known_token(self):
        """token_id 0 -> PAD, 1 -> UNK, tokens >= 2 -> real templates."""
        with open(VOCAB_PATH, encoding="utf-8") as fh:
            vocab = json.load(fh)

        assert vocab.get("0") == "<PAD>"
        assert vocab.get("1") == "<UNK>"
        # Find the smallest real token_id (>= 2) and verify it decodes properly
        real_keys = sorted(int(k) for k in vocab if int(k) >= 2)
        assert len(real_keys) > 0
        first_real = str(real_keys[0])
        text = vocab[first_real]
        assert len(text) > 0
        assert text not in ("<PAD>", "<UNK>")


# ---------------------------------------------------------------------------
# InferenceEngine.explain decoding
# ---------------------------------------------------------------------------

class TestEngineExplainDecode:
    @pytest.mark.skipif(
        not (_vocab_present and (ROOT / "models" / "baseline.pkl").exists()),
        reason="Artifacts not found",
    )
    def test_explain_produces_template_preview(self):
        from src.runtime.inference_engine import InferenceEngine
        from src.sequencing.models import Sequence

        eng = InferenceEngine(mode="baseline", window_size=10)
        eng.load_artifacts()

        # Use a real token id that exists in the vocab
        with open(VOCAB_PATH, encoding="utf-8") as fh:
            vocab = json.load(fh)
        # Pick the first token_id >= 2
        first_real_tok = min(int(k) for k in vocab if int(k) >= 2)

        seq = Sequence(
            sequence_id="test_explain",
            tokens=[first_real_tok] * 10,
            timestamps=[float(i) for i in range(10)],
        )
        evidence = eng.explain(seq)

        assert "templates_preview" in evidence
        assert len(evidence["templates_preview"]) > 0
        # Each preview should contain the template ID prefix
        preview = evidence["templates_preview"][0]
        assert "tid=" in preview

    @pytest.mark.skipif(
        not _vocab_present,
        reason="vocab.json not found",
    )
    def test_explain_window_timestamps(self):
        from src.runtime.inference_engine import InferenceEngine
        from src.sequencing.models import Sequence

        eng = InferenceEngine(mode="baseline", window_size=10)
        eng.load_artifacts()

        seq = Sequence(
            sequence_id="ts_test",
            tokens=[5] * 10,
            timestamps=[float(i) * 2 for i in range(10)],
        )
        evidence = eng.explain(seq)
        assert evidence["window_start_ts"] == 0.0
        assert evidence["window_end_ts"] == 18.0

    @pytest.mark.skipif(not _vocab_present, reason="vocab.json not found")
    def test_explain_unknown_token_graceful(self):
        """Token IDs 0 (PAD) and 1 (UNK) should not crash explain()."""
        from src.runtime.inference_engine import InferenceEngine
        from src.sequencing.models import Sequence

        eng = InferenceEngine(mode="baseline", window_size=5)
        eng.load_artifacts()

        # tokens with PAD (0) and UNK (1) → template_ids = [] (both < 2)
        seq = Sequence(sequence_id="pad_test", tokens=[0, 1, 0, 1, 0])
        evidence = eng.explain(seq)
        # template_ids should be empty since all tokens < 2
        assert evidence["template_ids"] == []
        assert evidence["templates_preview"] == []
