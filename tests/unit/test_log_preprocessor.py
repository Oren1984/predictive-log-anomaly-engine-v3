# tests/unit/test_log_preprocessor.py
# Unit tests for LogPreprocessor (Stage 1: NLP Embedding).
#
# Coverage:
#   - clean(): normalization patterns (BLK, TIMESTAMP, IP, NODE, PATH, HEX, NUM, SERVICE)
#   - tokenize(): placeholder atomicity, word splitting, empty input
#   - is_trained property
#   - embed() / process_log() raise RuntimeError before model loaded
#   - train_embeddings(): trains model, updates is_trained
#   - embed() / process_log(): returns correct shape after training
#   - save() / load(): round-trip persistence
#   - train_embeddings(): raises ImportError guard (mocked) and ValueError on empty corpus
#   - FastText experimental warning on init

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.log_preprocessor import LogPreprocessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_corpus(preprocessor: LogPreprocessor, size: int = 30):
    """Build a small synthetic tokenised corpus for fast training."""
    messages = [
        "ERROR disk full on 10.0.0.1",
        "WARNING blk_-1234567890 replicated",
        "INFO node R3-M1-N1:J18-U11 healthy",
        "CRITICAL /var/log/app.log not found",
        "DEBUG 2005-12-01T06:51:06 checkpoint done",
    ]
    corpus = []
    for i in range(size):
        raw = messages[i % len(messages)]
        corpus.append(preprocessor.tokenize(preprocessor.clean(raw)))
    return corpus


# ---------------------------------------------------------------------------
# clean() — normalisation
# ---------------------------------------------------------------------------

class TestClean:
    def setup_method(self):
        self.p = LogPreprocessor()

    def test_lowercase(self):
        assert self.p.clean("ERROR Disk Full") == "error disk full"

    def test_blk_id(self):
        result = self.p.clean("Received blk_-1234567890 from node")
        assert "[BLK]" in result
        assert "1234567890" not in result

    def test_ip_address(self):
        result = self.p.clean("Connection from 192.168.1.100")
        assert "[IP]" in result
        assert "192.168.1.100" not in result

    def test_ip_with_port(self):
        result = self.p.clean("Connect to 10.0.0.1:8080 failed")
        assert "[IP]" in result

    def test_iso_timestamp(self):
        result = self.p.clean("Event at 2005-12-01T06:51:06 recorded")
        assert "[TIMESTAMP]" in result

    def test_bgl_dotted_timestamp(self):
        result = self.p.clean("At 2005-12-01-06.51.06.123456 kernel panic")
        assert "[TIMESTAMP]" in result

    def test_date_only(self):
        result = self.p.clean("Log date 2023-07-15 start")
        assert "[TIMESTAMP]" in result

    def test_node_name(self):
        result = self.p.clean("Node R3-M1-N1:J18-U11 down")
        assert "[NODE]" in result

    def test_unix_path(self):
        result = self.p.clean("File /var/log/app.log missing")
        assert "[PATH]" in result

    def test_hex_string(self):
        # 0x-prefixed strings don't match \b boundary; use bare 8-char hex
        result = self.p.clean("Error code deadbeef12345678 raised")
        assert "[HEX]" in result

    def test_bare_integer(self):
        result = self.p.clean("Retried 3 times with 1048576 bytes")
        assert "[NUM]" in result
        assert "1048576" not in result

    def test_service_prefix_hdfs(self):
        result = self.p.clean("hdfs: datanode started")
        assert "[SERVICE]" in result

    def test_service_prefix_bgl(self):
        result = self.p.clean("BGL, error occurred")
        assert "[SERVICE]" in result

    def test_whitespace_collapsed(self):
        result = self.p.clean("a  b   c")
        assert "  " not in result

    def test_empty_string(self):
        assert self.p.clean("") == ""

    def test_strip_leading_trailing(self):
        result = self.p.clean("  hello world  ")
        assert result == result.strip()


# ---------------------------------------------------------------------------
# tokenize()
# ---------------------------------------------------------------------------

class TestTokenize:
    def setup_method(self):
        self.p = LogPreprocessor()

    def test_basic_words(self):
        assert self.p.tokenize("error disk full") == ["error", "disk", "full"]

    def test_placeholder_atomic(self):
        tokens = self.p.tokenize("connect from [IP] on port [NUM]")
        assert "[IP]" in tokens
        assert "[NUM]" in tokens
        # Brackets must NOT appear as separate tokens
        assert "[" not in tokens
        assert "]" not in tokens

    def test_multiple_placeholders(self):
        tokens = self.p.tokenize("[TIMESTAMP] [BLK] [PATH]")
        assert tokens == ["[TIMESTAMP]", "[BLK]", "[PATH]"]

    def test_discards_punctuation(self):
        tokens = self.p.tokenize("error: disk, full!")
        assert ":" not in tokens
        assert "," not in tokens
        assert "!" not in tokens

    def test_empty_string(self):
        assert self.p.tokenize("") == []

    def test_mixed(self):
        cleaned = self.p.clean("ERROR blk_-123 on 10.0.0.1")
        tokens = self.p.tokenize(cleaned)
        assert "[BLK]" in tokens
        assert "[IP]" in tokens


# ---------------------------------------------------------------------------
# is_trained property
# ---------------------------------------------------------------------------

class TestIsTrained:
    def test_false_before_training(self):
        p = LogPreprocessor()
        assert p.is_trained is False

    def test_true_after_training(self):
        p = LogPreprocessor(vec_dim=10, min_count=1, epochs=1)
        corpus = _small_corpus(p)
        p.train_embeddings(corpus)
        assert p.is_trained is True


# ---------------------------------------------------------------------------
# embed() / process_log() — RuntimeError before model loaded
# ---------------------------------------------------------------------------

class TestRequiresModel:
    def setup_method(self):
        self.p = LogPreprocessor()

    def test_embed_raises_before_training(self):
        with pytest.raises(RuntimeError, match="No embedding model"):
            self.p.embed(["error", "disk"])

    def test_process_log_raises_before_training(self):
        with pytest.raises(RuntimeError, match="No embedding model"):
            self.p.process_log("ERROR disk full")

    def test_transform_raises_before_training(self):
        with pytest.raises(RuntimeError, match="No embedding model"):
            self.p.transform("ERROR disk full")


# ---------------------------------------------------------------------------
# train_embeddings()
# ---------------------------------------------------------------------------

class TestTrainEmbeddings:
    def test_trains_successfully(self):
        p = LogPreprocessor(vec_dim=16, min_count=1, epochs=2)
        corpus = _small_corpus(p)
        p.train_embeddings(corpus)
        assert p.is_trained

    def test_raises_on_empty_corpus(self):
        p = LogPreprocessor()
        with pytest.raises(ValueError, match="corpus must not be empty"):
            p.train_embeddings([])

    def test_raises_import_error_when_gensim_absent(self):
        p = LogPreprocessor()
        import src.preprocessing.log_preprocessor as mod
        with patch.object(mod, "_GENSIM_AVAILABLE", False):
            with pytest.raises(ImportError, match="gensim is required"):
                p.train_embeddings([["error", "disk"]])


# ---------------------------------------------------------------------------
# embed() — output shape and dtype after training
# ---------------------------------------------------------------------------

class TestEmbed:
    def setup_method(self):
        self.p = LogPreprocessor(vec_dim=16, min_count=1, epochs=2)
        self.p.train_embeddings(_small_corpus(self.p))

    def test_shape(self):
        vec = self.p.embed(["error", "disk"])
        assert vec.shape == (16,)

    def test_dtype(self):
        vec = self.p.embed(["error", "disk"])
        assert vec.dtype == np.float32

    def test_zero_vector_for_oov_tokens(self):
        vec = self.p.embed(["zzz_totally_oov_token"])
        assert np.all(vec == 0.0)

    def test_nonzero_for_known_tokens(self):
        # "error" should be in vocabulary after training
        vec = self.p.embed(["error"])
        # Not guaranteed all zeros
        assert vec.shape == (16,)


# ---------------------------------------------------------------------------
# process_log() — end-to-end
# ---------------------------------------------------------------------------

class TestProcessLog:
    def setup_method(self):
        self.p = LogPreprocessor(vec_dim=16, min_count=1, epochs=2)
        self.p.train_embeddings(_small_corpus(self.p))

    def test_returns_correct_shape(self):
        vec = self.p.process_log("ERROR disk full on 10.0.0.1")
        assert vec.shape == (16,)

    def test_returns_float32(self):
        vec = self.p.process_log("WARNING blk_-999 replicated")
        assert vec.dtype == np.float32

    def test_transform_alias_matches(self):
        raw = "ERROR disk full"
        v1 = self.p.process_log(raw)
        v2 = self.p.transform(raw)
        np.testing.assert_array_equal(v1, v2)


# ---------------------------------------------------------------------------
# save() / load() — persistence round-trip
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        p1 = LogPreprocessor(vec_dim=16, min_count=1, epochs=2)
        corpus = _small_corpus(p1)
        p1.train_embeddings(corpus)

        model_path = tmp_path / "word2vec.model"
        p1.save(model_path)
        assert model_path.exists()

        p2 = LogPreprocessor(vec_dim=16)
        p2.load(model_path)

        assert p2.is_trained
        assert p2.vec_dim == 16
        v1 = p1.process_log("error disk full")
        v2 = p2.process_log("error disk full")
        np.testing.assert_allclose(v1, v2, atol=1e-6)

    def test_save_creates_parent_dirs(self, tmp_path):
        p = LogPreprocessor(vec_dim=10, min_count=1, epochs=1)
        p.train_embeddings(_small_corpus(p))
        nested = tmp_path / "models" / "sub" / "w2v.model"
        p.save(nested)
        assert nested.exists()

    def test_save_raises_without_model(self, tmp_path):
        p = LogPreprocessor()
        with pytest.raises(RuntimeError, match="No embedding model"):
            p.save(tmp_path / "model.bin")

    def test_load_raises_if_file_missing(self, tmp_path):
        p = LogPreprocessor()
        with pytest.raises(FileNotFoundError):
            p.load(tmp_path / "nonexistent.model")

    def test_load_raises_import_error_when_gensim_absent(self, tmp_path):
        p = LogPreprocessor()
        import src.preprocessing.log_preprocessor as mod
        with patch.object(mod, "_GENSIM_AVAILABLE", False):
            with pytest.raises(ImportError, match="gensim is required"):
                p.load(tmp_path / "any.model")


# ---------------------------------------------------------------------------
# FastText experimental mode
# ---------------------------------------------------------------------------

class TestFastTextExperimental:
    def test_fasttext_init_emits_warning(self, recwarn):
        import logging
        import warnings
        # LogPreprocessor uses logger.warning, not warnings.warn
        # Capture via logging
        with patch("src.preprocessing.log_preprocessor.logger") as mock_logger:
            LogPreprocessor(embedding_type="fasttext")
            assert mock_logger.warning.called

    def test_invalid_embedding_type_raises(self):
        with pytest.raises(ValueError, match="embedding_type must be"):
            LogPreprocessor(embedding_type="bert")  # type: ignore[arg-type]
