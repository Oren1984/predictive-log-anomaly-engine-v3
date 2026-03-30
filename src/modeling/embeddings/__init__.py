# src/modeling/embeddings/__init__.py
# v2 embedding sub-package — re-exports the public API of word2vec_trainer.
from .word2vec_trainer import Word2VecTrainer, build_corpus_from_messages

__all__ = ["Word2VecTrainer", "build_corpus_from_messages"]
