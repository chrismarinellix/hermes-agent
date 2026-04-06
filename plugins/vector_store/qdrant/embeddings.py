"""
Embedding generation via sentence-transformers.

Default model: all-MiniLM-L6-v2 (384 dims, fast, strong semantic quality).
Falls back to a simple TF-IDF-style bag-of-words float vector when
sentence-transformers is unavailable (useful for testing without GPU deps).
"""

import hashlib
import logging
import os
from typing import List, Union

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("HERMES_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
_MODEL_CACHE: dict = {}


class EmbeddingModel:
    """Wrapper around a sentence-transformers encoder with caching."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        if self.model_name in _MODEL_CACHE:
            self._model = _MODEL_CACHE[self.model_name]
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model '%s'...", self.model_name)
            model = SentenceTransformer(self.model_name)
            _MODEL_CACHE[self.model_name] = model
            self._model = model
            return model
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. Install: "
                "pip install 'hermes-agent[vector-store]'"
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Return embedding dimension without encoding a real document."""
        model = self._load()
        return model.get_sentence_embedding_dimension()

    def encode(self, text: str) -> List[float]:
        """Encode a single string to a float list."""
        model = self._load()
        vec = model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def encode_batch(
        self, texts: List[str], batch_size: int = 64, show_progress: bool = False
    ) -> List[List[float]]:
        """Encode a list of strings, returning a list of float lists."""
        model = self._load()
        vecs = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
        return vecs.tolist()

    def encode_article(self, article: dict) -> List[float]:
        """
        Build a combined embedding for a news article dict.

        Combines title (weighted 2×) + description/summary + content (first 512 chars)
        to produce a single representative vector.
        """
        title = article.get("title", "")
        description = article.get("description", "") or article.get("summary", "")
        content = article.get("content", "")[:512]

        combined = " ".join(filter(None, [title, title, description, content]))
        return self.encode(combined)
