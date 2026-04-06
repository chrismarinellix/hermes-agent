"""
Qdrant client manager.

Handles connection lifecycle, collection creation, and health checks.
Set QDRANT_URL (default: http://localhost:6333) and optionally QDRANT_API_KEY.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Collection config — one collection per article corpus
AI_NEWS_COLLECTION = "ai_news"
EMBEDDING_DIM = 384        # all-MiniLM-L6-v2
DISTANCE_METRIC = "Cosine"


class QdrantClientManager:
    """Thin wrapper around the Qdrant Python client."""

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection: str = AI_NEWS_COLLECTION,
        embedding_dim: int = EMBEDDING_DIM,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError as exc:
            raise ImportError(
                "qdrant-client is required. Install it with: "
                "pip install 'hermes-agent[vector-store]'"
            ) from exc

        self._QdrantClient = QdrantClient
        self._Distance = Distance
        self._VectorParams = VectorParams

        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY") or None
        self.collection = collection
        self.embedding_dim = embedding_dim
        self._client: Optional[QdrantClient] = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> "QdrantClientManager":
        """Open connection and ensure collection exists."""
        kwargs = {"url": self.url, "prefer_grpc": False}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        self._client = self._QdrantClient(**kwargs)
        self._ensure_collection()
        logger.info("Connected to Qdrant at %s (collection: %s)", self.url, self.collection)
        return self

    def disconnect(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "QdrantClientManager":
        return self.connect()

    def __exit__(self, *_) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    @property
    def client(self):
        if self._client is None:
            raise RuntimeError("Not connected. Call connect() first or use as context manager.")
        return self._client

    def _ensure_collection(self) -> None:
        from qdrant_client.models import VectorParams, Distance

        existing = {c.name for c in self._client.get_collections().collections}
        if self.collection not in existing:
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            # Payload indexes for fast filtered search
            from qdrant_client.models import PayloadSchemaType
            for field in ("source", "tags", "published_at"):
                self._client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            logger.info("Created collection '%s'", self.collection)

    def collection_info(self) -> dict:
        info = self.client.get_collection(self.collection)
        return {
            "name": self.collection,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": str(info.status),
        }

    def delete_collection(self) -> None:
        self.client.delete_collection(self.collection)
        logger.warning("Deleted collection '%s'", self.collection)

    def health(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False
