"""
News article indexer — upserts articles into Qdrant.

Article schema (dict):
    id          str   unique article identifier (URL or hash)
    title       str   headline
    url         str   canonical URL
    source      str   publisher domain / feed name
    published_at str  ISO-8601 datetime string
    description str   short summary / lede (optional)
    content     str   full body text (optional)
    tags        list[str]  topic tags (optional)
    author      str   (optional)
"""

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from .client import QdrantClientManager
from .embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


def _article_id(article: dict) -> str:
    """Stable UUID5 from article URL (or id field if present)."""
    key = article.get("id") or article.get("url", "")
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def _normalise(article: dict) -> dict:
    """Return a cleaned copy with required fields guaranteed."""
    out = dict(article)
    out.setdefault("title", "")
    out.setdefault("url", "")
    out.setdefault("source", "unknown")
    out.setdefault("description", "")
    out.setdefault("content", "")
    out.setdefault("tags", [])
    out.setdefault("author", "")
    # Normalise published_at to ISO string
    pub = out.get("published_at")
    if isinstance(pub, datetime):
        out["published_at"] = pub.isoformat()
    elif not pub:
        out["published_at"] = datetime.now(timezone.utc).isoformat()
    return out


class NewsIndexer:
    """Upserts news articles (with embeddings) into a Qdrant collection."""

    def __init__(
        self,
        client_manager: QdrantClientManager,
        embedding_model: Optional[EmbeddingModel] = None,
    ) -> None:
        self.mgr = client_manager
        self.emb = embedding_model or EmbeddingModel()

    # ------------------------------------------------------------------
    # Single article
    # ------------------------------------------------------------------

    def index_article(self, article: dict) -> str:
        """Upsert one article. Returns the point ID used."""
        from qdrant_client.models import PointStruct

        article = _normalise(article)
        point_id = _article_id(article)
        vector = self.emb.encode_article(article)

        payload = {k: v for k, v in article.items() if k != "content"}
        # Store a truncated content snippet for display
        payload["content_snippet"] = article["content"][:300]
        payload["indexed_at"] = datetime.now(timezone.utc).isoformat()

        self.mgr.client.upsert(
            collection_name=self.mgr.collection,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )
        logger.debug("Indexed article '%s' (id=%s)", article["title"][:60], point_id)
        return point_id

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def index_batch(
        self,
        articles: List[dict],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> List[str]:
        """
        Upsert a list of articles efficiently.

        Embeddings are generated in one batched forward pass; Qdrant upsert
        is called in chunks to stay within the server payload limit.
        """
        from qdrant_client.models import PointStruct

        if not articles:
            return []

        articles = [_normalise(a) for a in articles]
        point_ids = [_article_id(a) for a in articles]

        # Build combined texts for batch encoding
        texts = [
            " ".join(filter(None, [
                a["title"], a["title"],
                a.get("description", ""),
                a.get("content", "")[:512],
            ]))
            for a in articles
        ]

        logger.info("Encoding %d articles...", len(texts))
        vectors = self.emb.encode_batch(texts, batch_size=batch_size, show_progress=show_progress)

        now = datetime.now(timezone.utc).isoformat()
        points = []
        for article, pid, vec in zip(articles, point_ids, vectors):
            payload = {k: v for k, v in article.items() if k != "content"}
            payload["content_snippet"] = article["content"][:300]
            payload["indexed_at"] = now
            points.append(PointStruct(id=pid, vector=vec, payload=payload))

        # Upsert in chunks
        for i in range(0, len(points), batch_size):
            chunk = points[i : i + batch_size]
            self.mgr.client.upsert(
                collection_name=self.mgr.collection,
                points=chunk,
            )
            logger.debug("Upserted %d/%d points", min(i + batch_size, len(points)), len(points))

        logger.info("Indexed %d articles into '%s'", len(articles), self.mgr.collection)
        return point_ids

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def delete_article(self, url_or_id: str) -> None:
        from qdrant_client.models import PointIdsList
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, url_or_id))
        self.mgr.client.delete(
            collection_name=self.mgr.collection,
            points_selector=PointIdsList(points=[point_id]),
        )
        logger.debug("Deleted point %s", point_id)
