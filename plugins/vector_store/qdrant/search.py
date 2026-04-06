"""
Semantic and hybrid search over the AI news Qdrant collection.

Usage:
    with QdrantClientManager() as mgr:
        searcher = NewsSearcher(mgr)
        results = searcher.search("transformer architecture breakthroughs", limit=5)
"""

import logging
from typing import Any, Dict, List, Optional

from .client import QdrantClientManager
from .embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


def _result_to_dict(hit) -> dict:
    """Convert a ScoredPoint into a clean dict."""
    payload = hit.payload or {}
    return {
        "score": round(hit.score, 4),
        "id": str(hit.id),
        "title": payload.get("title", ""),
        "url": payload.get("url", ""),
        "source": payload.get("source", ""),
        "published_at": payload.get("published_at", ""),
        "description": payload.get("description", ""),
        "content_snippet": payload.get("content_snippet", ""),
        "tags": payload.get("tags", []),
        "author": payload.get("author", ""),
    }


class NewsSearcher:
    """Semantic search over indexed news articles."""

    def __init__(
        self,
        client_manager: QdrantClientManager,
        embedding_model: Optional[EmbeddingModel] = None,
    ) -> None:
        self.mgr = client_manager
        self.emb = embedding_model or EmbeddingModel()

    # ------------------------------------------------------------------
    # Core search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.0,
        source_filter: Optional[List[str]] = None,
        tags_filter: Optional[List[str]] = None,
        since: Optional[str] = None,       # ISO date string lower bound
        until: Optional[str] = None,       # ISO date string upper bound
    ) -> List[Dict[str, Any]]:
        """
        Semantic search with optional payload filters.

        Args:
            query:            Natural-language search string.
            limit:            Max results to return.
            score_threshold:  Minimum cosine similarity (0–1).
            source_filter:    Only include articles from these sources.
            tags_filter:      Only include articles with at least one of these tags.
            since / until:    ISO-8601 date range for published_at.

        Returns:
            List of result dicts sorted by descending score.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchAny, Range

        query_vector = self.emb.encode(query)

        # Build optional filter
        must_conditions = []

        if source_filter:
            must_conditions.append(
                FieldCondition(key="source", match=MatchAny(any=source_filter))
            )
        if tags_filter:
            must_conditions.append(
                FieldCondition(key="tags", match=MatchAny(any=tags_filter))
            )
        if since or until:
            range_kwargs = {}
            if since:
                range_kwargs["gte"] = since
            if until:
                range_kwargs["lte"] = until
            must_conditions.append(
                FieldCondition(key="published_at", range=Range(**range_kwargs))
            )

        search_filter = Filter(must=must_conditions) if must_conditions else None

        hits = self.mgr.client.search(
            collection_name=self.mgr.collection,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold if score_threshold > 0 else None,
            query_filter=search_filter,
            with_payload=True,
        )

        results = [_result_to_dict(h) for h in hits]
        logger.debug("Search '%s' → %d results", query[:60], len(results))
        return results

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def search_by_source(self, query: str, source: str, limit: int = 10) -> List[dict]:
        return self.search(query, limit=limit, source_filter=[source])

    def search_by_tag(self, query: str, tag: str, limit: int = 10) -> List[dict]:
        return self.search(query, limit=limit, tags_filter=[tag])

    def find_similar(self, url_or_id: str, limit: int = 5) -> List[dict]:
        """Find articles similar to an already-indexed article by its URL."""
        import uuid
        from qdrant_client.models import RecommendStrategy

        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, url_or_id))
        hits = self.mgr.client.recommend(
            collection_name=self.mgr.collection,
            positive=[point_id],
            limit=limit,
            with_payload=True,
        )
        return [_result_to_dict(h) for h in hits]

    def scroll_all(self, limit: int = 100, offset: Optional[str] = None) -> tuple:
        """
        Page through all articles.  Returns (results, next_offset).
        Pass next_offset into the next call to continue pagination.
        """
        hits, next_page = self.mgr.client.scroll(
            collection_name=self.mgr.collection,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        return [_result_to_dict(h) for h in hits], next_page
