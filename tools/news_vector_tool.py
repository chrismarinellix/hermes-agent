#!/usr/bin/env python3
"""
News Vector Tool

Integrates the Hermes web scraping stack with Qdrant vector storage.
Provides two agent-facing operations:
  - index_news:  scrape + embed + store articles from a query or URL list
  - search_news: semantic search over the indexed article corpus

Environment variables:
  QDRANT_URL       Qdrant server URL   (default: http://localhost:6333)
  QDRANT_API_KEY   Optional API key
  HERMES_EMBEDDING_MODEL  sentence-transformers model (default: all-MiniLM-L6-v2)

Dependencies (install via pip install 'hermes-agent[vector-store]'):
  qdrant-client>=1.9.0
  sentence-transformers>=3.0.0
  numpy>=1.24.0
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports — only required at call time so the rest of the agent starts
# fine even without the optional deps installed.
# ---------------------------------------------------------------------------

def _get_plugin():
    """Return connected (QdrantClientManager, EmbeddingModel)."""
    try:
        from plugins.vector_store.qdrant.client import QdrantClientManager
        from plugins.vector_store.qdrant.embeddings import EmbeddingModel
    except ImportError as exc:
        raise ImportError(
            "Vector store plugin not found. Ensure plugins/vector_store/qdrant/ exists."
        ) from exc
    mgr = QdrantClientManager()
    mgr.connect()
    return mgr, EmbeddingModel()


# ---------------------------------------------------------------------------
# Article extraction helpers
# ---------------------------------------------------------------------------

def _extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return "unknown"


def _web_search_articles(query: str, limit: int = 10) -> List[dict]:
    """Use the Hermes web_tools search backend to find news articles."""
    try:
        from tools.web_tools import web_search_tool
        raw = web_search_tool(query, limit=limit)
        # web_search_tool returns a JSON string or list
        if isinstance(raw, str):
            raw = json.loads(raw)
        if not isinstance(raw, list):
            raw = [raw]
        return raw
    except Exception as exc:
        logger.warning("web_search_tool failed: %s", exc)
        return []


def _web_extract_article(url: str) -> Optional[dict]:
    """Fetch and extract text content from a single URL."""
    try:
        from tools.web_tools import web_extract_tool
        raw = web_extract_tool([url], format="markdown")
        if isinstance(raw, str):
            data = json.loads(raw)
        else:
            data = raw
        if isinstance(data, list):
            data = data[0] if data else {}
        return data
    except Exception as exc:
        logger.warning("web_extract_tool failed for %s: %s", url, exc)
        return None


def _search_result_to_article(result: dict) -> dict:
    """Normalise a web_search_tool result to the article schema."""
    url = result.get("url", result.get("link", ""))
    return {
        "title": result.get("title", ""),
        "url": url,
        "source": _extract_domain(url),
        "published_at": result.get("published_date") or result.get("date") or datetime.now(timezone.utc).isoformat(),
        "description": result.get("description", result.get("snippet", result.get("text", ""))),
        "content": result.get("content", result.get("markdown", "")),
        "tags": result.get("tags", []),
        "author": result.get("author", ""),
    }


# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------

def index_news(
    query: Optional[str] = None,
    urls: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    limit: int = 20,
    fetch_full_content: bool = False,
) -> str:
    """
    Scrape AI news articles and index them into Qdrant.

    Args:
        query:              Search query (e.g. "large language model breakthroughs").
                            Used with the Hermes web search backend.
        urls:               Explicit list of article URLs to index.
        tags:               Extra tags to attach to every article in this batch.
        limit:              Max articles to index when using query mode.
        fetch_full_content: If True, fetch full page content for each article
                            (slower but richer embeddings).

    Returns:
        JSON string with indexing summary.
    """
    from plugins.vector_store.qdrant.indexer import NewsIndexer

    if not query and not urls:
        return json.dumps({"error": "Provide either query or urls."})

    mgr, emb = _get_plugin()
    indexer = NewsIndexer(mgr, emb)

    articles: List[dict] = []

    try:
        # --- Query-based discovery ---
        if query:
            raw_results = _web_search_articles(query, limit=limit)
            for r in raw_results:
                article = _search_result_to_article(r)
                if fetch_full_content and article["url"]:
                    extracted = _web_extract_article(article["url"])
                    if extracted:
                        article["content"] = extracted.get("markdown", extracted.get("content", article["content"]))
                if tags:
                    article["tags"] = list(set(article.get("tags", []) + tags))
                articles.append(article)

        # --- URL list mode ---
        if urls:
            for url in urls:
                extracted = _web_extract_article(url)
                if extracted:
                    article = {
                        "title": extracted.get("title", extracted.get("metadata", {}).get("title", "")),
                        "url": url,
                        "source": _extract_domain(url),
                        "published_at": extracted.get("metadata", {}).get("publishedTime", datetime.now(timezone.utc).isoformat()),
                        "description": extracted.get("description", ""),
                        "content": extracted.get("markdown", extracted.get("content", "")),
                        "tags": tags or [],
                        "author": extracted.get("metadata", {}).get("author", ""),
                    }
                    articles.append(article)

        if not articles:
            return json.dumps({"status": "no_articles", "indexed": 0})

        point_ids = indexer.index_batch(articles, show_progress=False)
        info = mgr.collection_info()

        return json.dumps({
            "status": "ok",
            "indexed": len(point_ids),
            "collection": info,
        }, default=str)

    finally:
        mgr.disconnect()


def search_news(
    query: str,
    limit: int = 10,
    score_threshold: float = 0.3,
    source_filter: Optional[List[str]] = None,
    tags_filter: Optional[List[str]] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
) -> str:
    """
    Semantic search over indexed AI news articles.

    Args:
        query:            Natural language query.
        limit:            Max results.
        score_threshold:  Min cosine similarity to include (0–1).
        source_filter:    Filter to specific publisher domains.
        tags_filter:      Filter to articles with these tags.
        since / until:    ISO-8601 date strings to bound published_at.

    Returns:
        JSON string with list of matching articles and their scores.
    """
    from plugins.vector_store.qdrant.search import NewsSearcher

    mgr, emb = _get_plugin()
    searcher = NewsSearcher(mgr, emb)

    try:
        results = searcher.search(
            query=query,
            limit=limit,
            score_threshold=score_threshold,
            source_filter=source_filter,
            tags_filter=tags_filter,
            since=since,
            until=until,
        )
        return json.dumps({"status": "ok", "count": len(results), "results": results}, default=str)
    finally:
        mgr.disconnect()


def collection_stats() -> str:
    """Return current Qdrant collection statistics."""
    mgr, _ = _get_plugin()
    try:
        info = mgr.collection_info()
        return json.dumps({"status": "ok", "collection": info})
    finally:
        mgr.disconnect()


# ---------------------------------------------------------------------------
# Tool schema definitions (for Hermes tool registration)
# ---------------------------------------------------------------------------

NEWS_VECTOR_TOOLS = [
    {
        "name": "index_news",
        "description": (
            "Scrape AI news articles via web search or explicit URLs and index them "
            "into the Qdrant vector database for later semantic search. "
            "Use this to build or refresh the article corpus."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Web search query to discover articles (e.g. 'GPT-5 capabilities 2025').",
                },
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Explicit list of article URLs to index.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Topic tags to attach to all articles in this batch.",
                },
                "limit": {
                    "type": "integer",
                    "default": 20,
                    "description": "Max articles to index (query mode only).",
                },
                "fetch_full_content": {
                    "type": "boolean",
                    "default": False,
                    "description": "Fetch full page content for richer embeddings (slower).",
                },
            },
        },
        "function": index_news,
    },
    {
        "name": "search_news",
        "description": (
            "Semantic search over indexed AI news articles. "
            "Returns articles most relevant to the query ranked by cosine similarity. "
            "Supports filtering by source, tags, and date range."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query.",
                },
                "limit": {
                    "type": "integer",
                    "default": 10,
                    "description": "Maximum number of results to return.",
                },
                "score_threshold": {
                    "type": "number",
                    "default": 0.3,
                    "description": "Minimum cosine similarity score (0–1).",
                },
                "source_filter": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Restrict to these publisher domains.",
                },
                "tags_filter": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Restrict to articles with at least one of these tags.",
                },
                "since": {
                    "type": "string",
                    "description": "ISO-8601 date lower bound for published_at (e.g. '2025-01-01').",
                },
                "until": {
                    "type": "string",
                    "description": "ISO-8601 date upper bound for published_at.",
                },
            },
            "required": ["query"],
        },
        "function": search_news,
    },
    {
        "name": "news_collection_stats",
        "description": "Return statistics about the indexed AI news Qdrant collection.",
        "input_schema": {"type": "object", "properties": {}},
        "function": lambda **_: collection_stats(),
    },
]


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python news_vector_tool.py search 'your query'")
        print("       python news_vector_tool.py index  'search topic'")
        print("       python news_vector_tool.py stats")
        sys.exit(1)

    cmd = sys.argv[1]
    arg = sys.argv[2] if len(sys.argv) > 2 else ""

    if cmd == "search":
        print(search_news(arg, limit=5))
    elif cmd == "index":
        print(index_news(query=arg, limit=5))
    elif cmd == "stats":
        print(collection_stats())
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
