"""
Interactions indexer — stores every conversation turn into Qdrant.

Collections:
    hermes_interactions   — Telegram, CLI, API, and other platform exchanges
    email_archive         — managed separately by email_tool.py

Schema per point:
    platform        str   "telegram" | "email" | "claude_code" | "cron" | ...
    conversation_id str   session key or thread identifier
    role            str   "user" | "assistant"
    content         str   message text (embedded for semantic search)
    timestamp       str   ISO-8601
    user_id         str   platform user ID (optional)
    user_name       str   display name (optional)
    chat_id         str   platform chat/channel ID (optional)
    chat_name       str   channel/group name (optional)
    tags            list  extra labels (e.g. ["cron", "morning-briefing"])
"""

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

INTERACTIONS_COLLECTION = "hermes_interactions"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2


def _point_id(platform: str, conversation_id: str, role: str, timestamp: str, content: str) -> str:
    key = f"{platform}:{conversation_id}:{role}:{timestamp}:{content[:64]}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


class InteractionsIndexer:
    """Upserts conversation turns into the hermes_interactions Qdrant collection."""

    def __init__(self, client_manager, embedding_model) -> None:
        self.mgr = client_manager
        self.emb = embedding_model

    def _ensure_collection(self) -> None:
        from qdrant_client.models import VectorParams, Distance, PayloadSchemaType
        existing = {c.name for c in self.mgr.client.get_collections().collections}
        if INTERACTIONS_COLLECTION not in existing:
            self.mgr.client.create_collection(
                collection_name=INTERACTIONS_COLLECTION,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            for field in ("platform", "conversation_id", "role", "timestamp", "user_id", "chat_id", "tags"):
                self.mgr.client.create_payload_index(
                    collection_name=INTERACTIONS_COLLECTION,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            logger.info("Created collection '%s'", INTERACTIONS_COLLECTION)

    def index_turn(
        self,
        platform: str,
        conversation_id: str,
        role: str,
        content: str,
        timestamp: Optional[str] = None,
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
        chat_id: Optional[str] = None,
        chat_name: Optional[str] = None,
        tags: Optional[list] = None,
    ) -> str:
        """Index a single conversation turn. Returns the point ID."""
        from qdrant_client.models import PointStruct

        if not content or not content.strip():
            return ""

        ts = timestamp or datetime.now(timezone.utc).isoformat()
        point_id = _point_id(platform, conversation_id, role, ts, content)

        vector = self.emb.encode(content[:2000])  # cap to avoid huge embeddings

        payload = {
            "platform": platform,
            "conversation_id": conversation_id,
            "role": role,
            "content": content[:4000],
            "timestamp": ts,
            "user_id": user_id or "",
            "user_name": user_name or "",
            "chat_id": chat_id or "",
            "chat_name": chat_name or "",
            "tags": tags or [],
        }

        self._ensure_collection()
        self.mgr.client.upsert(
            collection_name=INTERACTIONS_COLLECTION,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )
        return point_id

    def index_exchange(
        self,
        platform: str,
        conversation_id: str,
        user_content: str,
        assistant_content: str,
        timestamp: Optional[str] = None,
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
        chat_id: Optional[str] = None,
        chat_name: Optional[str] = None,
        tags: Optional[list] = None,
    ) -> None:
        """Index a full user↔assistant exchange as two points."""
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        common = dict(
            platform=platform,
            conversation_id=conversation_id,
            timestamp=ts,
            user_id=user_id,
            user_name=user_name,
            chat_id=chat_id,
            chat_name=chat_name,
            tags=tags,
        )
        if user_content:
            self.index_turn(role="user", content=user_content, **common)
        if assistant_content:
            self.index_turn(role="assistant", content=assistant_content, **common)
