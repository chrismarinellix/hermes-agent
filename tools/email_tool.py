#!/usr/bin/env python3
"""
Email Tool — comprehensive email management for Hermes.

Provides:
  - fetch        Fetch recent emails with priority scoring (skip/low/medium/high)
  - send         Send an email directly
  - draft        Create a draft held for human-in-the-loop approval
  - send_draft   Approve and send a pending draft by ID
  - reject_draft Discard a pending draft
  - list_drafts  List all drafts (pending/sent/rejected)
  - search       Semantic search over indexed email history (Qdrant RAG)
  - index        Embed + store emails into Qdrant for future RAG queries
  - unsubscribe  Smart unsubscribe using List-Unsubscribe headers + browser automation
  - commitments  List extracted action items and deadlines
  - add_commitment  Record an action item / deadline from an email
  - complete     Mark a commitment as done by ID

Environment variables:
    EMAIL_IMAP_HOST, EMAIL_IMAP_PORT (default 993)
    EMAIL_SMTP_HOST, EMAIL_SMTP_PORT (default 587)
    EMAIL_ADDRESS, EMAIL_PASSWORD
    QDRANT_URL  (default: http://localhost:6333)
    QDRANT_API_KEY (optional)
    HERMES_EMBEDDING_MODEL (default: all-MiniLM-L6-v2)

Qdrant deps: pip install 'hermes-agent[vector-store]'
"""

import email as email_lib
import imaplib
import json
import logging
import os
import re
import smtplib
import sqlite3
import ssl
import uuid
from datetime import datetime, timezone
from email.header import decode_header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _get_email_db_path() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "email.db"


def _open_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_get_email_db_path()), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    _init_db(conn)
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS email_drafts (
            id          TEXT PRIMARY KEY,
            to_addr     TEXT NOT NULL,
            subject     TEXT NOT NULL,
            body        TEXT NOT NULL,
            in_reply_to TEXT,
            references  TEXT,
            created_at  TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'pending',
            sent_at     TEXT,
            error       TEXT
        );

        CREATE TABLE IF NOT EXISTS email_commitments (
            id           TEXT PRIMARY KEY,
            message_id   TEXT,
            sender       TEXT,
            subject      TEXT,
            action_text  TEXT NOT NULL,
            deadline     TEXT,
            created_at   TEXT NOT NULL,
            completed_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_drafts_status ON email_drafts (status);
        CREATE INDEX IF NOT EXISTS idx_commitments_completed ON email_commitments (completed_at);
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# IMAP helpers
# ---------------------------------------------------------------------------

def _decode_header_value(raw: str) -> str:
    parts = decode_header(raw or "")
    out = []
    for part, charset in parts:
        if isinstance(part, bytes):
            out.append(part.decode(charset or "utf-8", errors="replace"))
        else:
            out.append(str(part))
    return " ".join(out).strip()


def _extract_email_address(raw: str) -> str:
    m = re.search(r"<([^>]+)>", raw)
    return (m.group(1) if m else raw).strip().lower()


def _strip_html(html: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    text = re.sub(r"<p[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_body(msg: email_lib.message.Message) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            disp = str(part.get("Content-Disposition", ""))
            if "attachment" in disp:
                continue
            if ct == "text/plain":
                pl = part.get_payload(decode=True)
                if pl:
                    return pl.decode(part.get_content_charset() or "utf-8", errors="replace")
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                pl = part.get_payload(decode=True)
                if pl:
                    return _strip_html(pl.decode(part.get_content_charset() or "utf-8", errors="replace"))
        return ""
    pl = msg.get_payload(decode=True)
    if pl:
        text = pl.decode(msg.get_content_charset() or "utf-8", errors="replace")
        return _strip_html(text) if msg.get_content_type() == "text/html" else text
    return ""


def _get_imap() -> imaplib.IMAP4_SSL:
    host = os.environ.get("EMAIL_IMAP_HOST", "")
    port = int(os.environ.get("EMAIL_IMAP_PORT", "993"))
    addr = os.environ.get("EMAIL_ADDRESS", "")
    pwd = os.environ.get("EMAIL_PASSWORD", "")
    imap = imaplib.IMAP4_SSL(host, port, timeout=30)
    imap.login(addr, pwd)
    return imap


def _get_smtp() -> smtplib.SMTP:
    host = os.environ.get("EMAIL_SMTP_HOST", "")
    port = int(os.environ.get("EMAIL_SMTP_PORT", "587"))
    addr = os.environ.get("EMAIL_ADDRESS", "")
    pwd = os.environ.get("EMAIL_PASSWORD", "")
    smtp = smtplib.SMTP(host, port, timeout=30)
    smtp.starttls(context=ssl.create_default_context())
    smtp.login(addr, pwd)
    return smtp


# ---------------------------------------------------------------------------
# Priority scoring (heuristic, no LLM — fast for inbox processing)
# ---------------------------------------------------------------------------

_SKIP_PATTERNS = (
    "noreply", "no-reply", "no_reply", "donotreply", "do-not-reply",
    "mailer-daemon", "postmaster", "bounce", "notifications@",
    "automated@", "auto-confirm", "auto-reply",
)
_HIGH_SUBJECT_KEYWORDS = (
    "urgent", "action required", "asap", "deadline", "expires",
    "expiring", "final notice", "overdue", "time sensitive",
    "important:", "critical", "response needed",
)
_LOW_SUBJECT_KEYWORDS = (
    "newsletter", "unsubscribe", "promotion", "offer", "sale",
    "% off", "deal", "coupon", "digest", "weekly", "monthly",
    "roundup", "recap", "summary", "update:", "news:",
)


def _score_priority(sender: str, subject: str, headers: dict) -> str:
    """Return 'skip' | 'low' | 'medium' | 'high'."""
    sender_l = sender.lower()
    if any(p in sender_l for p in _SKIP_PATTERNS):
        return "skip"

    # RFC automated-mail headers
    if headers.get("Auto-Submitted", "no").lower() != "no":
        return "skip"
    prec = headers.get("Precedence", "").lower()
    if prec in ("bulk", "list", "junk"):
        return "low"
    if headers.get("List-Unsubscribe"):
        return "low"

    subject_l = subject.lower()
    if any(k in subject_l for k in _HIGH_SUBJECT_KEYWORDS):
        return "high"
    if any(k in subject_l for k in _LOW_SUBJECT_KEYWORDS):
        return "low"

    return "medium"


# ---------------------------------------------------------------------------
# Qdrant email archive
# ---------------------------------------------------------------------------

_EMAIL_COLLECTION = "email_archive"


def _get_email_plugin():
    """Return (QdrantClientManager, EmbeddingModel) for the email_archive collection."""
    try:
        from plugins.vector_store.qdrant.client import QdrantClientManager
        from plugins.vector_store.qdrant.embeddings import EmbeddingModel
    except ImportError as exc:
        raise ImportError(
            "Vector store plugin not found. "
            "Install with: pip install 'hermes-agent[vector-store]'"
        ) from exc
    mgr = QdrantClientManager(collection=_EMAIL_COLLECTION)
    mgr.connect()
    return mgr, EmbeddingModel()


def _email_point_id(message_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, message_id or uuid.uuid4().hex))


# ---------------------------------------------------------------------------
# SMTP send helpers
# ---------------------------------------------------------------------------

def _build_mime(
    to_addr: str,
    subject: str,
    body: str,
    in_reply_to: Optional[str] = None,
    references: Optional[str] = None,
) -> tuple:
    """Return (MIMEMultipart, new_message_id)."""
    from_addr = os.environ.get("EMAIL_ADDRESS", "")
    domain = from_addr.split("@")[1] if "@" in from_addr else "hermes.local"
    msg_id = f"<hermes-{uuid.uuid4().hex[:16]}@{domain}>"

    msg = MIMEMultipart()
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg["Message-ID"] = msg_id

    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
        # Proper RFC 2822 threading: References = prior chain + parent id
        ref_chain = f"{references} {in_reply_to}".strip() if references else in_reply_to
        msg["References"] = ref_chain

    msg.attach(MIMEText(body, "plain", "utf-8"))
    return msg, msg_id


def _smtp_send(msg: MIMEMultipart) -> None:
    smtp = _get_smtp()
    try:
        smtp.send_message(msg)
    finally:
        try:
            smtp.quit()
        except Exception:
            smtp.close()


# ---------------------------------------------------------------------------
# Action: fetch
# ---------------------------------------------------------------------------

def _action_fetch(args: dict) -> str:
    limit = int(args.get("limit", 20))
    folder = args.get("folder", "INBOX")
    min_priority = args.get("min_priority", "low")   # skip/low/medium/high
    since_date = args.get("since")  # IMAP date string e.g. "05-Apr-2026"

    priority_rank = {"skip": 0, "low": 1, "medium": 2, "high": 3}
    min_rank = priority_rank.get(min_priority, 1)

    results = []
    try:
        imap = _get_imap()
        try:
            imap.select(folder)
            criteria = "UNSEEN"
            if since_date:
                criteria = f'(SINCE "{since_date}")'
            status, data = imap.uid("search", None, criteria)
            if status != "OK" or not data or not data[0]:
                return json.dumps({"status": "ok", "emails": [], "count": 0})

            uids = data[0].split()
            # Fetch newest first — UIDs are ascending, so reverse
            uids = uids[-min(limit * 3, len(uids)):]  # over-fetch to account for skip priority
            uids = list(reversed(uids))

            for uid in uids:
                if len(results) >= limit:
                    break
                status, msg_data = imap.uid("fetch", uid, "(RFC822)")
                if status != "OK":
                    continue
                raw = msg_data[0][1]
                msg = email_lib.message_from_bytes(raw)

                sender_raw = msg.get("From", "")
                sender_addr = _extract_email_address(sender_raw)
                sender_name = _decode_header_value(sender_raw)
                if "<" in sender_name:
                    sender_name = sender_name.split("<")[0].strip().strip('"')

                subject = _decode_header_value(msg.get("Subject", "(no subject)"))
                date_str = msg.get("Date", "")
                message_id = msg.get("Message-ID", "").strip()
                in_reply_to = msg.get("In-Reply-To", "").strip()
                references = msg.get("References", "").strip()
                headers = dict(msg.items())

                priority = _score_priority(sender_addr, subject, headers)
                if priority_rank[priority] < min_rank:
                    continue

                body = _extract_body(msg)
                snippet = body[:300].replace("\n", " ").strip()

                # Extract List-Unsubscribe header for unsubscribe support
                unsub = msg.get("List-Unsubscribe", "")

                results.append({
                    "uid": uid.decode() if isinstance(uid, bytes) else uid,
                    "message_id": message_id,
                    "sender": f"{sender_name} <{sender_addr}>",
                    "sender_addr": sender_addr,
                    "subject": subject,
                    "date": date_str,
                    "priority": priority,
                    "snippet": snippet,
                    "in_reply_to": in_reply_to,
                    "references": references,
                    "has_unsubscribe": bool(unsub),
                    "unsubscribe_header": unsub,
                })
        finally:
            try:
                imap.logout()
            except Exception:
                pass

    except Exception as exc:
        return json.dumps({"error": str(exc)})

    return json.dumps({
        "status": "ok",
        "count": len(results),
        "emails": results,
    }, default=str)


# ---------------------------------------------------------------------------
# Action: send
# ---------------------------------------------------------------------------

def _action_send(args: dict) -> str:
    to_addr = args.get("to", "")
    subject = args.get("subject", "")
    body = args.get("body", "")
    in_reply_to = args.get("in_reply_to")
    references = args.get("references")

    if not to_addr:
        return json.dumps({"error": "to is required"})
    if not subject:
        return json.dumps({"error": "subject is required"})
    if not body:
        return json.dumps({"error": "body is required"})

    try:
        msg, msg_id = _build_mime(to_addr, subject, body, in_reply_to, references)
        _smtp_send(msg)
        return json.dumps({"status": "sent", "message_id": msg_id, "to": to_addr, "subject": subject})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Action: draft (human-in-the-loop)
# ---------------------------------------------------------------------------

def _action_draft(args: dict) -> str:
    to_addr = args.get("to", "")
    subject = args.get("subject", "")
    body = args.get("body", "")
    in_reply_to = args.get("in_reply_to")
    references = args.get("references")

    if not to_addr or not subject or not body:
        return json.dumps({"error": "to, subject, and body are required"})

    draft_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()

    try:
        conn = _open_db()
        conn.execute(
            "INSERT INTO email_drafts (id,to_addr,subject,body,in_reply_to,references,created_at,status) "
            "VALUES (?,?,?,?,?,?,?,'pending')",
            (draft_id, to_addr, subject, body, in_reply_to, references, now),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        return json.dumps({"error": f"DB error: {exc}"})

    return json.dumps({
        "status": "draft_created",
        "draft_id": draft_id,
        "to": to_addr,
        "subject": subject,
        "body_preview": body[:300],
        "note": f"Draft saved (id={draft_id}). Use send_draft to approve and send, or reject_draft to discard.",
    })


def _action_send_draft(args: dict) -> str:
    draft_id = args.get("id", "").strip()
    if not draft_id:
        return json.dumps({"error": "id is required"})

    try:
        conn = _open_db()
        row = conn.execute(
            "SELECT * FROM email_drafts WHERE id=?", (draft_id,)
        ).fetchone()
        if not row:
            conn.close()
            return json.dumps({"error": f"Draft {draft_id} not found"})
        if row["status"] != "pending":
            conn.close()
            return json.dumps({"error": f"Draft {draft_id} status is '{row['status']}', not pending"})

        msg, msg_id = _build_mime(
            row["to_addr"], row["subject"], row["body"],
            row["in_reply_to"], row["references"],
        )
        _smtp_send(msg)

        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE email_drafts SET status='sent', sent_at=? WHERE id=?",
            (now, draft_id),
        )
        conn.commit()
        conn.close()

        return json.dumps({
            "status": "sent",
            "draft_id": draft_id,
            "message_id": msg_id,
            "to": row["to_addr"],
            "subject": row["subject"],
        })
    except Exception as exc:
        try:
            conn.execute(
                "UPDATE email_drafts SET status='error', error=? WHERE id=?",
                (str(exc), draft_id),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass
        return json.dumps({"error": str(exc)})


def _action_reject_draft(args: dict) -> str:
    draft_id = args.get("id", "").strip()
    if not draft_id:
        return json.dumps({"error": "id is required"})
    try:
        conn = _open_db()
        conn.execute(
            "UPDATE email_drafts SET status='rejected' WHERE id=? AND status='pending'",
            (draft_id,),
        )
        conn.commit()
        conn.close()
        return json.dumps({"status": "rejected", "draft_id": draft_id})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def _action_list_drafts(args: dict) -> str:
    status_filter = args.get("status", "pending")
    try:
        conn = _open_db()
        if status_filter == "all":
            rows = conn.execute(
                "SELECT id,to_addr,subject,created_at,status,sent_at FROM email_drafts ORDER BY created_at DESC LIMIT 50"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id,to_addr,subject,created_at,status,sent_at FROM email_drafts WHERE status=? ORDER BY created_at DESC LIMIT 50",
                (status_filter,),
            ).fetchall()
        conn.close()
        drafts = [dict(r) for r in rows]
        return json.dumps({"status": "ok", "count": len(drafts), "drafts": drafts})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Action: index (embed + store in Qdrant)
# ---------------------------------------------------------------------------

def _action_index(args: dict) -> str:
    limit = int(args.get("limit", 50))
    folder = args.get("folder", "INBOX")
    since_date = args.get("since")

    try:
        mgr, emb = _get_email_plugin()
    except ImportError as exc:
        return json.dumps({"error": str(exc)})

    # Ensure payload indexes exist for email_archive collection
    try:
        from qdrant_client.models import PayloadSchemaType
        existing = {c.name for c in mgr.client.get_collections().collections}
        if _EMAIL_COLLECTION in existing:
            # Try to create indexes (silently skip if already exist)
            for field in ("sender_addr", "date", "priority"):
                try:
                    mgr.client.create_payload_index(
                        collection_name=_EMAIL_COLLECTION,
                        field_name=field,
                        field_schema=PayloadSchemaType.KEYWORD,
                    )
                except Exception:
                    pass
    except Exception:
        pass

    emails_raw = []
    try:
        imap = _get_imap()
        try:
            imap.select(folder)
            criteria = "ALL"
            if since_date:
                criteria = f'(SINCE "{since_date}")'
            status, data = imap.uid("search", None, criteria)
            if status != "OK" or not data or not data[0]:
                mgr.disconnect()
                return json.dumps({"status": "ok", "indexed": 0})

            uids = data[0].split()[-min(limit, len(data[0].split())):]

            for uid in uids:
                status, msg_data = imap.uid("fetch", uid, "(RFC822)")
                if status != "OK":
                    continue
                msg = email_lib.message_from_bytes(msg_data[0][1])
                sender_raw = msg.get("From", "")
                sender_addr = _extract_email_address(sender_raw)
                subject = _decode_header_value(msg.get("Subject", ""))
                date_str = msg.get("Date", "")
                message_id = msg.get("Message-ID", "").strip()
                headers = dict(msg.items())
                priority = _score_priority(sender_addr, subject, headers)
                body = _extract_body(msg)

                emails_raw.append({
                    "message_id": message_id,
                    "sender_addr": sender_addr,
                    "subject": subject,
                    "date": date_str,
                    "priority": priority,
                    "body": body,
                })
        finally:
            try:
                imap.logout()
            except Exception:
                pass
    except Exception as exc:
        mgr.disconnect()
        return json.dumps({"error": f"IMAP error: {exc}"})

    if not emails_raw:
        mgr.disconnect()
        return json.dumps({"status": "ok", "indexed": 0})

    # Build points
    try:
        from qdrant_client.models import PointStruct

        texts = [
            f"{e['subject']} {e['body'][:512]}"
            for e in emails_raw
        ]
        vectors = emb.encode_batch(texts, show_progress=False)
        now = datetime.now(timezone.utc).isoformat()

        points = []
        for em, vec in zip(emails_raw, vectors):
            pid = _email_point_id(em["message_id"] or em["subject"] + em["date"])
            payload = {
                "message_id": em["message_id"],
                "sender_addr": em["sender_addr"],
                "subject": em["subject"],
                "date": em["date"],
                "priority": em["priority"],
                "body_snippet": em["body"][:400],
                "indexed_at": now,
            }
            points.append(PointStruct(id=pid, vector=vec.tolist(), payload=payload))

        batch_size = 64
        for i in range(0, len(points), batch_size):
            mgr.client.upsert(
                collection_name=_EMAIL_COLLECTION,
                points=points[i : i + batch_size],
            )

        info = mgr.collection_info()
    except Exception as exc:
        mgr.disconnect()
        return json.dumps({"error": f"Indexing error: {exc}"})
    finally:
        mgr.disconnect()

    return json.dumps({
        "status": "ok",
        "indexed": len(points),
        "collection": info,
    }, default=str)


# ---------------------------------------------------------------------------
# Action: search (semantic RAG over email archive)
# ---------------------------------------------------------------------------

def _action_search(args: dict) -> str:
    query = args.get("query", "")
    limit = int(args.get("limit", 10))
    score_threshold = float(args.get("score_threshold", 0.3))
    sender_filter = args.get("sender")
    priority_filter = args.get("priority")
    since = args.get("since")

    if not query:
        return json.dumps({"error": "query is required"})

    try:
        mgr, emb = _get_email_plugin()
    except ImportError as exc:
        return json.dumps({"error": str(exc)})

    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

        vector = emb.encode_text(query)

        must = []
        if sender_filter:
            must.append(FieldCondition(
                key="sender_addr",
                match=MatchValue(value=sender_filter.lower()),
            ))
        if priority_filter:
            must.append(FieldCondition(
                key="priority",
                match=MatchValue(value=priority_filter),
            ))

        filt = Filter(must=must) if must else None

        hits = mgr.client.search(
            collection_name=_EMAIL_COLLECTION,
            query_vector=vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=filt,
            with_payload=True,
        )

        results = [
            {
                "score": round(h.score, 4),
                "sender": h.payload.get("sender_addr"),
                "subject": h.payload.get("subject"),
                "date": h.payload.get("date"),
                "priority": h.payload.get("priority"),
                "snippet": h.payload.get("body_snippet", "")[:200],
                "message_id": h.payload.get("message_id"),
            }
            for h in hits
        ]

        return json.dumps({"status": "ok", "count": len(results), "results": results})
    except Exception as exc:
        return json.dumps({"error": str(exc)})
    finally:
        mgr.disconnect()


# ---------------------------------------------------------------------------
# Action: unsubscribe (browser automation + List-Unsubscribe header)
# ---------------------------------------------------------------------------

def _action_unsubscribe(args: dict) -> str:
    """
    Smart unsubscribe from a sender's mailing list.

    Strategy:
    1. Search IMAP for recent emails from the sender.
    2. Extract the List-Unsubscribe header (RFC 8058).
    3a. If mailto:  send an unsubscribe request email.
    3b. If https:   use browser automation to navigate and submit the form.
    4. Return a summary of what was done.
    """
    sender = args.get("sender", "").strip().lower()
    message_id = args.get("message_id", "").strip()

    if not sender and not message_id:
        return json.dumps({"error": "Provide either sender or message_id"})

    unsub_header = ""
    found_subject = ""

    # --- Fetch the List-Unsubscribe header from IMAP ---
    try:
        imap = _get_imap()
        try:
            imap.select("INBOX")
            if message_id:
                criteria = f'HEADER Message-ID "{message_id}"'
            else:
                criteria = f'FROM "{sender}"'
            status, data = imap.uid("search", None, criteria)
            if status == "OK" and data and data[0]:
                uid = data[0].split()[-1]
                status, msg_data = imap.uid("fetch", uid, "(RFC822)")
                if status == "OK":
                    msg = email_lib.message_from_bytes(msg_data[0][1])
                    unsub_header = msg.get("List-Unsubscribe", "")
                    found_subject = _decode_header_value(msg.get("Subject", ""))
                    if not sender:
                        sender = _extract_email_address(msg.get("From", ""))
        finally:
            try:
                imap.logout()
            except Exception:
                pass
    except Exception as exc:
        return json.dumps({"error": f"IMAP error: {exc}"})

    if not unsub_header:
        return json.dumps({
            "status": "no_unsubscribe_header",
            "sender": sender,
            "message": (
                "No List-Unsubscribe header found in recent emails from this sender. "
                "The sender may not support automated unsubscribe. "
                "Try searching the email body for an unsubscribe link manually."
            ),
        })

    # --- Parse unsubscribe URLs from header ---
    # Format: List-Unsubscribe: <https://example.com/unsub?id=abc>, <mailto:unsub@example.com>
    http_urls = re.findall(r"<(https?://[^>]+)>", unsub_header)
    mailto_addrs = re.findall(r"<mailto:([^>]+)>", unsub_header)

    actions_taken = []

    # --- mailto unsubscribe ---
    for mailto in mailto_addrs:
        parts = mailto.split("?", 1)
        to_addr = parts[0]
        # Extract subject from query params if present
        unsub_subject = "Unsubscribe"
        if len(parts) > 1:
            subject_m = re.search(r"subject=([^&]+)", parts[1], re.IGNORECASE)
            if subject_m:
                from urllib.parse import unquote_plus
                unsub_subject = unquote_plus(subject_m.group(1))
        try:
            from_addr = os.environ.get("EMAIL_ADDRESS", "")
            domain = from_addr.split("@")[1] if "@" in from_addr else "hermes.local"
            msg_id = f"<hermes-unsub-{uuid.uuid4().hex[:12]}@{domain}>"
            msg = MIMEMultipart()
            msg["From"] = from_addr
            msg["To"] = to_addr
            msg["Subject"] = unsub_subject
            msg["Message-ID"] = msg_id
            msg.attach(MIMEText("Please unsubscribe me from this mailing list.", "plain", "utf-8"))
            _smtp_send(msg)
            actions_taken.append({"method": "mailto", "to": to_addr, "subject": unsub_subject, "status": "sent"})
        except Exception as exc:
            actions_taken.append({"method": "mailto", "to": to_addr, "error": str(exc)})

    # --- HTTP unsubscribe via browser automation ---
    for url in http_urls:
        try:
            from tools.browser_tool import browser_navigate_tool, browser_snapshot_tool, browser_click_tool
            # Navigate to unsubscribe page
            nav_result = browser_navigate_tool({"url": url})
            if isinstance(nav_result, str):
                nav_result = json.loads(nav_result) if nav_result.startswith("{") else {"result": nav_result}

            # Take snapshot to see what's on the page
            snapshot = browser_snapshot_tool({})

            # Look for confirm/submit button in the page and click it
            # Parse snapshot for buttons containing unsubscribe-related text
            snapshot_text = json.dumps(snapshot) if not isinstance(snapshot, str) else snapshot
            confirm_patterns = [
                "unsubscribe", "confirm", "opt out", "opt-out", "remove me",
                "submit", "yes", "continue",
            ]
            clicked = False
            for pattern in confirm_patterns:
                if pattern in snapshot_text.lower():
                    try:
                        click_result = browser_click_tool({"text": pattern})
                        if click_result:
                            clicked = True
                            break
                    except Exception:
                        continue

            actions_taken.append({
                "method": "browser",
                "url": url,
                "status": "completed" if clicked else "navigated_no_button_found",
            })
        except ImportError:
            # Browser tool not available — return the URL for manual action
            actions_taken.append({
                "method": "browser",
                "url": url,
                "status": "browser_unavailable",
                "manual_action": f"Visit this URL to unsubscribe: {url}",
            })
        except Exception as exc:
            actions_taken.append({"method": "browser", "url": url, "error": str(exc)})

    if not actions_taken:
        return json.dumps({
            "status": "nothing_done",
            "sender": sender,
            "unsub_header": unsub_header,
        })

    return json.dumps({
        "status": "ok",
        "sender": sender,
        "subject": found_subject,
        "actions": actions_taken,
    })


# ---------------------------------------------------------------------------
# Action: commitments (track action items / deadlines)
# ---------------------------------------------------------------------------

def _action_add_commitment(args: dict) -> str:
    action_text = args.get("action", "").strip()
    deadline = args.get("deadline", "").strip() or None
    message_id = args.get("message_id", "").strip() or None
    sender = args.get("sender", "").strip() or None
    subject = args.get("subject", "").strip() or None

    if not action_text:
        return json.dumps({"error": "action is required"})

    cid = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()

    try:
        conn = _open_db()
        conn.execute(
            "INSERT INTO email_commitments (id,message_id,sender,subject,action_text,deadline,created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (cid, message_id, sender, subject, action_text, deadline, now),
        )
        conn.commit()
        conn.close()
        return json.dumps({
            "status": "ok",
            "id": cid,
            "action": action_text,
            "deadline": deadline,
        })
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def _action_commitments(args: dict) -> str:
    include_done = args.get("include_done", False)
    try:
        conn = _open_db()
        if include_done:
            rows = conn.execute(
                "SELECT * FROM email_commitments ORDER BY deadline ASC, created_at ASC LIMIT 100"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM email_commitments WHERE completed_at IS NULL "
                "ORDER BY deadline ASC, created_at ASC LIMIT 100"
            ).fetchall()
        conn.close()
        items = [dict(r) for r in rows]
        return json.dumps({"status": "ok", "count": len(items), "commitments": items})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def _action_complete(args: dict) -> str:
    cid = args.get("id", "").strip()
    if not cid:
        return json.dumps({"error": "id is required"})
    try:
        conn = _open_db()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE email_commitments SET completed_at=? WHERE id=? AND completed_at IS NULL",
            (now, cid),
        )
        conn.commit()
        conn.close()
        return json.dumps({"status": "ok", "id": cid, "completed_at": now})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

_ACTIONS = {
    "fetch": _action_fetch,
    "send": _action_send,
    "draft": _action_draft,
    "send_draft": _action_send_draft,
    "reject_draft": _action_reject_draft,
    "list_drafts": _action_list_drafts,
    "index": _action_index,
    "search": _action_search,
    "unsubscribe": _action_unsubscribe,
    "add_commitment": _action_add_commitment,
    "commitments": _action_commitments,
    "complete": _action_complete,
}


def email_tool(args: dict, **kw) -> str:
    action = args.get("action", "").strip()
    if not action:
        return json.dumps({"error": "action is required", "valid_actions": list(_ACTIONS)})
    handler = _ACTIONS.get(action)
    if not handler:
        return json.dumps({"error": f"Unknown action '{action}'", "valid_actions": list(_ACTIONS)})
    try:
        return handler(args)
    except Exception as exc:
        logger.exception("email_tool action=%s failed", action)
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def _check_email() -> bool:
    return bool(
        os.environ.get("EMAIL_ADDRESS")
        and os.environ.get("EMAIL_PASSWORD")
        and os.environ.get("EMAIL_IMAP_HOST")
        and os.environ.get("EMAIL_SMTP_HOST")
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

EMAIL_TOOL_SCHEMA = {
    "name": "email",
    "description": (
        "Comprehensive email management: fetch, send, draft (with human approval), "
        "semantic search over email history (RAG), smart unsubscribe, and commitment tracking. "
        "Use 'draft' + 'send_draft' for human-in-the-loop sending."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(_ACTIONS.keys()),
                "description": (
                    "fetch — fetch recent emails with priority scoring. "
                    "send — send an email directly. "
                    "draft — create a draft held for approval (returns draft_id). "
                    "send_draft — approve and send draft by id. "
                    "reject_draft — discard a draft by id. "
                    "list_drafts — list drafts (status: pending/sent/rejected/all). "
                    "index — embed emails into Qdrant for semantic search. "
                    "search — semantic search over indexed email history. "
                    "unsubscribe — smart unsubscribe from a sender. "
                    "add_commitment — record an action item / deadline from an email. "
                    "commitments — list open action items and deadlines. "
                    "complete — mark a commitment done by id."
                ),
            },
            # --- fetch ---
            "folder": {
                "type": "string",
                "description": "[fetch/index] IMAP folder to fetch from (default: INBOX).",
            },
            "limit": {
                "type": "integer",
                "description": "[fetch/index] Max emails to return (default: 20).",
            },
            "min_priority": {
                "type": "string",
                "enum": ["skip", "low", "medium", "high"],
                "description": "[fetch] Minimum priority to include (default: low). Use 'medium' to skip newsletters.",
            },
            "since": {
                "type": "string",
                "description": "[fetch/index] IMAP date string e.g. '01-Apr-2026' to fetch since a date.",
            },
            # --- send / draft ---
            "to": {
                "type": "string",
                "description": "[send/draft] Recipient email address.",
            },
            "subject": {
                "type": "string",
                "description": "[send/draft/add_commitment] Email subject or commitment subject.",
            },
            "body": {
                "type": "string",
                "description": "[send/draft] Email body text.",
            },
            "in_reply_to": {
                "type": "string",
                "description": "[send/draft] Message-ID of the email being replied to (for threading).",
            },
            "references": {
                "type": "string",
                "description": "[send/draft] Full References header chain from the original email.",
            },
            # --- draft management ---
            "id": {
                "type": "string",
                "description": "[send_draft/reject_draft/complete] Draft ID or commitment ID.",
            },
            "status": {
                "type": "string",
                "description": "[list_drafts] Filter by status: pending/sent/rejected/all (default: pending).",
            },
            # --- search ---
            "query": {
                "type": "string",
                "description": "[search] Natural language query e.g. 'invoice from Adobe last month'.",
            },
            "score_threshold": {
                "type": "number",
                "description": "[search] Minimum cosine similarity (default: 0.3).",
            },
            "sender": {
                "type": "string",
                "description": "[search/unsubscribe] Filter by sender email address.",
            },
            "priority": {
                "type": "string",
                "description": "[search] Filter results by priority (skip/low/medium/high).",
            },
            # --- unsubscribe ---
            "message_id": {
                "type": "string",
                "description": "[unsubscribe/add_commitment] Message-ID of the source email.",
            },
            # --- commitments ---
            "action": {
                "type": "string",
                "description": "[add_commitment] Description of the action/task to track.",
            },
            "deadline": {
                "type": "string",
                "description": "[add_commitment] ISO-8601 deadline date e.g. '2026-04-15'.",
            },
            "include_done": {
                "type": "boolean",
                "description": "[commitments] Include completed commitments (default: false).",
            },
        },
        "required": ["action"],
    },
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from tools.registry import registry

registry.register(
    name="email",
    toolset="email",
    schema=EMAIL_TOOL_SCHEMA,
    handler=email_tool,
    check_fn=_check_email,
    emoji="📧",
)
