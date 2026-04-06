---
name: ai-news-digest
description: >
  Fetch, index, and summarise the latest AI and tech news into a Telegram digest.
  Scrapes top stories from multiple sources, deduplicates via Qdrant semantic search,
  and formats a concise briefing with headlines, one-line summaries, and source links.
  Used by the daily news cron job and on-demand when the user asks for news.
version: 1.0.0
metadata:
  hermes:
    tags: [News, AI, Research, Telegram, Digest, Feeds]
    related_skills: [arxiv, blogwatcher]
---

# AI News Digest

Produce a daily AI/tech news briefing and deliver it to Telegram.

## What this skill does

1. Calls `index_news` to scrape and embed fresh articles into Qdrant
2. Calls `search_news` to pull the most relevant stories for each topic
3. Deduplicates across topics (same article may surface under multiple queries)
4. Formats a clean Telegram digest and sends it via `send_message`

## Topics to cover every run

Search these in order — use `index_news` first for each, then `search_news`:

| Topic | Query |
|-------|-------|
| LLM releases | `"new AI model release 2025"` |
| AI safety & alignment | `"AI safety alignment research"` |
| AI agents | `"autonomous AI agents agentic systems"` |
| Industry news | `"OpenAI Anthropic Google DeepMind news"` |
| AI tools & dev | `"AI developer tools coding assistant"` |
| Research breakthroughs | `"machine learning research breakthrough paper"` |

## Output format

Send to Telegram as a single message per section. Use this structure:

```
📰 *AI News Digest — {date}*

🤖 *LLM & Models*
• [Title](url) — one-line summary. _(source)_

🛡 *AI Safety*
• ...

⚙️ *AI Agents*
• ...

🏢 *Industry*
• ...

🛠 *Dev Tools*
• ...

🔬 *Research*
• ...

_Next update tomorrow at 8am_
```

## Rules

- Max 3 articles per section, 5 sections minimum
- Skip paywalled articles (no content extracted)
- If an article appears in multiple sections, only show it in the first
- Keep summaries to one sentence — what happened, why it matters
- Prefer articles from last 48 hours; fall back to last 7 days if thin
- Always include the source domain in italics after the summary

## Delivery

```
send_message(platform="telegram", chat_id="-1003701645201", thread_id="10", content=digest)
```

## On-demand usage

When the user asks "what's in the news" or "AI news" or "latest AI updates":
- Run a fresh `index_news` pass with today's date queries
- `search_news` across all topics
- Format and send the digest to the current chat
