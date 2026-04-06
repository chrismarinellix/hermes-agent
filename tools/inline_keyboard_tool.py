"""Inline Keyboard Tool — send Telegram messages with tappable inline buttons.

Sends a message to a Telegram chat with an InlineKeyboardMarkup attached.
When the user taps a button, the callback_data is dispatched as a regular
message event, triggering the agent as if they had typed it.

Intended use: after displaying a numbered list (e.g. watch list, search results),
send a keyboard so the user can tap to select rather than typing.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)


INLINE_KEYBOARD_SCHEMA = {
    "name": "send_inline_keyboard",
    "description": (
        "Send a Telegram message with tappable inline buttons. "
        "Use this after displaying a numbered list so the user can tap a button "
        "instead of typing. Each button has a label shown in Telegram and callback_data "
        "dispatched as a message when tapped. "
        "Only works when the current session is Telegram."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The message text to display above the buttons."
            },
            "buttons": {
                "type": "array",
                "description": (
                    "List of button rows. Each row is a list of button objects. "
                    "Each button has a \"label\" and EITHER \"data\" (callback text sent when "
                    "tapped) OR \"url\" (opens a link). "
                    "Put 2-3 buttons per row for compact layout. "
                    "Example: [[{\"label\": \"1. Warfare\", \"data\": \"download Warfare 2026\"}, "
                    "{\"label\": \"2. Accountant 2\", \"data\": \"download The Accountant 2 2026\"}], "
                    "[{\"label\": \"Open IMDb\", \"url\": \"https://imdb.com/title/tt123\"}]]"
                ),
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "data": {"type": "string", "description": "Callback text dispatched as a message when tapped."},
                            "url": {"type": "string", "description": "URL to open when tapped (mutually exclusive with data)."}
                        },
                        "required": ["label"]
                    }
                }
            },
            "silent": {
                "type": "boolean",
                "description": "Send without notification sound. Good for non-urgent updates."
            },
            "disable_preview": {
                "type": "boolean",
                "description": "Suppress link preview in the message text."
            },
            "chat_id": {
                "type": "string",
                "description": "Telegram chat ID. Defaults to current session's chat ID."
            },
            "thread_id": {
                "type": "string",
                "description": "Optional message thread ID for forum topics."
            }
        },
        "required": ["text", "buttons"]
    }
}


def send_inline_keyboard_tool(args, **kw):
    """Send a Telegram message with inline keyboard buttons."""
    text = args.get("text", "")
    buttons = args.get("buttons", [])
    chat_id = args.get("chat_id")
    thread_id = args.get("thread_id")

    if not text:
        return json.dumps({"error": "text is required"})
    if not buttons:
        return json.dumps({"error": "buttons is required"})

    silent = args.get("silent", False)
    disable_preview = args.get("disable_preview", False)

    # Validate button structure
    for row in buttons:
        for btn in row:
            if "label" not in btn:
                return json.dumps({"error": "Each button must have a 'label' key"})
            if "data" not in btn and "url" not in btn:
                return json.dumps({"error": "Each button must have either 'data' or 'url'"})
            # Telegram callback_data max is 64 bytes
            if "data" in btn and len(btn["data"].encode("utf-8")) > 64:
                btn["data"] = btn["data"][:60]

    # Resolve chat_id from current session if not provided
    if not chat_id:
        chat_id = _get_current_chat_id()
    if not chat_id:
        return json.dumps({"error": "Could not determine chat_id. Pass it explicitly."})

    try:
        token = _get_telegram_token()
    except Exception as e:
        return json.dumps({"error": f"Could not get Telegram token: {e}"})

    if not token:
        return json.dumps({"error": "Telegram token not available"})

    try:
        from model_tools import _run_async
        result = _run_async(_send_inline_keyboard_async(
            token, chat_id, text, buttons, thread_id,
            silent=silent, disable_preview=disable_preview,
        ))
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": f"Failed to send inline keyboard: {e}"})


async def _send_inline_keyboard_async(
    token, chat_id, text, buttons, thread_id=None,
    *, silent=False, disable_preview=False,
):
    """Async implementation using python-telegram-bot."""
    try:
        from telegram import Bot, InlineKeyboardMarkup, InlineKeyboardButton
        from telegram.constants import ParseMode
        from gateway.platforms.telegram import TelegramAdapter, _strip_mdv2

        def _make_button(btn):
            if "url" in btn:
                return InlineKeyboardButton(btn["label"], url=btn["url"])
            return InlineKeyboardButton(btn["label"], callback_data=btn["data"])

        keyboard = [
            [_make_button(btn) for btn in row]
            for row in buttons
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        # Format text using gateway's markdown→MarkdownV2 converter
        try:
            adapter = TelegramAdapter.__new__(TelegramAdapter)
            formatted = adapter.format_message(text)
        except Exception:
            formatted = text

        kwargs = {"reply_markup": reply_markup}
        if thread_id:
            kwargs["message_thread_id"] = int(thread_id)
        if silent:
            kwargs["disable_notification"] = True
        if disable_preview:
            kwargs["disable_web_page_preview"] = True

        bot = Bot(token=token)
        async with bot:
            try:
                msg = await bot.send_message(
                    chat_id=int(chat_id),
                    text=formatted,
                    parse_mode=ParseMode.MARKDOWN_V2,
                    **kwargs,
                )
            except Exception as md_err:
                if "parse" in str(md_err).lower() or "markdown" in str(md_err).lower():
                    msg = await bot.send_message(
                        chat_id=int(chat_id),
                        text=_strip_mdv2(formatted),
                        parse_mode=None,
                        **kwargs,
                    )
                else:
                    raise

        return {"success": True, "message_id": str(msg.message_id)}
    except Exception as e:
        logger.error("send_inline_keyboard_async failed: %s", e)
        return {"success": False, "error": str(e)}


def _get_telegram_token():
    """Get Telegram bot token from config or environment."""
    # Try env var first
    token = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_TOKEN")
    if token:
        return token
    # Try gateway config
    try:
        from gateway.config import load_gateway_config, Platform
        config = load_gateway_config()
        pconfig = config.platforms.get(Platform.TELEGRAM)
        if pconfig and pconfig.token:
            return pconfig.token
    except Exception:
        pass
    return None


def _get_current_chat_id():
    """Try to get the current session's chat ID from environment."""
    # Hermes sets HERMES_SESSION_CHAT_ID in the agent session context
    chat_id = os.environ.get("HERMES_SESSION_CHAT_ID")
    if chat_id:
        return chat_id
    # Try gateway home channel as fallback
    try:
        from gateway.config import load_gateway_config, Platform
        config = load_gateway_config()
        home = config.get_home_channel(Platform.TELEGRAM)
        if home:
            return home.chat_id
    except Exception:
        pass
    return None


def _check_inline_keyboard():
    """Check if inline keyboard tool is available."""
    try:
        from telegram import InlineKeyboardMarkup  # noqa
        return True
    except ImportError:
        return False


# --- Registry ---
from tools.registry import registry

registry.register(
    name="send_inline_keyboard",
    toolset="messaging",
    schema=INLINE_KEYBOARD_SCHEMA,
    handler=send_inline_keyboard_tool,
    check_fn=_check_inline_keyboard,
    emoji="⌨️",
)
