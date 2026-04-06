"""Telegram Actions Tool — delete, pin, and manage Telegram messages.

Provides message lifecycle operations beyond send/edit. Works via the
Telegram Bot API using the configured bot token.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)


TELEGRAM_ACTIONS_SCHEMA = {
    "name": "telegram_actions",
    "description": (
        "Manage Telegram messages: delete old messages, pin important ones, "
        "or remove buttons from a previous inline keyboard. "
        "Only works when the current session is Telegram."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["delete", "pin", "unpin", "remove_buttons"],
                "description": (
                    "delete: remove a message. "
                    "pin: pin a message in the chat. "
                    "unpin: unpin a message. "
                    "remove_buttons: strip inline keyboard from a message."
                ),
            },
            "message_id": {
                "type": "string",
                "description": "The message ID to act on.",
            },
            "chat_id": {
                "type": "string",
                "description": "Telegram chat ID. Defaults to current session's chat.",
            },
            "silent": {
                "type": "boolean",
                "description": "For pin: suppress notification (default true).",
            },
        },
        "required": ["action", "message_id"],
    },
}


def telegram_actions_tool(args, **kw):
    """Execute a Telegram message action."""
    action = args.get("action")
    message_id = args.get("message_id")
    chat_id = args.get("chat_id")
    silent = args.get("silent", True)

    if not action or not message_id:
        return json.dumps({"error": "action and message_id are required"})

    if not chat_id:
        chat_id = os.environ.get("HERMES_SESSION_CHAT_ID")
    if not chat_id:
        return json.dumps({"error": "Could not determine chat_id. Pass it explicitly."})

    token = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_TOKEN")
    if not token:
        try:
            from gateway.config import load_gateway_config, Platform
            config = load_gateway_config()
            pconfig = config.platforms.get(Platform.TELEGRAM)
            if pconfig and pconfig.token:
                token = pconfig.token
        except Exception:
            pass
    if not token:
        return json.dumps({"error": "Telegram token not available"})

    try:
        from model_tools import _run_async
        result = _run_async(_telegram_action_async(token, chat_id, message_id, action, silent))
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": f"Action failed: {e}"})


async def _telegram_action_async(token, chat_id, message_id, action, silent):
    """Async dispatch for Telegram actions."""
    try:
        from telegram import Bot
        bot = Bot(token=token)
        async with bot:
            if action == "delete":
                await bot.delete_message(chat_id=int(chat_id), message_id=int(message_id))
                return {"success": True, "action": "deleted"}
            elif action == "pin":
                await bot.pin_chat_message(
                    chat_id=int(chat_id),
                    message_id=int(message_id),
                    disable_notification=silent,
                )
                return {"success": True, "action": "pinned"}
            elif action == "unpin":
                await bot.unpin_chat_message(
                    chat_id=int(chat_id),
                    message_id=int(message_id),
                )
                return {"success": True, "action": "unpinned"}
            elif action == "remove_buttons":
                await bot.edit_message_reply_markup(
                    chat_id=int(chat_id),
                    message_id=int(message_id),
                    reply_markup=None,
                )
                return {"success": True, "action": "buttons_removed"}
            else:
                return {"error": f"Unknown action: {action}"}
    except Exception as e:
        logger.error("telegram_action_async failed: %s", e)
        return {"success": False, "error": str(e)}


def _check_telegram_actions():
    """Check if Telegram actions are available."""
    try:
        from telegram import Bot  # noqa
        return True
    except ImportError:
        return False


# --- Registry ---
from tools.registry import registry

registry.register(
    name="telegram_actions",
    toolset="messaging",
    schema=TELEGRAM_ACTIONS_SCHEMA,
    handler=telegram_actions_tool,
    check_fn=_check_telegram_actions,
    emoji="🔧",
)
