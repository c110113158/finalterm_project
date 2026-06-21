"""
app.py
Flask webhook server for the BTC Sentiment LINE Bot.

Required environment variables:
  LINE_CHANNEL_ACCESS_TOKEN - from LINE Developers Console > Messaging API tab
  LINE_CHANNEL_SECRET       - from LINE Developers Console > Basic settings tab

Run locally:
  export LINE_CHANNEL_ACCESS_TOKEN=...
  export LINE_CHANNEL_SECRET=...
  python app.py
"""
import os
import logging

from flask import Flask, request, abort

from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent

import core

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET")

if not CHANNEL_ACCESS_TOKEN or not CHANNEL_SECRET:
    logger.warning(
        "LINE_CHANNEL_ACCESS_TOKEN / LINE_CHANNEL_SECRET are not set. "
        "The /callback route will fail until these env vars are configured."
    )

configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN or "dummy")
handler = WebhookHandler(CHANNEL_SECRET or "dummy")


@app.route("/", methods=["GET"])
def health_check():
    """Simple health check so the host (Render, etc.) sees the service as alive."""
    return "BTC Sentiment LINE Bot is running.", 200


@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    logger.info("Request body: %s", body)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        logger.warning("Invalid signature. Check your channel access token/secret.")
        abort(400)
    except Exception:
        logger.exception("Unexpected error while handling webhook")
        abort(500)

    return "OK", 200


@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_text = event.message.text
    reply_text = core.answer(user_text)

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_text)],
            )
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
