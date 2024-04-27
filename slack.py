import json
import logging
import os
import time
from typing import Any

from slack_sdk import WebClient
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse

from slack_llm import SlackLLM

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


def determine_reply(user_message: str) -> bool:
    raw_response = determine_reply_bot.generate_response(
        f"Using ONLY JSON, evaluate this message: {user_message}",
        "",
        "json",
    )
    try:
        response = json.loads(raw_response)
        logger.info(
            f"Confidence: {response['confidence']}, replying: {response['should_reply']}",
        )
        return bool(response["should_reply"])
    except json.decoder.JSONDecodeError:
        logger.warning(raw_response)
        return True


def handle_message(event_data: dict[str, Any]) -> None:
    logging.info(f"Received event data: {event_data}")
    message = event_data["event"]

    # Check if the message comes from a user and not from the bot itself
    if message.get("subtype") is None and "bot_id" not in message:
        channel_id = message["channel"]
        user_message = message.get("text")
        logging.info(f"Processing message from user: {user_message}")
        should_reply = determine_reply(user_message)
        if should_reply:
            bot_response = main_bot.generate_response(user_message, channel_id)
            bot_response = bot_response.strip("\n")
            # Send the response back to the channel
            slack_client.chat_postMessage(channel=channel_id, text=bot_response)


def handle_slash(event_data: dict[str, Any]) -> None:
    command = event_data["command"]
    if command == "/system":
        main_bot.update_system_prompt(event_data["text"])
    elif command == "/clear":
        main_bot.clear_history()


def event_handler(client: SocketModeClient, req: SocketModeRequest) -> None:
    response = SocketModeResponse(envelope_id=req.envelope_id)
    client.send_socket_mode_response(response)
    match req.type:
        case "events_api":
            event_type = req.payload["event"]["type"]
            if event_type in ("app_mention", "message"):
                handle_message(req.payload)
        case "slash_commands":
            handle_slash(req.payload)
        case _:
            logger.warning(f"Received unknown event type: {req.type}")


if __name__ == "__main__":
    slack_token = os.environ["SLACK_BOT_TOKEN"]
    app_token = os.environ["SLACK_APP_TOKEN"]

    slack_client = WebClient(token=slack_token)
    socket_mode_client = SocketModeClient(app_token=app_token, web_client=slack_client)

    main_bot = SlackLLM()
    with open("determine_reply.txt") as f:
        system_prompt = "\n".join(f.readlines())
    determine_reply_bot = SlackLLM(system_prompt=system_prompt, max_len=0)

    socket_mode_client.connect()

    socket_mode_client.socket_mode_request_listeners.append(event_handler)  # type: ignore[arg-type]
    # Keep the script running
    while True:
        time.sleep(10)
