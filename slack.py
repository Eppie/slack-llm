import logging
import os
import time
from typing import Any

from slack_sdk import WebClient
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse

from slack_llm import SlackLLM

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load your tokens from environment variables
slack_token = os.environ["SLACK_BOT_TOKEN"]
app_token = os.environ["SLACK_APP_TOKEN"]

# Initialize the Slack client
slack_client = WebClient(token=slack_token)
socket_mode_client = SocketModeClient(app_token=app_token, web_client=slack_client)

llm = SlackLLM()


def handle_message(event_data: dict[str, Any]) -> None:
    logging.info(f"Received event data: {event_data}")
    message = event_data["event"]

    # Check if the message comes from a user and not from the bot itself
    if message.get("subtype") is None and "bot_id" not in message:
        channel_id = message["channel"]
        user_message = message.get("text")
        logging.info(
            f"Processing message from user: {user_message}"
        )
        bot_response = llm.generate_response(user_message)
        bot_response = bot_response.strip("\n")
        # Send the response back to the channel
        slack_client.chat_postMessage(channel=channel_id, text=bot_response)


def handle_slash(event_data: dict) -> None:
    global llm

    command = event_data["command"]
    if command == "/system":
        llm.update_system_prompt(event_data["text"])
    elif command == "/clear":
        llm.clear_history()


def event_handler(client: SocketModeClient, req: SocketModeRequest) -> None:
    print("called event handler")
    if req.type == "events_api":
        response = SocketModeResponse(envelope_id=req.envelope_id)
        client.send_socket_mode_response(response)

        event_type = req.payload["event"]["type"]
        print(f"{event_type=}")

        if event_type in ("app_mention", "message"):
            handle_message(req.payload)
    elif req.type == "slash_commands":
        response = SocketModeResponse(envelope_id=req.envelope_id)
        client.send_socket_mode_response(response)
        handle_slash(req.payload)


# Listening to events
socket_mode_client.socket_mode_request_listeners.append(event_handler)  # type: ignore[arg-type]

if __name__ == "__main__":
    socket_mode_client.connect()
    # Keep the script running
    while True:
        time.sleep(10)
