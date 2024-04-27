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

# Load your tokens from environment variables
slack_token = os.environ["SLACK_BOT_TOKEN"]
app_token = os.environ["SLACK_APP_TOKEN"]

# Initialize the Slack client
slack_client = WebClient(token=slack_token)
socket_mode_client = SocketModeClient(app_token=app_token, web_client=slack_client)

main_bot = SlackLLM()
determine_reply_bot = SlackLLM(
    system_prompt="""
You are a bot in a Slack channel. Your task is to analyze each incoming message and reply with a confidence score (0%-100%) on whether the message is directed towards you and expects a response. The response should be in valid JSON format. Evaluate the message for direct commands, questions to the bot, or references to the bot's functionalities. Consider the following:

- If a message directly asks a question or issues a command likely meant for the bot (like asking for calculations, creative writing, or specific information only the bot would provide), respond with a high confidence.
- If the message uses phrases that typically indicate interaction with a bot (e.g., "you can", "your function", referring to the bot indirectly), consider a moderate to high confidence.
- If the message is clearly directed at another user or is a general statement not requiring a bot's response, respond with low confidence.

Here are examples to guide you:

Example 1:
Message: Hi! Please help me plan a weekend trip to Maine with my friend. We know we want to see Acadia. What else should we see? Prioritize things that are unique to Maine that are hard to find elsewhere. Be specific.
Response: {"confidence": 90.0, "should_reply": true, "explanation": "This is a task that the bot is suited to help with."}

Example 2:
Message: Steve, are you okay?
Response: {"confidence": 5.0, "should_reply": false, "explanation": "This is a question being directed towards Steve."}

Example 3:
Message: Who are you?
Response: {"confidence": 95.0, "should_reply": true, "explanation": "The bot is the only non-human entity in the channel, so any direct query like this should be perceived as directed towards the bot."}

Example 4:
Message: I’m trying to be fancy, I haven’t written the code to actually make it not reply yet, but it is looking at the message to generate the output you’ll see so it can decide itself whether or not to reply.
Response: {"confidence": 15.0, "should_reply": false, "explanation": "The message is discussing the bot's functionality or settings without directly engaging with it."}

Remember to adjust the bot’s confidence scoring based on the likelihood that a message is addressed to the bot, and not merely mentioning it in passing or discussing it among users.
""",
    max_len=0,
)


def determine_reply(user_message: str) -> bool:
    raw_response = determine_reply_bot.generate_response(user_message, "")
    try:
        response = json.loads(raw_response)
        logger.info(f"Confidence: {response['confidence']}, replying: {response['should_reply']}")
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
    global main_bot

    command = event_data["command"]
    if command == "/system":
        main_bot.update_system_prompt(event_data["text"])
    elif command == "/clear":
        main_bot.clear_history()


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
