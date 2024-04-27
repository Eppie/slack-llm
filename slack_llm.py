import logging
from collections import deque, defaultdict

import ollama  # type: ignore[import-untyped]

DEFAULT_SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
DEFAULT_MAX_LEN = 500
DEFAULT_MODEL = "llama3:latest"

logger = logging.getLogger(__name__)


class SlackLLM:
    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_len: int = DEFAULT_MAX_LEN,
        model: str = DEFAULT_MODEL,
    ):
        self.system_prompt: str = system_prompt
        self.last_messages: dict[str, deque[dict[str, str]]] = defaultdict(lambda: deque(maxlen=max_len))
        self.model: str = model

    def clear_history(self) -> None:
        self.last_messages.clear()

    def update_system_prompt(self, new_system_prompt: str) -> None:
        self.system_prompt = new_system_prompt
        self.clear_history()
        logger.info(f"System prompt is now: {self.system_prompt}")

    @staticmethod
    def generate_message(role: str, content: str) -> dict[str, str]:
        m = {"role": role, "content": content}
        return m

    def generate_system_message(self) -> dict[str, str]:
        return self.generate_message("system", self.system_prompt)

    def generate_user_message(self, user_text: str, channel: str) -> dict[str, str]:
        m = self.generate_message("user", user_text)
        self.last_messages[channel].append(m)
        return m

    def generate_assistant_message(self, assistant_text: str, channel: str) -> dict[str, str]:
        m = self.generate_message("assistant", assistant_text)
        self.last_messages[channel].append(m)
        return m

    def generate_messages(self, user_text: str, channel: str) -> list[dict[str, str]]:
        result = (
            [self.generate_system_message()]
            + list(self.last_messages[channel])
            + [self.generate_user_message(user_text, channel)]
        )
        return result

    def generate_response(self, user_text: str, channel: str, format: str = "") -> str:
        response = ""
        for part in ollama.chat(
            model=self.model,
            messages=self.generate_messages(user_text, channel),
            stream=True,
            format=format,
        ):
            print(part["message"]["content"], end="", flush=True)
            response += part["message"]["content"]
        print("\n")
        self.generate_assistant_message(response, channel)
        return response
