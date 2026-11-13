import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model, BaseChatModel

load_dotenv()

# Cheap Haiku by default. Override with CHAT_MODEL in .env, not in code.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def get_model() -> BaseChatModel :
    return init_chat_model(
        os.getenv("CHAT_MODEL", DEFAULT_MODEL),
        model_provider="anthropic",
    )
