import os

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .schema import LLM


def get_llm(llm: LLM) -> BaseChatModel:
    """
    Get an LLM instance based on the LLM enum.
    """

    match llm:
        case LLM.OPENAI_GPT_41:
            return ChatOpenAI(model="gpt-4.1")
        case LLM.OPENROUTER_GEMINI_3_PRO:
            return ChatOpenAI(
                model="google/gemini-3-pro-preview",
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY"),
            )
        case LLM.OPENROUTER_GEMINI_25_FLASH:
            return ChatOpenAI(
                model="google/gemini-2.5-flash",
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY"),
            )
        case _:
            raise ValueError(f"Unknown LLM: {llm}")
