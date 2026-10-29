import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "acme_erp.db")

VP_APPROVAL_THRESHOLD = 10_000
CFO_APPROVAL_THRESHOLD = 50_000
MAX_RETRIES = 2


def get_llm(temperature: float = 0) -> ChatOpenAI:
    base_url = os.environ.get("LLM_BASE_URL")
    api_key = os.environ.get("LLM_API_KEY")
    model = os.environ.get("LLM_MODEL")
    max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "4096"))

    if base_url and api_key:
        return ChatOpenAI(
            model=model or "grok-4.7",
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    xai_key = os.environ.get("XAI_API_KEY")
    if xai_key:
        return ChatOpenAI(
            model=model or "grok-4.7",
            api_key=xai_key,
            base_url="https://api.x.ai/v1",
            temperature=temperature,
            max_tokens=max_tokens,
        )

    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        return ChatOpenAI(
            model=model or "openai/gpt-oss-120b",
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=temperature,
            max_tokens=max_tokens,
        )

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        return ChatOpenAI(
            model=model or "gpt-4o",
            api_key=openai_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    raise EnvironmentError(
        "No LLM API key found. Set XAI_API_KEY, GROQ_API_KEY, OPENAI_API_KEY, "
        "or both LLM_BASE_URL and LLM_API_KEY in your .env file."
    )
