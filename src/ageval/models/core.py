"""
For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.

Get API models.

Currently implemented via langhchain, but could 
be adapted to work via other frameworks instead.
"""

import ast
from loguru import logger
from langchain_core.language_models import BaseChatModel

from ageval.models.dummy import DummyLangchainModel
from ageval.utils import setup_api_keys
import time
import os
import threading


class RateLimiter:
    _used_tokens: int
    _last_used_tokens_update: float
    _tokens_per_minute: int
    _lock: threading.Lock

    def __init__(self, tokens_per_minute: int):
        self._tokens_per_minute = tokens_per_minute
        self._used_tokens = -tokens_per_minute
        self._last_used_tokens_update = time.time()
        self._lock = threading.Lock()

    def add_used_tokens(self, n_tokens: int):
        with self._lock:
            self._used_tokens += n_tokens
        logger.info(f"Adding usage: {n_tokens}")

    def maybe_sleep(self):
        with self._lock:
            cur_time = time.time()
            tokens_per_second = self._tokens_per_minute / 60
            used_tokens = (
                self._used_tokens
                - (cur_time - self._last_used_tokens_update) * tokens_per_second
            )
            logger.info(f"Used tokens: {used_tokens}.")
            self._used_tokens = max(-self._tokens_per_minute, used_tokens)
            self._last_used_tokens_update = cur_time
        if used_tokens > 0:  # We overspent, sleep a bit.
            time.sleep(-(used_tokens * tokens_per_second))


class APIModel:

    def __init__(self, langchain_model: BaseChatModel, model_name: str) -> None:
        self.langchain_model = langchain_model
        self.model_name = model_name
        self.rate_limiter = RateLimiter(
            100_000 / int(os.environ.get("HYDRA_N_JOBS", 1))
        )

    def invoke_with_messages(
        self,
        messages: list[dict],
        return_type: str = None,
    ) -> str:
        """Get a string response based on messages.

        Messaged should be in the form [{"role": "<user/system>", "content", "..."}]"""

        raw_output = self.langchain_model.invoke(messages)
        n_tokens_used = raw_output.response_metadata["token_usage"]["total_tokens"]
        self.rate_limiter.add_used_tokens(n_tokens_used)
        self.rate_limiter.maybe_sleep()
        model_output = raw_output.content
        parsed_output = self.parse_output(model_output, return_type=return_type)

        return parsed_output

    def invoke(self, text: str, return_type: str = None):
        messages = [{"role": "user", "content": text}]
        return self.invoke_with_messages(messages, return_type=return_type)

    @staticmethod
    def parse_output(model_output: str, return_type: str | None):
        if return_type == "list":
            model_output = model_output.strip()
            try:
                parsed_output = ast.literal_eval(model_output)
                assert isinstance(parsed_output, list)
                return parsed_output
            except:
                logger.error(
                    f"Could not parse list correctly: '{model_output}'. Returning empty list"
                )
                return []
        elif return_type is None:
            return model_output
        else:
            raise ValueError(f"Return type {return_type} not recognized.")


def get_model(
    model: str,
    temp: float = 0.0,
    max_tokens: int = 512,
    model_kwargs: dict | None = None,
) -> APIModel:
    """Get API model."""

    if model_kwargs is None:
        model_kwargs = {}

    setup_api_keys()

    model_provider, model_name = model.split("/")

    if model_provider == "openai":
        from langchain_openai import ChatOpenAI
        import os

        # Support custom base URLs via environment variable
        base_url = os.environ.get("OPENAI_API_BASE")
        
        langchain_model = ChatOpenAI(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temp,
            model_kwargs=model_kwargs,
            max_retries=500,
            base_url=base_url,
        )
    elif model_provider == "openai-assistant":
        from ageval.models.openai import AssistantModelOpenAI

        langchain_model = AssistantModelOpenAI(
            model_name=model_name,
            temperature=temp,
            **model_kwargs,
        )
    elif model_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        langchain_model = ChatAnthropic(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temp,
            model_kwargs=model_kwargs,
        )
    elif model_provider == "google":
        # install via pip install langchain-google-genai
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Note: this require GOOGLE_API_KEY to be set
        # (can be set via ageval_secrets.toml)
        # Note: this is NOT via Vertex API
        langchain_model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temp,
            max_tokens=max_tokens,
            max_retries=500,
        )

    elif model_provider == "apple_lc":
        import ageval.internal.endpoint_model

        langchain_model = ageval.internal.endpoint_model.get_internal_langchain_model(
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temp,
        )
    elif model_provider == "apple":
        import ageval.internal.endpoint_model

        langchain_model = ageval.internal.endpoint_model.InternalModel(
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temp,
        )
    elif model_provider == "dummy":
        logger.warning("Dummy model used. This should only be used for debugging.")
        langchain_model = DummyLangchainModel(**model_kwargs)
    else:
        raise ValueError(
            f"Requested model provider currently not available: {model_provider}"
        )

    return APIModel(langchain_model=langchain_model, model_name=model)
