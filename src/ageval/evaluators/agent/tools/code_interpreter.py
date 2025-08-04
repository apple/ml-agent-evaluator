"""
For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.

A tool for code execution.

Uses GPT-4o with code interpreter tool.
"""

from typing import Type
from pydantic.v1 import BaseModel, Field

from ageval.evaluators.agent.tools.base import ToolBase
from ageval.evaluators.agent.tools.registry import register_tool
from ageval.experiments.config import PromptConfig
from ageval.models.utils import retry_with_exponential_backoff


class ServerError(Exception):
    pass


class ClientError(Exception):
    pass


@register_tool("code_interpreter")
class ToolCodeInterpreter(ToolBase):

    assistant_instruction: str = (
        "You are a coding expert. "
        "Your goal is to evaluate whether code from a student is correct. "
        "Write and run code to verify the provided answer to the prompt. "
        "Think of unit tests to verify whether the code is correct. "
        "Only report back whether the solution was correct. "
        "Do not try to correct the code, they need to do that themselves."
    )

    @classmethod
    def create_run_condition(cls, prompts: PromptConfig) -> Type[BaseModel]:
        class TextAssessment(BaseModel):
            """Assessment of a text."""

            code_useful: bool = Field(
                description="Whether text might benefit from running code."
            )

        return TextAssessment

    @classmethod
    def should_run(cls, *, condition_data: dict) -> bool:
        return condition_data["code_useful"]

    @classmethod
    def _code_interpreter_call(cls, *, client, content: str) -> str:
        """Calls the OpenAI code interpreter and returns final message.

        Arguments:
            client: OpenAI client.
            content: String to be used in user message.

        Returns:
            Final message string.

        Raises:
            ServerError: Error where we should simply retry.
            ClientError: Client configuration is incorrect.
        """
        assistant = client.beta.assistants.create(
            instructions=cls.assistant_instruction,
            model="gpt-4o",
            tools=[{"type": "code_interpreter"}],
        )
        thread = client.beta.threads.create()
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=content,
        )

        # Start the run and wait till it finishes.
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=assistant.id,
        )

        if run.status == "completed":
            messages = client.beta.threads.messages.list(thread_id=thread.id)
            final_str = messages.data[0].content[0].text.value
        else:
            if run.last_error.code in ("server_error", "rate_limit_exceeded"):
                raise ServerError()
            else:
                raise ClientError()
        return final_str

    @classmethod
    def run(
        cls, *, text: str, prompt: str, model_name: str, debug: bool = False
    ) -> dict:
        from openai import OpenAI, RateLimitError

        client = OpenAI()
        content = f"For the prompt:\n```{prompt}\n```\nis the provided answer correct?\n```{text}\n```"

        @retry_with_exponential_backoff(
            max_time=20 * 60, errors=(ServerError, RateLimitError)
        )
        def _get_completion_with_backoff() -> str:
            return cls._code_interpreter_call(client=client, content=content)

        return _get_completion_with_backoff()
