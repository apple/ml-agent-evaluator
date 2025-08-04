# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from typing import Optional, Union
import openai
from langchain_core.pydantic_v1 import BaseModel
from loguru import logger

import ageval.models.utils
from langchain_core.messages import BaseMessage
from ageval.evaluators.agent.tools.fact_checking import run_web_search
import logging
import json


MAX_RETRY_TIME = 20 * 60


def get_search_result(search_query: str) -> str:
    data = run_web_search(search_query=search_query, max_num_results_per_search=5)
    return json.dumps({"query": search_query, "results": data})


class AssistantModelOpenAI:
    """OpenAI assistant as langchain-compliant model.

    Always has code interpreter available.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        tools: list[dict[str, str]] | None = None,
        temperature: float = 0.0,
        instructions: str | None = None,
    ) -> None:
        """"""
        self.temperature = temperature

        logging.getLogger("httpx").setLevel(logging.WARNING)

        self.client = openai.OpenAI()
        if tools is None:
            tools = (["search", "code_interpreter"],)
        tools_api = []
        for tool in tools:
            if tool == "code_interpreter":
                tools_api.append({"type": "code_interpreter"})
            elif tool == "search":
                tools_api.append(
                    {
                        "type": "function",
                        "function": {
                            "name": "get_search_result",
                            "description": "Get search results from the web. Call this whenever you want to search for extra information to verify something.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "search_query": {
                                        "type": "string",
                                        "description": "Query to use for a web search engine.",
                                    },
                                },
                                "required": ["search_query"],
                                "additionalProperties": False,
                            },
                            "strict": True,
                        },
                    }
                )
            else:
                raise ValueError("Tool '%s' not supported!", tool)

        assert instructions is None
        self._system_message = (
            "You have access to provided tools. Use them if you think they may help."
        )
        self._assistant_kwargs = {
            "temperature": temperature,
            "instructions": instructions,
            "model": "gpt-4o-mini",
            "tools": tools_api,
        }
        self._instruction_assistants = {
            self._system_message: self.client.beta.assistants.create(
                **self._assistant_kwargs,
            )
        }

    def with_structured_output(self, schema: BaseModel):
        """Get model that returns schema-based structured output."""
        raise NotImplementedError

    def invoke(self, messages: Union[list, str]) -> BaseMessage:

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        messages_without_system = [
            message for message in messages if message["role"] != "system"
        ]
        system_messages = [
            message for message in messages if message["role"] == "system"
        ]
        if system_messages:
            assert len(system_messages) == 1
            system_message = f"{system_messages[0]['content']} {self._system_message}"
            if system_message not in self._instruction_assistants:
                kwargs = {**self._assistant_kwargs, "instructions": system_message}
                # We create a new assistant.
                self._instruction_assistants[system_message] = (
                    self.client.beta.assistants.create(
                        **kwargs,
                    )
                )
        else:
            # Just the default system message.
            system_message = self._system_message

        assistant_id = self._instruction_assistants[system_message].id

        def handle_function_calls(client, run, thread_id):
            tool_outputs = []

            # Loop through each tool in the required action section
            for tool in run.required_action.submit_tool_outputs.tool_calls:
                if tool.function.name == "get_search_result":
                    args = json.loads(tool.function.arguments)
                    tool_outputs.append(
                        {"tool_call_id": tool.id, "output": get_search_result(**args)}
                    )
                else:
                    raise ValueError("Unknown function call: %s", tool.function.name)

            assert tool_outputs

            # Submit all tool outputs at once after collecting them in a list
            try:
                run = client.beta.threads.runs.submit_tool_outputs_and_poll(
                    thread_id=thread_id, run_id=run.id, tool_outputs=tool_outputs
                )
                logger.info("Tool outputs submitted successfully.")
                return run
            except Exception as e:
                logger.error("Failed to submit tool outputs:", e)
                raise e

        @ageval.models.utils.retry_with_exponential_backoff(max_time=MAX_RETRY_TIME)
        def get_completion_with_backoff(messages_without_system, client):
            thread = client.beta.threads.create()
            for message in messages_without_system:
                _ = client.beta.threads.messages.create(
                    thread_id=thread.id,
                    role=message["role"],
                    content=message["content"],
                )

            # Start the run and wait till it finishes.
            run = client.beta.threads.runs.create_and_poll(
                thread_id=thread.id,
                assistant_id=assistant_id,
                # tool_choice="required",
            )

            while run.status == "requires_action":
                run = handle_function_calls(client=client, run=run, thread_id=thread.id)

            assert run.status == "completed", run

            return_messages = client.beta.threads.messages.list(thread_id=thread.id)
            response_text = return_messages.data[0].content[0].text.value

            # If you want to debug internal coding steps.
            # run_steps = client.beta.threads.runs.steps.list(
            #     thread_id=thread.id,
            #     run_id=run.id
            # )

            return BaseMessage(
                content=response_text,
                type="text",
                role="assistant",
                response_metadata={
                    "token_usage": {"total_tokens": run.usage.total_tokens}
                },
            )

        return get_completion_with_backoff(
            messages_without_system=messages_without_system, client=self.client
        )

    def generate(
        self,
        prompt: str,
        do_debug: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_attempts: int = 1000,
        timeout: int = 60,
        retry_interval: int = 10,
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.invoke(messages=messages).content
