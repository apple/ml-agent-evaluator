# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from typing import Type
from pydantic.v1 import BaseModel, Field

import ageval.models
import ageval.external.web_search
from ageval.evaluators.agent.tools.base import ToolBase
from ageval.evaluators.agent.tools.registry import register_tool


@register_tool("word_count")
class ToolWordCount(ToolBase):
    @classmethod
    def create_run_condition(cls) -> Type[BaseModel]:
        class TextAssessment(BaseModel):
            """Assessment of a text."""

            is_word_count: bool = Field(
                description="Is the response attempting to count words in any form?"
            )

        return TextAssessment

    @classmethod
    def should_run(cls, *, condition_data: dict) -> bool:
        return condition_data["is_word_count"]

    @classmethod
    def run(cls, *, text: str, prompt: str, model_name: str) -> dict:
        raise NotImplementedError
