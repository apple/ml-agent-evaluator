"""
For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.

A tool for checking math via GPT4o + code interpretation.
"""

from typing import Type
from pydantic.v1 import BaseModel, Field

from ageval.evaluators.agent.tools.code_interpreter import ToolCodeInterpreter
from ageval.evaluators.agent.tools.registry import register_tool
from ageval.experiments.prompts import PromptConfig


@register_tool("math_checker")
class ToolMathChecker(ToolCodeInterpreter):

    assistant_instruction: str = (
        "You are a personal math tutor. "
        "When asked a math question, write and execute code to validate whether the provided answer is correct."
    )

    @classmethod
    def create_run_condition(cls, prompts: PromptConfig) -> Type[BaseModel]:
        class TextAssessment(BaseModel):
            """Assessment of a text."""

            math_question: bool = Field(
                description="Whether the text involves math or arithmetic that may benefit from careful checking."
            )

        return TextAssessment

    @classmethod
    def should_run(cls, *, condition_data: dict) -> bool:
        return condition_data["math_question"]
