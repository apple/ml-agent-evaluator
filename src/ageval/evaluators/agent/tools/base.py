# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from pydantic.v1 import BaseModel
from typing import Type

from ageval.experiments.config import PromptConfig


class ToolBase:
    @classmethod
    def create_run_condition(cls, prompts: PromptConfig) -> Type[BaseModel]:
        """Create a run condition, used to determine whether to run."""
        ...

    @classmethod
    def should_run(cls, *, condition_data: dict) -> bool:
        """Assess whether this tool should run."""
        ...

    @classmethod
    def run(cls, *, text: str, prompt: str, model_name: str) -> dict:
        """Run the tool."""
        raise NotImplementedError(cls)
