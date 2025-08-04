# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from __future__ import annotations
from typing import Union
import copy
from pydantic.v1 import BaseModel
from langchain_core.messages import AIMessage


class DummyLangchainModel:

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        constant_return_str: str | None = None,
    ) -> None:
        self.structured_output_schema = None
        self.structured_output_schema_dict = None
        if constant_return_str is None:
            self.constant_return_str = "dummy_response_for_testing"
        else:
            self.constant_return_str = constant_return_str

    def with_structured_output(
        self, schema: Union[BaseModel, dict]
    ) -> DummyLangchainModel:
        self_copy = copy.deepcopy(self)
        if isinstance(schema, dict):
            self_copy.structured_output_schema_dict = schema
        elif isinstance(schema, type(BaseModel)):
            self_copy.structured_output_schema = schema
            self_copy.structured_output_schema_dict = schema.schema()
        else:
            raise ValueError(f"Couldn't parse structured output format: {schema}")
        return self_copy

    def _get_dummy_structured_output(self) -> Union[BaseModel, dict]:

        kwargs = {}
        args = self.structured_output_schema_dict["properties"]
        for arg_name, arg_cfg in args.items():
            if arg_cfg["type"] == "string":
                if "enum" in arg_cfg:
                    kwargs[arg_name] = arg_cfg["enum"][0]
                else:
                    kwargs[arg_name] = "dummy_test_string"
            elif arg_cfg["type"] == "boolean":
                kwargs[arg_name] = True
            elif arg_cfg["type"] == "array":
                kwargs[arg_name] = [""]
            elif arg_cfg["type"] == "integer":
                kwargs[arg_name] = 0
            else:
                raise ValueError(
                    f"Could not parse arg '{arg_name}' with config: {arg_cfg}"
                )

        # follows same output logic as langchain structured output model
        # returns Schema(**kwargs) if schema is a BaseModel,
        # and a dict otherwise
        if self.structured_output_schema is not None:
            return self.structured_output_schema(**kwargs)
        else:
            return kwargs

    def invoke(self, messages: Union[list, str]) -> Union[BaseModel, dict, AIMessage]:
        if self.structured_output_schema_dict is not None:
            return self._get_dummy_structured_output()
        else:
            return AIMessage(self.constant_return_str)
