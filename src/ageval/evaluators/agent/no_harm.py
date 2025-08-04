# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from typing import Tuple
from pydantic.v1 import BaseModel, Field
from loguru import logger

from ageval.constants import DEFAULT_OPENAI_MODEL
from ageval.evaluators.core import PairwiseEvaluator
from ageval.evaluators.utils import get_evaluator_cls_from_str


class NoHarmAgentEvaluator(PairwiseEvaluator):

    def __init__(
        self,
        baseline_cls: str = "ageval.evaluators.basic.BasicPairwiseEvaluator",
        baseline_kwargs: dict | None = None,
        agent_cls: str = "ageval.evaluators.agent.PairwiseEvaluatorAgent",
        agent_kwargs: dict | None = None,
        agent_act_threshold: int = 5,
        model: str = DEFAULT_OPENAI_MODEL,
        temperature: float = 0,
        max_tokens: int = 256,
        additional_instructions: str = None,
        model_kwargs: dict | None = None,
        **kwargs,
    ) -> None:

        baseline_kwargs = {"model": model, **(baseline_kwargs or {})}
        agent_kwargs = {"model": model, **(agent_kwargs or {})}

        self.baseline: PairwiseEvaluator = get_evaluator_cls_from_str(baseline_cls)(
            **baseline_kwargs
        )
        self.agent: PairwiseEvaluator = get_evaluator_cls_from_str(agent_cls)(
            **agent_kwargs
        )
        self.agent_act_threshold = agent_act_threshold

        if (
            self.baseline.model.model_name != model
            or self.agent.model.model_name != model
        ):
            logger.warning(
                (
                    "Using different API models between baseline and agent in NoHarmAgent. "
                    f"This may be unintentional. Models used {self.baseline.model.model_name}"
                    f" (baseline) and {self.agent.model.model_name} (agent), and {model} (NoHarmAgent)."
                )
            )

        super().__init__(
            model,
            temperature,
            max_tokens,
            additional_instructions,
            model_kwargs,
            **kwargs,
        )

    def _predict_no_harm_with_agent(
        self, text_a: str, text_b: str, prompt: str
    ) -> Tuple[bool, int]:
        class Assessment(BaseModel):
            reasoning: str = Field(
                description="A short justification for your assessment."
            )
            confidence_websearch_will_help: int = Field(
                description=self.prompts.no_harm_agent.will_websearch_help_description
            )

        lc_model = self.model.langchain_model
        struct_model = lc_model.with_structured_output(Assessment)
        struct_prompt = (
            f"Consider the following two texts, your task is to select the better text "
            f"\n\n### text_a: {text_a}"
            f"\n\n### text_b: {text_b}"
        )
        if prompt is not None:
            struct_prompt += (
                f"\nBoth texts were a response to the following context: {prompt}"
            )
        struct_prompt += "First return an initial assement to determine the next steps."
        pred_result = dict(struct_model.invoke(struct_prompt))
        score = int(pred_result["confidence_websearch_will_help"])

        return score >= self.agent_act_threshold, pred_result

    def evaluate(self, text_a: str, text_b: str, prompt: str | None = None) -> dict:

        pair_kwargs = dict(text_a=text_a, text_b=text_b, prompt=prompt)

        pred_no_harm_with_agent, pred_result = self._predict_no_harm_with_agent(
            **pair_kwargs
        )

        if pred_no_harm_with_agent:
            return_dict = self.agent.evaluate(**pair_kwargs)
        else:
            logger.debug(
                f"Noharm agent running baseline {self.baseline} with args {pair_kwargs}"
            )
            return_dict = self.baseline.evaluate(**pair_kwargs)

        return_dict["NoHarmAgentEvaluator"] = {
            "predict_no_harm_with_agent": {
                "decision": pred_no_harm_with_agent,
                "raw_output": pred_result,
            }
        }

        return return_dict
