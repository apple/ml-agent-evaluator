"""
For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.

Agentic evaluator.
"""

from typing import Literal
from loguru import logger
from ageval.constants import DEFAULT_OPENAI_MODEL
from ageval.evaluators.core import PairwiseEvaluator
from ageval.evaluators.utils import get_evaluator_cls_from_str
from ageval.evaluators.agent.tools import registry
from ageval.evaluators.agent.tools.base import ToolBase
import ageval.evaluators.agent.tools.fact_checking  # ensures fact checking tool registered
import ageval.evaluators.agent.tools.code_interpreter  # import code_interpreter tool to register
import ageval.evaluators.agent.tools.math_checker  # import math_checker tool to register
from pydantic.v1 import BaseModel, Field


class PairwiseEvaluatorAgent(PairwiseEvaluator):

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_MODEL,
        max_tokens: int = 1024,
        tools: list[str] | None = None,
        tools_to_always_run: list[str] | None = None,
        baseline: (
            str | PairwiseEvaluator | None
        ) = "ageval.evaluators.basic.BasicPairwiseEvaluator",
        baseline_kwargs: dict | None = None,
        prompt_type: Literal["arena_hard"] | None = None,
        **kwargs,
    ) -> None:
        """Pairwise evaluator agent.

        Unless otherwise specified in baseline kwargs, the baseline LLM model is the same as the main agent model.

        Args:
            model (str, optional): Name of model to use inside evaluator. Defaults to DEFAULT_OPENAI_MODEL.
            max_tokens (int, optional): Maximum number of output tokens. Defaults to 1024.
            tools (list[str] | None, optional): List of tool names to make available to annotator. Defaults to None.
            tools_to_always_run (list[str] | None, optional): List of tools that should always run,
                independent of the run condition of the tool. Defaults to None.
            baseline (str  |  PairwiseEvaluator, optional): Baseline evaluator to fall back to if no tools applicable.
                Defaults to "ageval.evaluators.basic.BasicPairwiseEvaluator". Can be passed either as already instantiated
                PairwiseEvaluator or as string with class path.
            baseline_kwargs (dict | None, optional): Only works if baseline is passed as string.
                Then, this will be used as arguments when constructing the evaluator. Defaults to None.
        """

        if tools is None:
            tools = ["fact_check", "code_interpreter", "math_checker"]
        if tools_to_always_run is None:
            tools_to_always_run = []

        # Note: omegaconf does not currently support sets,
        # thus passing tools as list to this class
        self.tools = registry.get_tools(enabled_tools=set(tools))
        self.tools_to_always_run = tools_to_always_run
        self.baseline = self._get_baseline(
            baseline=baseline,
            baseline_kwargs=baseline_kwargs,
            model=model,
        )
        self.prompt_type = prompt_type

        super().__init__(
            model=model,
            max_tokens=max_tokens,
            **kwargs,
        )

    def _get_baseline(
        self,
        baseline: str | PairwiseEvaluator,
        baseline_kwargs: dict | None,
        model: str,
    ) -> None:
        kwargs = {"model": model}
        kwargs.update(baseline_kwargs or {})

        if isinstance(baseline, str) and baseline != "None":
            return get_evaluator_cls_from_str(baseline)(**kwargs)
        else:
            return baseline

    def _initial_assessment(
        self, text: str, *, tools: dict[str, type[ToolBase]], prompt: str | None = None
    ) -> dict:
        """Run initial assessment of a single response to determine further steps.

        Args:
            text (str): Text of response to assess.
            prompt (str | None, optional): Corresponding prompt that response text replies to. Defaults to None.

        Returns:
            dict: a dictionary containing a the initial assessment
        """

        final_schema = None
        for tool in tools.values():
            schema = tool.create_run_condition(prompts=self.prompts).schema()
            if final_schema is None:
                final_schema = schema
            else:
                final_schema["properties"].update(schema["properties"])
                final_schema["required"] += schema["required"]

        lc_model = self.model.langchain_model
        struct_model = lc_model.with_structured_output(final_schema)
        struct_prompt = f"Assess the following text: {text}"
        if prompt is not None:
            struct_prompt += (
                f"\nThe text is a response to the following context: {prompt}"
            )
        return dict(struct_model.invoke(struct_prompt))

    def _final_assessment(self, summary: dict) -> dict:
        """Make final assessment of which of two responses is preferred based on prior assessments.

        Args:
            summary (dict):dict containing summary of tool/subroutine assessments as well as
                basic information like prompt and response texts.

        Returns:
            dict: final decision and reasoning as well as prior assessment results
        """

        # this class defines the output of the evaluation
        class EvaluationResult(BaseModel):
            reasoning: str = Field(
                description="A short justification for selecting one text over the other."
            )
            selected_text: Literal["text_a", "text_b"] = Field(
                description="Selected text that is better than the other text. Must be one of the following two strings: 'text_a' or 'text_b'. Do not set as the selected text string itself."
            )

        lc_model = self.model.langchain_model
        struct_model = lc_model.with_structured_output(EvaluationResult)

        prompt = summary["prompt"]
        if self.prompt_type is None:
            struct_prompt = (
                f"Compare the following two texts and select the better text "
                "according to the information provided:"
                f"\n\n### text_a: {summary['text_a']['text']}"
                f"\n\n### text_b: {summary['text_b']['text']}"
                f"\nThe following tool output should also be taken into account:"
                f"\n\n### tool_output for text_a: {summary['text_a'].get('tool_output', {})}"
                f"\n\n### tool_output for text_b: {summary['text_b'].get('tool_output', {})}"
            )

            if prompt is not None:
                struct_prompt += (
                    f"\nBoth texts were a response to the following context: {prompt}"
                )
        elif self.prompt_type == "arena_hard":
            user_content = "<|User Prompt|>\n{question_1}\n\n<|The Start of Assistant A's Answer|>\n{answer_1}\n<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\n{answer_2}\n<|The End of Assistant B's Answer|>".format(
                question_1=prompt,
                answer_1=summary["text_a"]["text"],
                answer_2=summary["text_b"]["text"],
            )
            user_content += f"\n\n<|The Start of Tool Output on Assistant A's Answer|>\n{summary['text_a'].get('tool_output', {})}\n<|The End of Tool Output on Assistant A's Answer|>\n\n"
            user_content += f"<|The Start of Tool Output on Assistant B's Answer|>\n{summary['text_b'].get('tool_output', {})}\n<|The End of Tool Output on Assistant B's Answer|>"
            messages = [
                {
                    "role": "system",
                    "content": "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\nBegin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.\n\nWhen evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.\n\nThen consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\nThen consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.\n\nAfter providing your explanation, you must output only one of the following choices as your final verdict with a label:\n\n1. Assistant A is significantly better: [[A>>B]]\n2. Assistant A is slightly better: [[A>B]]\n\n3. Assistant B is slightly better: [[B>A]]\n4. Assistant B is significantly better: [[B>>A]]\n\nExample output: \"My final verdict is Assistant B is slightly better: [[B>A]]\". Note that ties are not allowed, always make a decision.",
                },
                {"role": "user", "content": user_content},
            ]
            struct_prompt = messages
        else:
            raise ValueError(f"Unknown prompt type: {self.prompt_type}!")

        evaluation_result = dict(struct_model.invoke(struct_prompt))
        summary["overall_result"] = evaluation_result
        summary["preferred_text"] = evaluation_result["selected_text"]

        return summary

    def evaluate(self, text_a: str, text_b: str, prompt: str | None = None) -> dict:
        """Judge two responses for a given prompt.

        This is the main function of the evaluator, allowing the judgement of two
        responses (`text_a` and `text_b`).

        Args:
            text_a (str): First response to judge
            text_b (str): Second response to judge
            prompt (str | None, optional): Prompt that response reply to.
                Defaults to None.

        Returns:
            dict: judgement results, including information of intermediate steps
                and reasoning
        """
        summary = {"prompt": prompt, "errors": []}

        for text_name, text in [("text_a", text_a), ("text_b", text_b)]:

            summary[text_name] = {"text": text}

            # run initial assessment
            summary[text_name]["initial_assessment"] = self._initial_assessment(
                text=text,
                prompt=prompt,
                tools=self.tools,
            )

            # depending on initial assessment, run fact-check subroutine/tool
            tool_output = {}
            num_tools_used = 0
            tools_run = []
            tools_forced_to_run = []
            for tool_name, tool in self.tools.items():
                if not tool.should_run(
                    condition_data=summary[text_name]["initial_assessment"]
                ):
                    # check if tool if not forced to run
                    if tool_name not in self.tools_to_always_run:
                        continue
                    else:
                        logger.debug(
                            f"Note: agent running tool `{tool_name}` only because forced to run."
                        )
                        tools_forced_to_run.append(tool_name)

                try:
                    logger.debug(f"Agent running tool `{tool_name}`.")
                    tool_output[tool_name] = tool.run(
                        text=text, prompt=prompt, model_name=self.model.model_name
                    )
                    tools_run.append(tool_name)
                    num_tools_used += 1  # only add to num tools used if fully ran
                except Exception as e:
                    logger.debug(f"Tool error for `{tool_name}`: {e}")
                    tool_output[tool_name] = (
                        "An error occured during executing this test"
                    )
                    summary["errors"].append(
                        f"Error whilst executing tool '{tool_name}' on '{text_name}': {repr(e)}"
                    )
            summary[text_name]["tool_output"] = tool_output

        if num_tools_used > 0 or self.baseline is None:
            # run final assessment based on tool outputs
            final_result_summary = self._final_assessment(summary)
            final_result_summary["reverted_to_baseline"] = False

            # log whether the initial assessment recommended reverting to baseline
            if len(tools_forced_to_run) == num_tools_used or self.baseline is None:
                final_result_summary["initial_assessment_recommended_baseline"] = True
            else:
                final_result_summary["initial_assessment_recommended_baseline"] = False
        else:
            # run baseline evaluator if no tools were actually used
            final_result_summary = self.baseline.evaluate(
                text_a=text_a, text_b=text_b, prompt=prompt
            )
            final_result_summary["reverted_to_baseline"] = True
            final_result_summary["initial_assessment_recommended_baseline"] = True
            final_result_summary["agent_intermediate_results"] = summary

        final_result_summary["tools_forced_to_run"] = tools_forced_to_run
        final_result_summary["tools_run"] = tools_run
        final_result_summary["num_tools_used"] = num_tools_used

        logger.debug(
            f'Agent final assessment: preferred text is `{final_result_summary["preferred_text"]}`.'
        )

        return final_result_summary
