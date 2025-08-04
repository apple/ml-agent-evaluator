"""
For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.

A tool for fact checking.

Relevant docs: https://python.langchain.com/v0.2/docs/how_to/structured_output/#the-with_structured_output-method
"""

import os
from typing import Type
from loguru import logger
from pydantic.v1 import BaseModel, Field
from langchain_core.language_models import BaseChatModel

import ageval.models
import ageval.external.web_search
from ageval.evaluators.agent.tools.base import ToolBase
from ageval.evaluators.agent.tools.registry import register_tool
from ageval.experiments.config import PromptConfig


@register_tool("fact_check")
class ToolFact(ToolBase):
    @classmethod
    def create_run_condition(cls, prompts: PromptConfig) -> Type[BaseModel]:
        class TextAssessment(BaseModel):
            """Assessment of a text."""

            contains_facts: bool = Field(
                description=prompts.tools.fact_check.contains_facts_desc
            )
            is_like_wiki: bool = Field(
                description=prompts.tools.fact_check.is_like_wiki_desc,  # check if long-form factual text
            )
            is_maths: bool = Field(
                description=prompts.tools.fact_check.is_maths_desc,
            )
            is_wordcount: bool = Field(
                description=prompts.tools.fact_check.is_word_count_desc
            )
            confidence_websearch_will_help: int = Field(
                description=prompts.tools.fact_check.confidence_web_helps_desc
            )

        return TextAssessment

    @classmethod
    def should_run(cls, *, condition_data: dict) -> bool:

        # check for basics initially
        # that the response contains facts,
        # and that it's not a maths or word count problem
        if (
            condition_data["contains_facts"]
            and not condition_data["is_maths"]
            and not condition_data["is_wordcount"]
        ):
            if condition_data["is_like_wiki"]:
                # if long-form factual text like from a wiki, always run
                return True
            elif int(condition_data["confidence_websearch_will_help"]) >= 5:
                # otherwise only run if confidence websearch will help is very high
                return True

        return False

    @classmethod
    def run(cls, *, text: str, prompt: str, model_name: str) -> dict:
        return check(text=text, prompt=prompt, model_name=model_name)


def make_list_to_str(ls: list):
    return "\n  - " + "\n  - ".join(ls) + "\n"


def check(
    text: str,
    prompt: str,
    model_name: str,
    use_web_search: bool = True,
) -> dict:
    """Check factuality of text.

    Returns a dictionary containing information about the factuality of the given text.
    """

    lc_model = ageval.models.get_model(model_name).langchain_model

    logger.debug("Starting fact checking...")

    logger.debug("Step 1: generating atomic facts")
    atomic_facts = create_atomic_facts(text=text, model=lc_model)
    logger.debug(f"Atomic facts generated:{make_list_to_str(atomic_facts)}")

    logger.debug("Step 2: making facts self-contained")
    processed_facts = [
        {
            "initial": fact,
            "contained": make_fact_self_contained(
                fact=fact, context=text, prompt=prompt, model=lc_model
            ),
        }
        for fact in atomic_facts
    ]
    self_contained_fact_str_list = [
        f'{fact["initial"] :-<50}→ {fact["contained"]}' for fact in processed_facts
    ]
    logger.debug(
        f"Self-contained facts:{make_list_to_str(self_contained_fact_str_list)}"
    )

    final_fact_str_list = []
    for fact in processed_facts:
        fact["result"] = check_fact_supported(
            fact=fact["contained"],
            model=lc_model,
            use_web_search=use_web_search,
        )
        if fact["result"]["truthful"]:
            final_fact_str_list.append("✅ " + fact["contained"])
        else:
            final_fact_str_list.append("❌ " + fact["contained"])

    logger.debug(
        f"Final truthfulness assessment:{make_list_to_str(final_fact_str_list)}"
    )

    return processed_facts


def create_atomic_facts(text: str, model: BaseChatModel) -> list[str]:

    class AtomicFacts(BaseModel):
        """List of individual atomic facts that can be checked with a web search."""

        atomic_facts: list[str] = Field(
            description="A list of separate individual facts."
        )

    struct_model = model.with_structured_output(AtomicFacts)
    return struct_model.invoke(
        (
            f"Break down the following statement into separate individual facts:\n\n{text}"
            "\n  Ignore things that cannot be verified in a web search."
        )
    ).atomic_facts


def make_fact_self_contained(
    fact: str, context: str, prompt: str, model: BaseChatModel
):
    class SelfContainedFact(BaseModel):
        """A self contained fact."""

        self_contained_fact: str = Field(
            description="A self-contained fact that does not require external information to be understood. Do not add additional information that is not needed."
        )

    struct_model = model.with_structured_output(SelfContainedFact)
    struct_prompt = ""
    if prompt is not None:
        struct_prompt += f"We have a response text for the following prior conversation:\n{prompt}\n\n"
    struct_prompt += (
        "You are given the following response "
        f"context:\n\n{context}\n\nUse this context to make the following statement "
        f"self-contained (if necessary, otherwise return unchanged):{fact}"
    )
    return struct_model.invoke(struct_prompt).self_contained_fact


def check_fact_supported(
    fact: str, model: BaseChatModel, use_web_search: bool = True
) -> dict:

    class FactCheckingResult(BaseModel):
        """A self contained fact."""

        reasoning: str = Field(
            description="A short justification for the truthfulness verdict. Max three sentences."
        )
        truthful: bool = Field(
            description="Whether or not the fact is truthful. Must be true or false."
        )

    struct_model = model.with_structured_output(FactCheckingResult)
    if not use_web_search:
        prompt = f"Is the following statement truthful: {fact}"
    else:
        web_search_results = get_information_from_web_searches(fact=fact, model=model)
        prompt = (
            f"You have the following statement: {fact}\n"
            "\nYou also have the following web search results:"
            f"\n```\n{web_search_results}\n```"
            "Is the truthfulness of the statement supported by these search results? "
            "Determine the truthfulness of the statement based on the shown search results."
        )

    result = struct_model.invoke(prompt)

    return dict(result)


def get_information_from_web_searches(
    fact: str,
    model: BaseChatModel,
    max_num_searches: int = 3,
    max_num_results_per_search: int = 5,
    enable_skipping_search: bool = True,
) -> str:

    class NextSearchQuery(BaseModel):
        """Query to search the web via search engine."""

        query: str = Field(description="Search query string.")
        skip_further_searches: bool = Field(
            description="Whether to skip further searches because enough information available."
        )

    struct_model = model.with_structured_output(NextSearchQuery)

    prev_search_queries = []
    results = []

    for i in range(max_num_searches):
        prompt = (
            f"We have the following statement: {fact}\n"
            "Come up with a search query for a web search engine that may help "
            "check the truthfulness of this statement."
        )
        if prev_search_queries:
            prompt += "Please create new search query different from previous queries:"
            prompt += f"\n```\n{results}\n```"
            if enable_skipping_search:
                prompt += (
                    "\nIf enough information is available to evaluate the truthfulness of the "
                    "statement, STOP further searches by returning a json with a True skip_further_searches value."
                )

        logger.debug(prompt)

        next_query_decision: NextSearchQuery = struct_model.invoke(prompt)
        query = next_query_decision.query
        prev_search_queries.append(query)

        if (
            enable_skipping_search
            and next_query_decision.skip_further_searches
            and i > 0
        ):
            logger.debug(
                f"Skipping further searches (completed {min(1,i)} out of {max_num_searches})"
            )
            break

        logger.debug(f"Searching for query '{query}'")

        result = run_web_search(
            search_query=query,
            max_num_results_per_search=max_num_results_per_search,
        )
        results.append({"query": query, "results": result})

    return results


def run_web_search(
    search_query: str,
    max_num_results_per_search: int = 5,
):

    web_search_agent = ageval.internal.web_search.WebSearch(
        api_key=os.environ["RAPID_API_KEY"],
        limit=max_num_results_per_search,
        passage_selector_type=None,
    )

    return web_search_agent.rapidapi_web_search(search_query)["data"]
