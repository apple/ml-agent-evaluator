"""
For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.

Implementation of Search-Augmented Factuality Evaluator (SAFE).

Some of the code in this file is adapted from:
https://github.com/google-deepmind/long-form-factuality
Copyright (c) 2023 Sewon Min.
Licensed under the MIT License.

Paper: https://arxiv.org/abs/2403.18802

Major changes over the original implementation:
- Using RapidAPI google search instead of SerperAPI. 
  Note that this change may lead to worse performance,
  as SerperAPI includes additional information, such as
  knowledge graph information, beyond standard search
  results.
"""

import pathlib
from loguru import logger

from ageval.constants import DEFAULT_OPENAI_MODEL
import ageval.evaluators.core
import ageval.utils

# patch if new openai version installed

import openai

if openai.version.VERSION.startswith("1."):
    logger.warning(
        f"Using more recent OpenAI version ({openai.version.VERSION}). Adding additional compat patches."
    )
    openai.error = None
    openai.openai_object = None

import third_party.long_form_factuality.eval.safe.get_atomic_facts
import third_party.long_form_factuality.eval.safe.search_augmented_factuality_eval
import third_party.long_form_factuality.common.modeling as modeling


def patch_atomic_fact_generation(
    demon_dir="third_party/long_form_factuality/third_party/factscore/demos/",
):
    """Fixes and patches to make long form factuality code run for us."""
    import ageval.utils
    import common.shared_config
    import os

    # Patch 1: make sure openai api key is available
    ageval.utils.setup_api_keys()
    common.shared_config.openai_api_key = os.environ["OPENAI_API_KEY"]

    # Patch 2: make sure demonstrations path is correct
    import third_party.factscore.atomic_facts

    if not pathlib.Path(demon_dir).exists():
        new_demon_dir = "../" + demon_dir
        if pathlib.Path(new_demon_dir).exists():
            demon_dir = new_demon_dir
        else:
            logger.warning(
                "Could not find the SAFE example "
                "demonstrations. This may prevent SAFE algorithm from running."
                "Run patch_atomic_fact_generation with the correct demon_dir path to fix this. "
                "Typically the location is "
                "'third_party/long_form_factuality/third_party/factscore/demos/'."
            )

    class_to_change = third_party.factscore.atomic_facts.AtomicFactGenerator
    original_init = class_to_change.__init__

    def patched_init(self, demon_dir=demon_dir, *args, **kwargs):
        original_init(self, demon_dir=demon_dir, *args, **kwargs)

    class_to_change.__init__ = patched_init


def patch_error_messages():
    from common import utils

    def new_print_error(
        message: str | Exception | None,
        additional_info: str = "",
        verbose: bool = False,
    ) -> None:
        print(message)

    utils.maybe_print_error = new_print_error


def patch_replace_serper_with_rapid():
    import os
    from copy import copy
    import ageval.external.web_search
    import ageval.utils
    import eval.safe.query_serper as query_serper

    ageval.utils.setup_api_keys()

    NUM_RESULTS_REQUESTED = 5

    original_serper_api = copy(query_serper.SerperAPI)

    class RapidSearchAPI:

        def __init__(self, *args, **kwargs) -> None:
            self.web_search_agent = ageval.external.web_search.WebSearch(
                api_key=os.environ.get("RAPID_API_KEY"),
                limit=NUM_RESULTS_REQUESTED,
                passage_selector_type=None,
            )

        def run(self, query: str, **kwargs):
            logger.info(f"Doing a web search with query {query}.")
            serper_style_result = {}
            serper_style_result["organic"] = self.web_search_agent.rapidapi_web_search(
                query
            )["data"]
            parsed_result = original_serper_api(
                "norealkey", k=NUM_RESULTS_REQUESTED
            )._parse_results(serper_style_result)
            return parsed_result

    if "RapidSearchAPI" not in str(query_serper.SerperAPI):
        logger.info("Applying SerperAPI to RapidAPI patch.")
        query_serper.SerperAPI = RapidSearchAPI
    else:
        logger.info("RapidAPI patch already applied, skipping.")
        pass


def apply_all_patches():
    logger.info("Applying patches to original SAFE implementation.")
    patch_atomic_fact_generation()
    patch_error_messages()
    patch_replace_serper_with_rapid()
    logger.info("Patches successfully applied.")


def compute_metrics(rating_result: dict, at_values: list = [64, 178]) -> dict:
    metrics = {
        "Supported": rating_result["Supported"],
        "Not Supported": rating_result["Not Supported"],
        "Irrelevant": rating_result["Irrelevant"],
    }

    if not (metrics["Supported"] == 0 and metrics["Not Supported"] == 0):
        metrics["Precision"] = metrics["Supported"] / (
            metrics["Supported"] + metrics["Not Supported"]
        )

        def add_at_metrics(metrics: dict, num: int):
            metrics[f"Recall{num}"] = min(metrics["Supported"] / num, 1)
            if metrics["Supported"] > 0:
                metrics[f"F@{num}"] = (
                    2 * metrics["Precision"] * metrics[f"Recall{num}"]
                ) / (metrics["Precision"] + metrics[f"Recall{num}"])
            else:
                metrics[f"F@{num}"] = 0
            return metrics

        for value in at_values:
            metrics = add_at_metrics(metrics, num=value)
    else:
        error_msg = "Neither supported nor not supported facts found. No facts to begin with or all irrelevant."
        metrics["error"] = error_msg
        logger.error(error_msg)

    return metrics


def algorithm(
    prompt: str,
    response: str,
    algorithm_model: str,
    max_facts: int,
):
    """SAFE algorithm."""

    if not algorithm_model.startswith("apple"):
        lff_model = modeling.Model(
            model_name=algorithm_model.replace("/", ":"),
            temperature=0.0,
        )
        logger.info(f"Running SAFE with model: {algorithm_model}")
    else:
        import ageval.internal.endpoint_model

        model_name = algorithm_model.split("/")[1]
        lff_model = ageval.internal.endpoint_model.InternalModel(
            model_name=model_name,
        )

    logger.info("SAFE: Starting SAFE algorithm.")
    # Step 1: Split response into individual facts
    atomic_facts = third_party.long_form_factuality.eval.safe.get_atomic_facts.main(
        response=response,
        model=lff_model,
    )
    facts_generated = [
        fact
        for sentence in atomic_facts["all_atomic_facts"]
        for fact in sentence["atomic_facts"]
    ]

    num_facts_to_parse = 0
    facts_to_parse = []
    all_atomic_facts_to_parse = []

    if max_facts is None:
        num_facts_to_parse = len(facts_generated)
        all_atomic_facts_to_parse = atomic_facts["all_atomic_facts"]
    else:
        for sentence in atomic_facts["all_atomic_facts"]:
            additional_facts = sentence["atomic_facts"]
            if num_facts_to_parse + len(additional_facts) > max_facts:
                sentence["atomic_facts"] = sentence["atomic_facts"][
                    : max_facts - num_facts_to_parse
                ]
                all_atomic_facts_to_parse.append(sentence)
                facts_to_parse += sentence["atomic_facts"]
                num_facts_to_parse = max_facts
                break
            else:
                all_atomic_facts_to_parse.append(sentence)
                facts_to_parse += sentence["atomic_facts"]
                num_facts_to_parse += len(sentence["atomic_facts"])

    logger.info(
        f"SAFE: Step 1 completed, atomic facts extracted. Number of facts extracted: {len(facts_generated)}"
    )

    logger.info(
        "Facts to be revised and checked:\n  - " + "\n  - ".join(facts_generated)
    )
    logger.info(
        f"SAFE: Number of facts that will actually be evaluated: first {num_facts_to_parse} out of {len(facts_generated)}."
    )

    logger.error(facts_to_parse)

    # Steps 2-4: (2) first revise facts to be self-contained, then (3) determine relevance,
    # and then finally (4) rate whether the facts are supported by google search results
    logger.info("SAFE: Completing remaining steps.")
    rating_result = third_party.long_form_factuality.eval.safe.search_augmented_factuality_eval.classify_relevance_and_rate(
        prompt=prompt,
        response=response,
        sentences_and_atomic_facts=all_atomic_facts_to_parse,
        rater=lff_model,
    )
    logger.info("SAFE: Remaining steps completed, returning results.")

    return {
        "prompt": prompt,
        "response": response,
        "metrics": compute_metrics(rating_result),
        "meta": {
            "num_facts_generated": len(facts_generated),
            "num_facts_evaluated": num_facts_to_parse,
            "facts": facts_generated,
        },
        **atomic_facts,
        **rating_result,
    }


class SingleEvaluator(ageval.evaluators.core.SingleEvaluator):

    def __init__(
        self,
        model: str,
        max_facts: int,
        max_tokens: int,
    ) -> None:
        self.model_name = model
        self.max_facts = max_facts
        apply_all_patches()
        super().__init__(model=model, max_tokens=max_tokens)

    def evaluate(self, response: str, prompt: str) -> dict:
        try:
            results = algorithm(
                prompt=prompt,
                response=response,
                algorithm_model=self.model_name,
                max_facts=self.max_facts,
            )
            if "Precision" in results["metrics"].keys():
                score = results["metrics"]["Precision"]
                return {"score": score, "msg": [], "meta": results["meta"]}
            else:
                error_msg = "Error: evaluation yielded no precision metric."
                logger.warning(error_msg)
                return {"score": None, "msg": [error_msg], "meta": results["meta"]}
        except KeyboardInterrupt as e:
            logger.error("Evaluator script interrupted (KeyboardInterrupt)")
        except Exception as e:
            error_msg = f"Error: SAFE algorithm failed completely:\n{e}"
            logger.exception(error_msg)
            return {"score": None, "msg": [error_msg]}


class PairwiseEvaluator(ageval.evaluators.core.PairwiseFromSingleEvaluator):

    def __init__(
        self,
        model: str,
        additional_instructions: str,
        max_facts: int,
        max_tokens: int,
    ) -> None:
        super().__init__(
            evaluator=SingleEvaluator(
                model=model, max_facts=max_facts, max_tokens=max_tokens
            ),
            larger_is_better=True,
            additional_instructions=additional_instructions,
        )
