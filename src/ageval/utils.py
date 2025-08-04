# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from dotenv import load_dotenv
import pathlib
from loguru import logger
import sys
import os
import pathlib

DEFAULT_DOTENV_PATH = pathlib.Path("ageval_secrets.toml").absolute()


def setup_api_keys(path: str = None):
    """Tries to set up API keys based on local secrets file.
    
    Also supports custom OpenAI API base URLs via OPENAI_API_BASE variable.
    """

    if path is None:
        path = DEFAULT_DOTENV_PATH
    else:
        path = pathlib.Path(path)

    if load_dotenv(
        path,
        verbose=True,
    ):
        logger.debug(f"Loaded API keys from '{path}'.")
        
        # If OPENAI_API_BASE is set in the file, ensure it's in environment
        if "OPENAI_API_BASE" in os.environ:
            logger.debug(f"Using custom OpenAI API base: {os.environ['OPENAI_API_BASE']}")
    else:
        logger.warning(
            f"No secrets file found at '{path}' or no variable set inside file. "
            "Make sure API keys are set in a different way to continue."
        )


def setup_logger(level: str = "INFO"):
    """Make logger look good"""
    log_str = "<red>>>></red> {level: <4} | <level>{message}</level>"
    logger.remove()
    logger.level("INFO", color="")
    logger.add(
        sys.stdout,
        colorize=True,
        format=log_str,
        level=level,
        enqueue=True,
    )
    try:
        logger.level("INTERNAL_MODEL", no=14, color="<blue>")
    except TypeError:
        pass


def setup_experiment_log_files(dir: str | None):
    if dir is not None and dir != "None":
        logger.add(
            os.path.join(pathlib.Path(dir) / "internal_log.txt"),
            filter=lambda record: record["level"].name == "INTERNAL_MODEL",
            enqueue=True,  # for multiprocessing
        )
        logger.add(
            os.path.join(pathlib.Path(dir) / "general_log.txt"),
            enqueue=True,  # for multiprocessing
        )
