"""Core utilities for MEDS pipelines built with these tools."""

import os

import hydra
from loguru import logger as log


def hydra_loguru_init() -> None:
    """Adds loguru output to the logs that hydra scrapes.

    Must be called from a hydra main!
    """
    hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.add(os.path.join(hydra_path, "main.log"))