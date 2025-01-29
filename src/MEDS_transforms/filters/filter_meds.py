#!/usr/bin/env python
from collections.abc import Callable
from functools import partial

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over


def filter_medication_code(df: pl.LazyFrame, code_to_filter: str) -> pl.LazyFrame:
    """Create a function that filters out rows with specific medication codes."""
    end_code = f"//{code_to_filter}"
    return df.filter(
        ~(
            pl.col("code").cast(pl.Utf8).str.contains("MEDICATION")
            & pl.col("code").cast(pl.Utf8).str.ends_with(end_code)
        )
    )


def filter_meds_fntr(stage_cfg: DictConfig) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Return a function that filters out rows with specific medication codes."""
    compute_fns = []
    if stage_cfg.codes_to_filter:
        logger.info(f"Filtering rows with medication codes {stage_cfg.codes_to_filter}")
        if isinstance(stage_cfg.codes_to_filter, str):
            stage_cfg.codes_to_filter = [stage_cfg.codes_to_filter]
        for code in stage_cfg.codes_to_filter:
            compute_fns.append(partial(filter_medication_code, code_to_filter=code))

    def fn(data: pl.LazyFrame) -> pl.LazyFrame:
        for compute_fn in compute_fns:
            data = compute_fn(data)
        return data

    return fn


@hydra.main(
    version_base=None, config_path=str(PREPROCESS_CONFIG_YAML.parent), config_name=PREPROCESS_CONFIG_YAML.stem
)
def main(cfg: DictConfig):
    """TODO."""

    map_over(cfg, compute_fn=filter_meds_fntr)


if __name__ == "__main__":  # pragma: no cover
    main()
