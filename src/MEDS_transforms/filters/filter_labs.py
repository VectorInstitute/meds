#!/usr/bin/env python
from collections.abc import Callable
from functools import partial

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over


def filter_nan_values(df: pl.LazyFrame) -> pl.LazyFrame:
    """Create a function that filters out rows with NaN values."""
    return df.filter(~(pl.col("code").cast(pl.Utf8).str.contains("LAB//") & pl.col("numeric_value").is_nan()))


def filter_labs_fntr(stage_cfg: DictConfig) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Return a function that filters out rows with NaN values."""
    compute_fns = []
    if stage_cfg.filter_nan_values:
        logger.info("Filtering rows with NaN values.")
        compute_fns.append(partial(filter_nan_values))

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

    map_over(cfg, compute_fn=filter_labs_fntr)


if __name__ == "__main__":  # pragma: no cover
    main()
