#!/usr/bin/env python
from collections.abc import Callable
from functools import partial

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over


def select_columns(df: pl.LazyFrame, columns_to_keep: list) -> pl.LazyFrame:
    """Create a function that selects only the specified columns from the DataFrame."""
    df_selected = df.select(columns_to_keep)
    return df_selected


def select_columns_fntr(stage_cfg: DictConfig) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Return a function that selects only the specified columns from the DataFrame."""
    compute_fns = []
    if stage_cfg.columns_to_keep:
        logger.info(f"Selecting columns in {stage_cfg.columns_to_keep}")
        compute_fns.append(partial(select_columns, columns_to_keep=stage_cfg.columns_to_keep))

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

    map_over(cfg, compute_fn=select_columns_fntr)


if __name__ == "__main__":  # pragma: no cover
    main()
