#!/usr/bin/env python
from collections.abc import Callable
from functools import partial

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over


def filter_codes_by_name(df: pl.LazyFrame, code_name: str) -> pl.LazyFrame:
    """Create a function that removes rows where the code column value starts with a certain code_name."""
    df_filtered = df.filter(~pl.col("code").cast(pl.Utf8).str.starts_with(code_name))
    return df_filtered


def filter_codes_fntr(stage_cfg: DictConfig) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Return a function that removes rows where the code column value starts with a certain code_name."""
    compute_fns = []
    codes_to_filter = stage_cfg.codes_to_filter
    if isinstance(codes_to_filter, str):
        codes_to_filter = [codes_to_filter]
    if codes_to_filter:
        for code in codes_to_filter:
            logger.info(f"Filtering rows with code {code}")
            compute_fns.append(partial(filter_codes_by_name, code_name=code))

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

    map_over(cfg, compute_fn=filter_codes_fntr)


if __name__ == "__main__":  # pragma: no cover
    main()
