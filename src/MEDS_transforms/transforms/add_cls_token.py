#!/usr/bin/env python
from collections.abc import Callable

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over


def add_new_events_fntr(fn: Callable[[pl.DataFrame], pl.DataFrame]) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Creates a "meta" functor that computes the input functor on a MEDS shard then combines both dataframes.

    Args:
        fn: The function that computes the new events.

    Returns:
        A function that computes the new events and combines them with the original DataFrame, returning a
        result in proper MEDS sorted order.
    """

    def out_fn(df: pl.DataFrame) -> pl.DataFrame:
        new_events = fn(df)

        df = df.with_row_count("__idx")
        new_events = new_events.with_columns(pl.lit(0).alias("__idx").cast(pl.UInt32))
        return (
            pl.concat([df, new_events], how="diagonal").sort(by=["subject_id", "time", "__idx"]).drop("__idx")
        )

    return out_fn


def cls_fntr(cfg: DictConfig) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Create a function that adds "CLS" token for every patient."""

    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        min_time_expr = pl.col("time").min().over("subject_id")
        cls_df = df.unique(subset=["subject_id"]).with_columns(
            [
                pl.lit(cfg.cls_code).alias("code"),
                (min_time_expr - pl.duration(seconds=1)).alias("time"),
            ]
        )
        return cls_df

    return fn


def add_cls_token_fntr(stage_cfg: DictConfig) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Return a function that adds a "CLS" token for every patient."""
    compute_fns = []
    logger.info("Adding CLS token")
    compute_fns.append(add_new_events_fntr(cls_fntr(stage_cfg)))

    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        for compute_fn in compute_fns:
            df = compute_fn(df)
        return df

    return fn


@hydra.main(
    version_base=None, config_path=str(PREPROCESS_CONFIG_YAML.parent), config_name=PREPROCESS_CONFIG_YAML.stem
)
def main(cfg: DictConfig):
    """Adds time-derived measurements to a MEDS cohort as separate observations at each unique time."""

    map_over(cfg, compute_fn=add_cls_token_fntr)


if __name__ == "__main__":  # pragma: no cover
    main()
