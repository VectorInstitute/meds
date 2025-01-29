#!/usr/bin/env python
from collections.abc import Callable
from functools import partial

import hydra
import polars as pl
from omegaconf import DictConfig

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over


def update_discharge_code(df: pl.LazyFrame) -> pl.LazyFrame:
    """Create a function that updates the discharge code."""
    return df.with_columns(
        pl.when(pl.col("code") == "TRANSFER_TO//discharge//UNK")
        .then(pl.lit("TRANSFER_TO//discharge"))
        .otherwise(pl.col("code"))
        .alias("code")
    )


def update_ed_code(df: pl.LazyFrame) -> pl.LazyFrame:
    """Create a function that updates the emergency department code."""
    return df.with_columns(
        pl.when(pl.col("code") == "TRANSFER_TO//ED//Emergency Department")
        .then(pl.lit("TRANSFER_TO//ED"))
        .otherwise(pl.col("code"))
        .alias("code")
    )


def update_admit_code(df: pl.LazyFrame) -> pl.LazyFrame:
    """Create a function that updates the admit code."""
    return df.with_columns(
        pl.when(pl.col("code").str.starts_with("TRANSFER_TO//admit//"))
        .then(
            pl.concat_str(
                [
                    pl.lit("TRANSFER_TO//"),
                    pl.col("code")
                    .str.extract(r"TRANSFER_TO//admit//(.+)", 1)
                    .str.to_lowercase()
                    .str.replace_all(" ", "_"),
                ]
            )
        )
        .otherwise(pl.col("code"))
        .alias("code")
    )


def update_transfer_code(df: pl.LazyFrame) -> pl.LazyFrame:
    """Create a function that updates the transfer code."""
    return df.with_columns(
        pl.when(pl.col("code").str.starts_with("TRANSFER_TO//transfer//"))
        .then(
            pl.concat_str(
                [
                    pl.lit("TRANSFER_TO//"),
                    pl.col("code")
                    .str.extract(r"TRANSFER_TO//transfer//(.+)", 1)
                    .str.to_lowercase()
                    .str.replace_all(" ", "_"),
                ]
            )
        )
        .otherwise(pl.col("code"))
        .alias("code")
    )


def update_transfers_fntr(stage_cfg: DictConfig) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Return a function that updates the transfer codes."""
    compute_fns = []
    compute_fns.append(partial(update_discharge_code))
    compute_fns.append(partial(update_ed_code))
    compute_fns.append(partial(update_admit_code))
    compute_fns.append(partial(update_transfer_code))

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

    map_over(cfg, compute_fn=update_transfers_fntr)


if __name__ == "__main__":  # pragma: no cover
    main()
