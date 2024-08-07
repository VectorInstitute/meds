#!/usr/bin/env python
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Callable
from polars import StringCache

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from MEDS_polars_functions.mapper import wrap as rwlock_wrap
from MEDS_polars_functions.utils import hydra_loguru_init, write_lazyframe

pl.enable_string_cache()

def add_code_token_column() -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Create a function that adds a new column with the part of the code that comes after "//".
    """
    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        df = df.with_columns([
            pl.col("code").cast(pl.Utf8).str.extract(r'//(.*)', 1).alias("code_token")
        ])
        return df
    return fn

def update_lab_code() -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Create a function that updates the code column for rows where the code starts with LAB,
    replacing it with {code_token}_{binned_lab_value}.
    """
    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        df = df.with_columns([
            pl.when(pl.col("code_token").str.starts_with("LAB//"))
            .then(pl.concat_str([pl.col("code_token"), pl.lit("_"), pl.col("binned_lab_value").cast(pl.Utf8)], separator=""))
            .otherwise(pl.col("code_token"))
            .alias("code_token")
        ])
        return df
    return fn


def aggregate_fntr() -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Create a function that filters out rows with specific lab codes.
    """
    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        column_names = df.collect_schema().names()
        return df.group_by("patient_id").agg([pl.col(col) for col in column_names if col != "patient_id"])
    return fn


@hydra.main(version_base=None, config_path="../../configs", config_name="extraction")
def main(cfg: DictConfig):
    """Aggregates the column values in the rows that have the same patient_id into lists."""

    hydra_loguru_init()

    logger.info(
        f"Running with config:\n{OmegaConf.to_yaml(cfg)}\n"
        f"Stage: {cfg.stage}\n\n"
        f"Stage config:\n{OmegaConf.to_yaml(cfg.stage_cfg)}"
    )
    
    shards = json.loads((Path(cfg.stage_cfg.metadata_input_dir) / "splits.json").read_text())
    splits = [key.split('/')[0] for key in shards.keys()]
    patient_splits = set(splits)
    
    logger.info("Starting patient sequence generation.")
    
    input_dir = Path(cfg.stage_cfg.data_input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input cohort directory not found: {input_dir}")

    code_fn = add_code_token_column()
    lab_fn = update_lab_code()
    agg_fn = aggregate_fntr()

    compute_fns = [code_fn, lab_fn, agg_fn]

    for sp in ["train"]:
        in_dir = input_dir / sp
        all_files = sorted(list(in_dir.glob("**/*.parquet")))

        for f in all_files:
            out_fp = Path(cfg.stage_cfg.output_dir) / sp / f.name
            logger.info(f"Aggregating {f} shards into {out_fp}")
            
            rwlock_wrap(
                f,
                out_fp,
                pl.scan_parquet,
                write_lazyframe,
                *compute_fns,
                do_return=False,
                cache_intermediate=False,
                do_overwrite=cfg.do_overwrite,
            )
            # break
            
    logger.info("Aggregation completed.")

if __name__ == "__main__":
    main()
