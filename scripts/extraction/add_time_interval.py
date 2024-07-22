#!/usr/bin/env python
import json
import random
from functools import partial
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

def add_new_events_fntr(fn: Callable[[pl.LazyFrame], pl.LazyFrame]) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Create a "meta" functor that computes the input functor on a MEDS shard then combines both dataframes.
    """
    def out_fn(df: pl.DataFrame) -> pl.DataFrame:
        new_events = fn(df)
        df = df.with_row_count("__idx")
        new_events = new_events.with_columns(pl.lit(0).alias("__idx").cast(pl.UInt32))
        return (
            pl.concat([df, new_events], how="diagonal")
            .sort(by=["patient_id", "timestamp", "__idx"])
            .drop("__idx")
        )

    return out_fn

def interval_fntr(discharge_code, interval_code) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Create a function that adds the interval between DISCHARGE//VE events and their next event to a DataFrame.
    """
    def fn(df: pl.DataFrame) -> pl.DataFrame:
        discharge_df = df.filter(pl.col("code") == discharge_code)
        next_events = (
            df.join(discharge_df, on="patient_id")
            .filter(pl.col("timestamp") > pl.col("timestamp_right"))
            .group_by(["patient_id", "timestamp_right"])
            .agg(pl.col("timestamp").min().alias("next_timestamp"))
        )
        with StringCache():
            interval_df = (
                discharge_df
                .join(
                    next_events, left_on=["patient_id", "timestamp"], right_on=["patient_id","timestamp_right"])
                .with_columns(pl.col("timestamp").alias("discharge_timestamp"))
                .select([
                    "patient_id",
                    "genc_id",
                    "discharge_timestamp",
                    "next_timestamp",
                    (pl.col("next_timestamp") - pl.col("discharge_timestamp")).alias("time_diff")
                 ])
                .with_columns([
                    pl.when((pl.col("time_diff") / pl.duration(days=1)) < 0)
                    .then(None)
                    .otherwise(pl.col("time_diff")).alias("time_diff"),
                ])
                .filter(pl.col("time_diff").is_not_null()).with_columns([
                    pl.when((pl.col("time_diff") / pl.duration(days=1)) < 28)
                    .then(pl.format(f"{interval_code}//[W_{{}}]", ((pl.col("time_diff") / pl.duration(days=1)) // 7).cast(pl.Int64)))
                    .when(
                        ((pl.col("time_diff") / pl.duration(days=1)) >= 28) & 
                        ((pl.col("time_diff") / pl.duration(days=1)) <= 365)
                    )
                    .then(pl.format(f"{interval_code}//[M_{{}}]", ((pl.col("time_diff") / pl.duration(days=1)) // 30).cast(pl.Int64)))
                    .when(
                        ((pl.col("time_diff") / pl.duration(days=1)) > 365)
                    )
                    .then(pl.format(f"{interval_code}//[LT]")).alias("code")
                ])
                .with_columns(pl.col("code").cast(pl.Categorical))
                .select([
                    "patient_id",
                    "genc_id",
                    (pl.col("discharge_timestamp") + 
                    (pl.col("next_timestamp") - pl.col("discharge_timestamp")) / 2).alias("timestamp"),
                    "code"
                ])
            )
        return interval_df
    return fn

@hydra.main(version_base=None, config_path="../../configs", config_name="extraction")
def main(cfg: DictConfig):
    """Add inter-visit time intervals."""

    hydra_loguru_init()

    logger.info(
        f"Running with config:\n{OmegaConf.to_yaml(cfg)}\n"
        f"Stage: {cfg.stage}\n\n"
        f"Stage config:\n{OmegaConf.to_yaml(cfg.stage_cfg)}"
    )
    
    shards = json.loads((Path(cfg.stage_cfg.metadata_input_dir) / "splits.json").read_text())
    splits = [key.split('/')[0] for key in shards.keys()]
    patient_splits = set(splits)
    
    logger.info("Starting time interval calculation.")
    
    input_dir = Path(cfg.stage_cfg.data_input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input cohort directory not found: {input_dir}")

    compute_fns = []
    
    interval_fn = interval_fntr(cfg.stage_cfg.discharge_code, cfg.stage_cfg.interval_prefix)
    compute_fns.append(add_new_events_fntr(interval_fn))
    
    for sp in ["train"]:
        in_dir = input_dir / sp
        all_files = sorted(list(in_dir.glob("**/*.parquet")))

        for f in all_files:
            out_fp = Path(cfg.stage_cfg.output_dir) / sp / f.name
            logger.info(f"Computing time interveal for {f} and writing into {out_fp}")
            
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
            
    logger.info("Time interval addition completed.")

if __name__ == "__main__":
    main()
