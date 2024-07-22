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
            pl.concat([df, new_events], how="diagonal")
            .sort(by=["patient_id", "timestamp", "__idx"])
            .drop("__idx")
        )

    return out_fn

def add_reg_events_fntr(cfg: DictConfig) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Create a function that adds "REG" events after each "DISCHARGE//VE" event.
    """
    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        discharge_df = df.filter(pl.col("code") == cfg.discharge_code)
        
        with StringCache():
            reg_df = (
                discharge_df.with_columns([
                    pl.col("timestamp").alias("discharge_timestamp"),
                    (pl.col("timestamp") + pl.duration(seconds=1)).alias("timestamp"),
                    pl.lit(cfg.reg_code).alias("code")
                ])
                .with_columns(pl.col("code").cast(pl.Categorical))
                .select([
                    "patient_id",
                    "genc_id",
                    "timestamp",
                    "code"
                ])
            )
            
        return reg_df

    return fn


@hydra.main(version_base=None, config_path="../../configs", config_name="extraction")
def main(cfg: DictConfig):
    """Add register token column."""

    hydra_loguru_init()

    logger.info(
        f"Running with config:\n{OmegaConf.to_yaml(cfg)}\n"
        f"Stage: {cfg.stage}\n\n"
        f"Stage config:\n{OmegaConf.to_yaml(cfg.stage_cfg)}"
    )
    
    shards = json.loads((Path(cfg.stage_cfg.metadata_input_dir) / "splits.json").read_text())
    splits = [key.split('/')[0] for key in shards.keys()]
    patient_splits = set(splits)
    
    logger.info("Starting to add REG token column.")
    
    input_dir = Path(cfg.stage_cfg.data_input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input cohort directory not found: {input_dir}")

    # read_fn = partial(pl.scan_parquet, glob=False)
    compute_fns = []
    reg_cfg = DictConfig({"discharge_code": "DISCHARGE//VE", "reg_code": "REGISTER//REG"})

    # Define the function to add "REG" events after discharge events
    reg_fn = add_reg_events_fntr(reg_cfg)
    compute_fns.append(add_new_events_fntr(reg_fn))
    
    for sp in ["train"]:
        in_dir = input_dir / sp
        all_files = sorted(list(in_dir.glob("**/*.parquet")))

        for f in all_files:
            out_fp = Path(cfg.stage_cfg.output_dir) / sp / f.name
            logger.info(f"Adding register token for {f} and writing into {out_fp}")
            
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
            
    logger.info("REG token addition completed.")

if __name__ == "__main__":
    main()
