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

def add_age() -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Create a function that fills missing age values in the LazyFrame.
    """
    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        age_mapping = df.filter(pl.col("code") == "ADMISSION//VS").select(["genc_id", "age"])
        df_filled = df.join(age_mapping, on="genc_id", how="left").with_columns([
            pl.when(pl.col("age").is_null()).then(pl.col("age_right")).otherwise(pl.col("age")).alias("age_filled")
        ]).select([col for col in df.columns if col != "age"] + ["age_filled"])
        df_filled = df_filled.rename({"age_filled": "age"})
        return df_filled
    return fn

def add_token_type(token_type_mapping) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Creates a function that adds token_type_id based on the code values and token_type_mapping.
    """
    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        token_type_mapping_expr = pl.when(
            pl.col("code").cast(pl.Utf8).str.extract(r'([^//]+)', 1).is_in(list(token_type_mapping.keys()))
        ).then(
            pl.col("code").cast(pl.Utf8).str.extract(r'([^//]+)', 1).replace(token_type_mapping)
        ).otherwise(pl.lit(-1)).alias("type_id")
        
        df = df.with_columns(token_type_mapping_expr)
        return df
    return fn

def add_visit_segments() -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Create a function that adds visit_segments based on alternating genc_id for each patient.
    """
    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        df = df.with_columns([
            (pl.col("genc_id") != pl.col("genc_id").shift(1).fill_null(pl.col("genc_id"))).cast(pl.Int32).cum_sum().over("patient_id").alias("genc_id_group")
        ])
        
        # Assign alternating visit_segments based on the genc_id_group
        df = df.with_columns([
            ((pl.col("genc_id_group") % 2) + 1).alias("visit_segment")
            ]).drop("genc_id_group")
        
        return df
    return fn

def add_visit_order() -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Create a function that adds visit_order based on changes in genc_id for each patient, starting from 1.
    """
    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        df = df.with_columns([
            (pl.col("genc_id") != pl.col("genc_id").shift(1).fill_null(pl.col("genc_id")))
            .cast(pl.Int32)
            .cum_sum()
            .over("patient_id")
            .alias("visit_order")
        ])
        df = df.with_columns((pl.col("visit_order") + 1).alias("visit_order"))
        return df
    return fn
    
def add_time(reference_timestamp: pl.datetime) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Creates a function that adds a time column based on the difference in weeks between the timestamp column and a reference timestamp.
    """
    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        df = df.with_columns([
            (pl.col("timestamp") - reference_timestamp).alias("duration")
            ]).with_columns([
            (pl.col("duration") / pl.duration(weeks=1)).alias("time")
            ]).with_columns([
            pl.col("time").ceil().alias("time")]).drop("duration")
        
        return df
    return fn


@hydra.main(version_base=None, config_path="../../configs", config_name="extraction")
def main(cfg: DictConfig):
    """Processes the DataFrame to fill age values and writes the output."""

    hydra_loguru_init()

    logger.info(
        f"Running with config:\n{OmegaConf.to_yaml(cfg)}\n"
        f"Stage: {cfg.stage}\n\n"
        f"Stage config:\n{OmegaConf.to_yaml(cfg.stage_cfg)}"
    )
    
    shards = json.loads((Path(cfg.stage_cfg.metadata_input_dir) / "splits.json").read_text())
    splits = [key.split('/')[0] for key in shards.keys()]
    patient_splits = set(splits)
    
    logger.info("Starting to add embedding columns.")
    
    patient_final_dir = Path(cfg.stage_cfg.data_input_dir)
    if not patient_final_dir.is_dir():
        raise FileNotFoundError(f"Patient final cohort directory not found: {patient_final_dir}")

    # Create the function to fill age values
    age_fn = add_age()
    token_fn = add_token_type(cfg.stage_cfg.token_type_mapping)
    visit_segments_fn = add_visit_segments()
    visit_order_fn = add_visit_order()
    
    year = cfg.stage_cfg.reference_year
    month = cfg.stage_cfg.reference_month
    day = cfg.stage_cfg.reference_day
    reference_timestamp = pl.datetime(year, month, day)
    time_fn = add_time(reference_timestamp)

    compute_fns = [age_fn, token_fn, visit_segments_fn, visit_order_fn, time_fn]

    for sp in ["train"]:
        in_dir = patient_final_dir / sp
        all_files = sorted(list(in_dir.glob("**/*.parquet")))

        for f in all_files:
            out_fp = Path(cfg.stage_cfg.output_dir) / sp / f.name
            logger.info(f"Adding embedding columns for {f} and writing into {out_fp}")
            
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
            
    logger.info("Adding embedding columns completed.")

if __name__ == "__main__":
    main()
