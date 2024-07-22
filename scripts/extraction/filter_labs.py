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


def filter_lab_codes_fntr(filtered_lab_codes: set) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Create a function that filters out rows with specific lab codes.
    """
    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        return df.filter(
            ~(
                pl.col("code").cast(pl.Utf8).str.starts_with("LAB//") & 
                pl.col("code").cast(pl.Utf8).str.replace("LAB//", "").is_in(filtered_lab_codes)
            )
        )
    return fn


def filter_non_numeric_lab_values() -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Create a function that filters out rows with non-numeric lab values for LAB codes.
    """
    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        valid_number_expr = (
            (pl.col("lab_value").cast(pl.Utf8).str.contains(r'^\d+$') |               # Integers
             pl.col("lab_value").cast(pl.Utf8).str.contains(r'^\d+\.\d+$') |         # Decimal numbers
             pl.col("lab_value").cast(pl.Utf8).str.contains(r'^\.\d+$') |            # Leading decimal point
             pl.col("lab_value").cast(pl.Utf8).str.contains(r'^\d+\.$'))             # Trailing decimal point
        )

        df_filtered = df.filter(
            ~(
                pl.col("code").cast(pl.Utf8).str.starts_with("LAB//") &
                ~(valid_number_expr)
            )
        )
        return df_filtered
    return fn
    # def fn(df: pl.LazyFrame) -> pl.LazyFrame:
    #     return df.filter(
    #         ~(
    #              pl.col("code").cast(pl.Utf8).str.starts_with("LAB//") &
    #              pl.col("lab_value").cast(pl.Utf8).str.contains(r'[^0-9.]') |
    #              pl.col("lab_value").is_in(["", "."])
    #         )
    #     )
    # return fn

@hydra.main(version_base=None, config_path="../../configs", config_name="extraction")
def main(cfg: DictConfig):
    """Filter lab codes and values."""

    hydra_loguru_init()

    logger.info(
        f"Running with config:\n{OmegaConf.to_yaml(cfg)}\n"
        f"Stage: {cfg.stage}\n\n"
        f"Stage config:\n{OmegaConf.to_yaml(cfg.stage_cfg)}"
    )
    
    shards = json.loads((Path(cfg.stage_cfg.metadata_input_dir) / "splits.json").read_text())
    splits = [key.split('/')[0] for key in shards.keys()]
    patient_splits = set(splits)
    
    logger.info("Starting to filter lab codes and values.")
    
    input_dir = Path(cfg.stage_cfg.data_input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input cohort directory not found: {input_dir}")

    # Load the JSON file containing the lab codes to be filtered
    with open(cfg.stage_cfg.lab_codes_file, 'r') as f:
        filtered_lab_codes = set(json.load(f))

    # Create the function to filter out the specified lab codes
    codes_fn = filter_lab_codes_fntr(filtered_lab_codes)
    values_fn = filter_non_numeric_lab_values()

    compute_fns = [codes_fn, values_fn]

    for sp in ["train"]:
        in_dir = input_dir / sp
        all_files = sorted(list(in_dir.glob("**/*.parquet")))

        for f in all_files:
            out_fp = Path(cfg.stage_cfg.output_dir) / sp / f.name
            logger.info(f"Filtering LAB codes for {f} and writing into {out_fp}")
            
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
            
    logger.info("Filtering completed.")

if __name__ == "__main__":
    main()
