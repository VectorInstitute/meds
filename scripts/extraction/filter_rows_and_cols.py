#!/usr/bin/env python
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Callable, List
from polars import StringCache

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from MEDS_polars_functions.mapper import wrap as rwlock_wrap
from MEDS_polars_functions.utils import hydra_loguru_init, write_lazyframe

pl.enable_string_cache()

def select_columns(columns_to_keep: List[str]) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Creates a function that selects only the specified columns from the DataFrame.
    """
    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        df_selected = df.select(columns_to_keep)
        return df_selected
    return fn

def remove_diagnosis_rows() -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Create a function that removes rows where the code column value starts with 'DIAGNOSIS'.
    """
    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        df_filtered = df.filter(~pl.col("code").cast(pl.Utf8).str.starts_with("DIAGNOSIS"))
        return df_filtered
    return fn

@hydra.main(version_base=None, config_path="../../configs", config_name="extraction")
def main(cfg: DictConfig):

    hydra_loguru_init()

    logger.info(
        f"Running with config:\n{OmegaConf.to_yaml(cfg)}\n"
        f"Stage: {cfg.stage}\n\n"
        f"Stage config:\n{OmegaConf.to_yaml(cfg.stage_cfg)}"
    )
    
    shards = json.loads((Path(cfg.stage_cfg.metadata_input_dir) / "splits.json").read_text())
    splits = [key.split('/')[0] for key in shards.keys()]
    patient_splits = set(splits)
    
    logger.info("Starting to filter rows and cols.")
    
    input_dir = Path(cfg.stage_cfg.data_input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input cohort directory not found: {input_dir}")

    # Create the function to filter rows and cols
    row_fn = remove_diagnosis_rows()
    col_fn = select_columns(cfg.stage_cfg.columns_to_keep)

    compute_fns = [row_fn, col_fn]

    for sp in ["train"]:
        in_dir = input_dir / sp
        all_files = sorted(list(in_dir.glob("**/*.parquet")))

        for f in all_files:
            out_fp = Path(cfg.stage_cfg.output_dir) / sp / f.name
            logger.info(f"Filtering df from {f} and writing into {out_fp}")
            
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
