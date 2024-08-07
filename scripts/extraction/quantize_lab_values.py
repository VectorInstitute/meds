#!/usr/bin/env python
import json
import random
from functools import partial
from pathlib import Path
from datetime import datetime
from typing import Callable
from polars import StringCache
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
from pathlib import Path
import time
from typing import Dict, Callable, List, Optional, Any
from tqdm import tqdm
import gc

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from MEDS_polars_functions.mapper import wrap as rwlock_wrap
from MEDS_polars_functions.utils import hydra_loguru_init
from MEDS_polars_functions.utils import hydra_loguru_init, write_lazyframe

pl.enable_string_cache()


def quantize_lab_values_fntr(bins: set) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Create a function that quantizes the lab values into specified bins for each LAB code.
    """
    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        lab_codes = df.filter(pl.col('code').cast(pl.Utf8).str.contains('^LAB//')).select('code').unique().collect().to_series()
        updated_frames = []

        for code in lab_codes:
            lab_df = df.filter(pl.col('code') == code)

            lab_df = lab_df.with_columns(
                pl.col('lab_value').cast(pl.Utf8).cast(pl.Float64).alias('lab_value')
            )
            valid_lab_df = lab_df.filter(pl.col('lab_value').is_not_null())
            
            code_suffix = code.split("//")[1]
            bin_edges = bins[code_suffix].get("bin_edges")
            bin_labels = bins[code_suffix].get("bin_labels")
            
            if bin_edges is None:
                logger.warning(f"Min and max values are equal for lab code {code}. Casting lab_value to string.")
                valid_lab_df = valid_lab_df.with_columns(
                    pl.col('lab_value').cast(pl.Int64).cast(pl.Utf8).alias('binned_lab_value')
                )

            else:
                valid_lab_df = valid_lab_df.with_columns(
                    pl.Expr.cut(pl.col('lab_value'),
                    breaks=bin_edges, 
                    labels=bin_labels,
                    include_breaks=False).cast(pl.Utf8).alias('binned_lab_value')
                )

            updated_frames.append(valid_lab_df)

        unchanged_df = df.filter(~pl.col('code').cast(pl.Utf8).str.contains('^LAB//') | pl.col('lab_value').is_null())
        unchanged_df = unchanged_df.with_columns(
            pl.col('lab_value').cast(pl.Utf8).cast(pl.Float64).alias('lab_value')
        )

        return (
            pl.concat(updated_frames + [unchanged_df], how="diagonal")
            .sort(by=["patient_id", "timestamp"])
        )

    return fn


def write_fn(df: pl.LazyFrame, out_fp: Path) -> None:
    df.collect().write_parquet(out_fp, use_pyarrow=True)


@hydra.main(version_base=None, config_path="../../configs", config_name="extraction")
def main(cfg: DictConfig):
    """Quantizes the lab values in the rows that have the same patient_id into specified bins."""

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
    
    patient_final_dir = Path(cfg.stage_cfg.data_input_dir)
    if not patient_final_dir.is_dir():
        raise FileNotFoundError(f"Patient final cohort directory not found: {patient_final_dir}")

    # Load the bins and labels from the configuration
    bins_number = cfg.stage_cfg.bins_number
    
    # Load the JSON file containing the labs min and max
    with open(cfg.stage_cfg.lab_minmax_file, 'r') as f:
        lab_minmax = json.load(f)
        
    for lab_code, values in lab_minmax.items():
        min_val = values["min"]
        max_val = values["max"]
        
        if min_val == max_val:
            logger.warning(f"Min and max values are equal for lab code {lab_code}. Skipping binning for this code.")
            continue
        
        bin_edges = [min_val + i * (max_val - min_val) / bins_number for i in range(bins_number + 1)]
        bin_labels = [f"{i}" for i in range(bins_number + 2)]
        
        lab_minmax[lab_code]["bin_edges"] = bin_edges
        lab_minmax[lab_code]["bin_labels"] = bin_labels

    # Create the function to quantize lab values
    quantize_fn = quantize_lab_values_fntr(lab_minmax)

    compute_fns = [quantize_fn]

    for sp in ["train"]:
        in_dir = patient_final_dir / sp
        all_files = sorted(list(in_dir.glob("**/*.parquet")))

        for f in all_files:
            logger.info(f.name)
            out_fp = Path(cfg.stage_cfg.output_dir) / sp / f.name
            logger.info(out_fp)
            logger.info(f"Quantizing lab values for {f} and writing into {out_fp}")
            
            rwlock_wrap(
                f,
                out_fp,
                pl.scan_parquet,
                write_fn,
                *compute_fns,
                do_return=False,
                cache_intermediate=False,
                do_overwrite=cfg.do_overwrite,
            )
            
            # break
        # break
    logger.info("Quantization completed.")

if __name__ == "__main__":
    main()
