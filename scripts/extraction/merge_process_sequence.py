#!/usr/bin/env python
from pathlib import Path
from typing import Callable, Dict
import json

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from MEDS_polars_functions.mapper import wrap as rwlock_wrap
from MEDS_polars_functions.utils import hydra_loguru_init, write_lazyframe

pl.enable_string_cache()

# Constants
MAX_LENGTH = 2048
COLUMN_MAPPINGS: Dict[str, str] = {
    "code_token": f"event_tokens_{MAX_LENGTH}",
    "age": f"age_tokens_{MAX_LENGTH}",
    "time": f"time_tokens_{MAX_LENGTH}",
    "type_id": f"type_tokens_{MAX_LENGTH}",
    "visit_segment": f"visit_tokens_{MAX_LENGTH}",
    "visit_order": f"position_tokens_{MAX_LENGTH}",
    "timestamp": f"elapsed_tokens_{MAX_LENGTH}"
}

# Define the constant for columns to be removed
COLUMNS_TO_REMOVE = ["discharge_disposition", "lab_value", "binned_lab_value"]

# Token replacement mapping
TOKEN_REPLACEMENTS = {
    "VS": "[VS]",
    "VE": "[VE]",
    "REG": "[REG]",
    "N/A": "[UNK]",
    "n/a": "[UNK]",
    "0.0": "[UNK]",
    "0": "[UNK]"
}

def token_replacement(x: str) -> str:
    """Define the logic for processing specific token names."""
    if x in TOKEN_REPLACEMENTS:
        return TOKEN_REPLACEMENTS[x]
    if x.endswith(".0") and len(x) > 2:
        return x[:-2]
    return x

def apply_processing(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Apply processing to multiple columns of the DataFrame.
    
    :param df: Input LazyFrame
    :return: Processed LazyFrame
    """
    # Step 0: Remove unnecessary columns
    df = df.drop(COLUMNS_TO_REMOVE)

    # Step 1: Rename columns
    df = df.rename(COLUMN_MAPPINGS)

    # Step 2: Process event_tokens (replace special tokens) and count visits
    df = df.with_columns([
        pl.col(f"event_tokens_{MAX_LENGTH}").list.eval(
            pl.element().cast(pl.Utf8).map_elements(token_replacement, return_dtype=pl.Utf8)
        )
    ])

    # Step 3: Count visits (number of "[VS]" tokens)
    df = df.with_columns([
        pl.col(f"event_tokens_{MAX_LENGTH}").list.count_matches("[VS]").alias("num_visits")
    ])

    # Step 4: Truncate all columns to MAX_LENGTH
    for col in COLUMN_MAPPINGS.values():
        df = df.with_columns([
            pl.col(col).list.slice(0, MAX_LENGTH)
        ])

    # Step 5: Change the last two event tokens to [VE] and [REG]
    df = df.with_columns([
        pl.col(f"event_tokens_{MAX_LENGTH}").list.len().alias("token_length")
    ])
    df = df.with_columns([
        pl.col(f"event_tokens_{MAX_LENGTH}")
        .list.slice(0, pl.col("token_length") - 2)
        .list.concat(pl.Series([["[VE]", "[REG]"]]))
    ])
    df = df.with_columns([
        pl.col(f"type_tokens_{MAX_LENGTH}")
        .list.slice(0, pl.col("token_length") - 2)
        .list.concat(pl.Series([[3, 8]]))
    ])

    # Step 6: Join the event_tokens list into a space-separated string
    df = df.with_columns([
        pl.col(f"event_tokens_{MAX_LENGTH}").list.join(" ")
    ])

    return df

def create_vocabulary(df: pl.DataFrame, output_file: Path):
    """Create a vocabulary from the event tokens in the DataFrame and write it to a JSON file."""
    logger.info("Creating vocabulary from event tokens")
    
    # Extract unique tokens from the event_tokens column
    vocab = (
        df.select(pl.col(f"event_tokens_{MAX_LENGTH}"))
        .get_column(f"event_tokens_{MAX_LENGTH}")
        .str.split(" ")
        .explode()
        .unique()
        .sort()
        .to_list()
    )
    
    # Do not add special tokens to the vocabulary
    vocab = [token for token in vocab if token[0] != '[']
    
    logger.info(f"Vocabulary size: {len(vocab)}")
    
    # Write vocabulary to JSON file
    with open(output_file, "w") as f:
        json.dump(vocab, f, indent=2)
    
    logger.info(f"Vocabulary written to {output_file}")

def compute_statistics(df: pl.DataFrame) -> dict:
    """Compute basic statistics of the DataFrame."""
    stats = {
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "column_stats": {}
    }
    for col in df.columns:
        col_stats = df[col].describe().to_dict()
        stats["column_stats"][col] = col_stats
    return stats

def merge_and_process_files(input_dir: Path, output_file: Path, process_fn: Callable[[pl.LazyFrame], pl.LazyFrame]):
    """Merge all Parquet files in the input directory, apply processing, and save the result."""
    parquet_files = list(input_dir.glob("**/*.parquet"))
    
    if not parquet_files:
        logger.warning(f"No Parquet files found in {input_dir}")
        return None

    df = pl.concat([pl.scan_parquet(file) for file in parquet_files])
    df = process_fn(df).collect()

    logger.info("Begin writing to disk...")
    df.write_parquet(output_file, use_pyarrow=False)
    
    logger.info(f"Merged and processed data saved to {output_file}")
    
    return df

@hydra.main(version_base=None, config_path="../../configs", config_name="extraction")
def main(cfg: DictConfig):
    """Merges parquet files, applies processing, computes statistics, and tests the result."""

    hydra_loguru_init()

    logger.info(
        f"Running with config:\n{OmegaConf.to_yaml(cfg)}\n"
        f"Stage: {cfg.stage}\n\n"
        f"Stage config:\n{OmegaConf.to_yaml(cfg.stage_cfg)}"
    )
    
    input_dir = Path(cfg.stage_cfg.data_input_dir) / "train"
    out_fp = Path(cfg.stage_cfg.output_dir) / "train" / "dataset.parquet"
    
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    logger.info(f"Merging and processing files from {input_dir}")
    df = merge_and_process_files(input_dir, out_fp, apply_processing)#.collect()

    logger.info(f"Statistics:\n{compute_statistics(df)}")
    create_vocabulary(df, Path(cfg.stage_cfg.output_dir) / "train" / "vocabulary.json")

    logger.info("All operations ran successfully.")

if __name__ == "__main__":
    main()