#!/usr/bin/env python
import json
import os
from collections.abc import Callable
from functools import partial

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over


def extract_vocab(parquet_dir: str, output_dir: str, split: str = "train"):
    """Extract unique codes from the dataset and save them to a JSON file."""
    output_json = os.path.join(output_dir, "vocab.json")
    df_lazy = pl.scan_parquet(os.path.join(parquet_dir, split, "*.parquet")).select("code")
    unique_codes_df = df_lazy.unique().sort("code").collect()
    unique_codes = unique_codes_df["code"].to_list()
    with open(output_json, "w") as f:
        json.dump(unique_codes, f, indent=4)
    logger.info(f"Unique codes have been saved to {output_json}")


def extract_subjects(parquet_dir: str, output_dir: str, split: str = "train"):
    """Extract unique subject IDs from the dataset and save them to a JSON file."""
    output_json = os.path.join(output_dir, "subject_ids.json")
    parquet_files_pattern = os.path.join(parquet_dir, split, "*.parquet")
    lazy_df = pl.scan_parquet(parquet_files_pattern)
    unique_subject_ids_df = lazy_df.select("subject_id").unique().collect()
    subject_id_list = unique_subject_ids_df["subject_id"].to_list()
    with open(output_json, "w") as f:
        json.dump(subject_id_list, f, indent=4)
    logger.info(f"Unique subject IDs have been saved to {output_json}")


def aggregate(df: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate the DataFrame by subject_id."""
    df = df.select(["subject_id", "code"])
    column_names = df.collect_schema().names()
    return df.group_by("subject_id").agg([pl.col(col) for col in column_names if col != "subject_id"])


def generate_sequence_fntr(stage_cfg: DictConfig) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Return a function that generates sequences from the DataFrame."""
    extract_vocab(stage_cfg.data_input_dir, stage_cfg.output_dir)
    extract_subjects(stage_cfg.data_input_dir, stage_cfg.output_dir)
    compute_fns = []
    logger.info("Aggregating patient data")
    compute_fns.append(partial(aggregate))

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

    map_over(cfg, compute_fn=generate_sequence_fntr)


if __name__ == "__main__":  # pragma: no cover
    main()
