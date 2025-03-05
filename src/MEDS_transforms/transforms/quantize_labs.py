#!/usr/bin/env python
import glob
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


def filter_and_get_min_max_lab_values(parquet_folder_path: str, output_json_path: str):
    """Filter out lab values and get the min and max values for each lab code."""
    lab_min_max = {}
    parquet_files = glob.glob(f"{parquet_folder_path}/**/*.parquet", recursive=True)

    for parquet_file in parquet_files:
        df = pl.scan_parquet(parquet_file)

        lab_df = df.filter(pl.col("code").str.contains("LAB//"))
        lab_df_collected = lab_df.collect()

        for lab_code in lab_df_collected["code"].unique():
            lab_code_df = lab_df_collected.filter(pl.col("code") == lab_code)
            numeric_values = [float(v) for v in lab_code_df["numeric_value"]]

            if numeric_values:
                if lab_code not in lab_min_max:
                    lab_min_max[lab_code] = {"min": min(numeric_values), "max": max(numeric_values)}
                else:
                    lab_min_max[lab_code]["min"] = min(lab_min_max[lab_code]["min"], min(numeric_values))
                    lab_min_max[lab_code]["max"] = max(lab_min_max[lab_code]["max"], max(numeric_values))

    with open(output_json_path, "w") as f:
        json.dump(lab_min_max, f, indent=4)


def quantize_lab_values(df: pl.LazyFrame, bins: set) -> pl.LazyFrame:
    """Create a function that quantizes the lab values into specified bins for each LAB code."""
    lab_codes = (
        df.filter(pl.col("code").cast(pl.Utf8).str.contains("LAB//"))
        .select("code")
        .unique()
        .collect()
        .to_series()
    )
    updated_frames = []

    for code in lab_codes:
        lab_df = df.filter(pl.col("code") == code)
        lab_df = lab_df.with_columns(
            pl.col("numeric_value").cast(pl.Utf8).cast(pl.Float64).alias("numeric_value")
        )
        valid_lab_df = lab_df.filter(pl.col("numeric_value").is_not_null())

        bin_breaks = bins[code].get("bin_breaks")
        bin_labels = bins[code].get("bin_labels")

        if bin_breaks is None:
            logger.warning(f"Min and max values are equal for lab code {code}. Casting lab_value to string.")
            valid_lab_df = valid_lab_df.with_columns(
                pl.col("numeric_value").cast(pl.Int64).cast(pl.Utf8).alias("binned_lab_value")
            )

        else:
            valid_lab_df = valid_lab_df.with_columns(
                pl.col("numeric_value")
                .cut(breaks=bin_breaks, labels=bin_labels, include_breaks=False)
                .cast(pl.Utf8)
                .alias("binned_lab_value")
            )

        valid_lab_df = valid_lab_df.with_columns(
            pl.concat_str([pl.col("code"), pl.lit("//"), pl.col("binned_lab_value")]).alias("code")
        )
        updated_frames.append(valid_lab_df)

    unchanged_df = df.filter(
        ~pl.col("code").cast(pl.Utf8).str.contains("LAB//") | pl.col("numeric_value").is_null()
    )
    unchanged_df = unchanged_df.with_columns(
        pl.col("numeric_value").cast(pl.Utf8).cast(pl.Float64).alias("numeric_value")
    )

    return pl.concat(updated_frames + [unchanged_df], how="diagonal").sort(by=["subject_id", "time"])


def quantize_lab_values_qa(df: pl.LazyFrame, bins: set) -> pl.LazyFrame:
    """Create a function that quantizes the lab values into specified bins for each LAB code
    for the QA task."""
    df = df.with_columns(pl.col("numeric_value").cast(pl.Utf8).cast(pl.Float64).alias("numeric_value"))
    df = df.with_columns(pl.lit(None).cast(pl.Int64).alias("binned_lab_value"))

    lab_codes = (
        df.filter(pl.col("code").cast(pl.Utf8).str.contains("LAB//"))
        .select("code")
        .unique()
        .collect()
        .to_series()
    )
    updated_frames = []

    for code in lab_codes:
        try:
            lab_df = df.filter(pl.col("code") == code)
            valid_lab_df = lab_df.filter(pl.col("numeric_value").is_not_null())

            bin_breaks = bins[code].get("bin_breaks")
            bin_labels = bins[code].get("bin_labels")

            if bin_breaks is None:
                logger.warning(
                    f"Min and max values are equal for lab code {code}. Casting lab_value to string."
                )
                valid_lab_df = valid_lab_df.with_columns(
                    pl.col("numeric_value").cast(pl.Int64).alias("binned_lab_value")
                )
            else:
                valid_lab_df = valid_lab_df.with_columns(
                    pl.col("numeric_value")
                    .cut(breaks=bin_breaks, labels=bin_labels, include_breaks=False)
                    .cast(pl.Int64)
                    .alias("binned_lab_value")
                )
            updated_frames.append(valid_lab_df)
        except KeyError:
            logger.warning(f"Lab code {code} not found in bins. Skipping binning for this code.")
            continue

    unchanged_df = df.filter(
        ~pl.col("code").cast(pl.Utf8).str.contains("LAB//") | pl.col("numeric_value").is_null()
    )

    result_df = pl.concat(updated_frames + [unchanged_df], how="diagonal").sort(by=["subject_id", "time"])

    return result_df


def quantize_fntr(stage_cfg: DictConfig) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Return a function that quantizes the lab values into specified bins for each LAB code."""
    compute_fns = []
    logger.info(stage_cfg)

    if os.path.exists(stage_cfg.lab_minmax_file):
        logger.info(f"Using lab minmax file: {stage_cfg.lab_minmax_file}")
    else:
        filter_and_get_min_max_lab_values(stage_cfg.data_input_dir, stage_cfg.lab_minmax_file)

    bins_number = stage_cfg.bins_number
    with open(stage_cfg.lab_minmax_file) as f:
        lab_minmax = json.load(f)

    for lab_code, values in lab_minmax.items():
        min_val = values["min"]
        max_val = values["max"]

        if min_val == max_val:
            logger.warning(
                f"Min and max values are equal for lab code {lab_code}. Skipping binning for this code."
            )
            continue

        bin_breaks = [min_val + i * (max_val - min_val) / bins_number for i in range(1, bins_number)]
        bin_labels = [f"{i}" for i in range(bins_number)]

        lab_minmax[lab_code]["bin_breaks"] = bin_breaks
        lab_minmax[lab_code]["bin_labels"] = bin_labels

    if stage_cfg.is_qa:
        compute_fns.append(partial(quantize_lab_values_qa, bins=lab_minmax))
    else:
        compute_fns.append(partial(quantize_lab_values, bins=lab_minmax))

    def fn(data: pl.LazyFrame) -> pl.LazyFrame:
        for compute_fn in compute_fns:
            data = compute_fn(data)
        return data

    return fn


@hydra.main(
    version_base=None, config_path=str(PREPROCESS_CONFIG_YAML.parent), config_name=PREPROCESS_CONFIG_YAML.stem
)
def main(cfg: DictConfig):
    """Adds time-derived measurements to a MEDS cohort as separate observations at each unique time."""
    map_over(cfg, compute_fn=quantize_fntr)


if __name__ == "__main__":  # pragma: no cover
    main()
