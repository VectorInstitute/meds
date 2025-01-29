import json
from collections.abc import Callable
from functools import partial

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over


def add_new_events_fntr(fn: Callable[[pl.DataFrame], pl.DataFrame]) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Creates a "meta" functor that computes the input functor on a MEDS shard
    then combines both dataframes."""

    def out_fn(df: pl.DataFrame) -> pl.DataFrame:
        new_events = fn(df)
        df = df.with_row_count("__idx")
        new_events = new_events.with_columns(pl.lit(0).alias("__idx").cast(pl.UInt32))
        return (
            pl.concat([df, new_events], how="diagonal").sort(by=["subject_id", "time", "__idx"]).drop("__idx")
        )

    return out_fn


def calculate_time_diff_hours(df: pl.LazyFrame) -> pl.LazyFrame:
    """Creates a function that adds a time diff column based on the difference in hours
    between the timestamp column and the first event for each patient."""
    time_diff_expr = pl.col("time").diff().over("subject_id").dt.total_hours()
    return df.with_columns(time_diff_expr.alias("time_diff"))


def calculate_minutes_for_zero_hours(df: pl.LazyFrame) -> pl.LazyFrame:
    """Updates the time_diff column to calculate time differences in minutes for rows where time_diff is 0."""
    time_diff_in_minutes_expr = pl.when(pl.col("time_diff") == 0).then(  # Check if time_diff is 0
        pl.col("time").diff().over("subject_id").dt.total_minutes()
    )  # Calculate the time difference in minutes

    return df.with_columns(time_diff_in_minutes_expr.alias("time_diff_minutes"))


def hour_bins_fntr(stage_cfg: DictConfig) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Creates a function that adds hour bins to the LazyFrame."""
    if stage_cfg.hour_bins_file is not None:
        with open(stage_cfg.hour_bins_file) as f:
            hour_bins = json.load(f)
    else:
        raise ValueError("Either hour_bins or hour_bins_file must be specified in the stage configuration.")

    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        hour_bin_dict = {int(k): v for k, v in hour_bins.items()}
        hour_bins_df = pl.LazyFrame(
            {"time_diff": list(hour_bin_dict.keys()), "time_diff_bin": list(hour_bin_dict.values())}
        )
        df_with_bins = df.join(hour_bins_df, on="time_diff", how="inner")
        df_with_bins = df_with_bins.with_columns((pl.col("time") - pl.duration(seconds=1)).alias("time"))
        time_df = df_with_bins.with_columns(
            [
                pl.concat_str([pl.lit("TIME//"), pl.col("time_diff_bin")]).alias("code"),
            ]
        ).drop("time_diff_bin")
        return time_df

    return fn


def minute_bins_fntr(stage_cfg: DictConfig) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Creates a function that adds minute bins to the LazyFrame."""
    if stage_cfg.minute_bins_file is not None:
        with open(stage_cfg.minute_bins_file) as f:
            minute_bins = json.load(f)
    else:
        raise ValueError(
            "Either minute_bins or minute_bins_file must be specified in the stage configuration."
        )

    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        minute_bin_dict = {int(k): v for k, v in minute_bins.items()}
        minute_bins_df = pl.LazyFrame(
            {
                "time_diff_minutes": list(minute_bin_dict.keys()),
                "time_diff_bin": list(minute_bin_dict.values()),
            }
        )
        df_with_bins = df.join(minute_bins_df, on="time_diff_minutes", how="inner")
        df_with_bins = df_with_bins.with_columns((pl.col("time") - pl.duration(seconds=1)).alias("time"))
        time_df = df_with_bins.with_columns(
            [
                pl.concat_str([pl.lit("TIME//"), pl.col("time_diff_bin")]).alias("code"),
            ]
        ).drop("time_diff_bin")
        return time_df

    return fn


def add_time_fntr(stage_cfg: DictConfig) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Return a function that adds time tokens to the DataFrame."""
    compute_fns = []

    compute_fns.append(partial(calculate_time_diff_hours))
    compute_fns.append(partial(calculate_minutes_for_zero_hours))
    compute_fns.append(add_new_events_fntr(minute_bins_fntr(stage_cfg)))
    compute_fns.append(add_new_events_fntr(hour_bins_fntr(stage_cfg)))

    logger.info("Adding time tokens.")

    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        for compute_fn in compute_fns:
            df = compute_fn(df)
        return df

    return fn


@hydra.main(
    version_base=None, config_path=str(PREPROCESS_CONFIG_YAML.parent), config_name=PREPROCESS_CONFIG_YAML.stem
)
def main(cfg: DictConfig):
    """TODO."""

    map_over(cfg, compute_fn=add_time_fntr)


if __name__ == "__main__":  # pragma: no cover
    main()
