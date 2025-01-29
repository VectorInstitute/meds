#!/usr/bin/env python
from collections.abc import Callable

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from MEDS_transforms import INFERRED_STAGE_KEYS, PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over


def add_new_events_fntr(
    fn: Callable[[pl.DataFrame], pl.DataFrame], cfg: DictConfig
) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Creates a "meta" functor that computes the input functor on a MEDS shard
    then combines both dataframes."""

    def out_fn(df: pl.DataFrame) -> pl.DataFrame:
        new_events = fn(df)
        df = df.with_row_index("__idx")
        df = df.filter(pl.col("code") != cfg.DOB_code)
        new_events = new_events.with_columns(pl.lit(0, dtype=df.schema["__idx"]).alias("__idx"))
        return (
            pl.concat([df, new_events], how="diagonal").sort(by=["subject_id", "time", "__idx"]).drop("__idx")
        )

    return out_fn


TIME_DURATION_UNITS = {
    "seconds": (["s", "sec", "secs", "second", "seconds"], 1),
    "minutes": (["m", "min", "mins", "minute", "minutes"], 60),
    "hours": (["h", "hr", "hrs", "hour", "hours"], 60 * 60),
    "days": (["d", "day", "days"], 60 * 60 * 24),
    "weeks": (["w", "wk", "wks", "week", "weeks"], 60 * 60 * 24 * 7),
    "months": (["mo", "mos", "month", "months"], 60 * 60 * 24 * 30.436875),
    "years": (["y", "yr", "yrs", "year", "years"], 60 * 60 * 24 * 365.2422),
}


def normalize_time_unit(unit: str) -> tuple[str, float]:
    """Normalize a time unit string to a canonical form and return the number of seconds in that unit.

    Note that this function is designed for computing _approximate_ time durations over long periods, not
    canonical, local calendar time durations. E.g., a "month" is not a fixed number of seconds, but this
    function will return the average number of seconds in a month, accounting for leap years.

    TODO: consider replacing this function with the use of https://github.com/wroberts/pytimeparse

    Args:
        unit: The input unit to normalize.

    Returns:
        A tuple containing the canonical unit and the number of seconds in that unit.

    Raises:
        ValueError: If the input unit is not recognized.

    Examples:
        >>> normalize_time_unit("s")
        ('seconds', 1)
        >>> normalize_time_unit("min")
        ('minutes', 60)
        >>> normalize_time_unit("hours")
        ('hours', 3600)
        >>> normalize_time_unit("day")
        ('days', 86400)
        >>> normalize_time_unit("wks")
        ('weeks', 604800)
        >>> normalize_time_unit("month")
        ('months', 2629746.0)
        >>> normalize_time_unit("years")
        ('years', 31556926.080000002)
        >>> normalize_time_unit("fortnight")
        Traceback (most recent call last):
            ...
        ValueError: Unknown time unit 'fortnight'. Valid units include:
          * seconds: s, sec, secs, second, seconds
          * minutes: m, min, mins, minute, minutes
          * hours: h, hr, hrs, hour, hours
          * days: d, day, days
          * weeks: w, wk, wks, week, weeks
          * months: mo, mos, month, months
          * years: y, yr, yrs, year, years
    """
    for canonical_unit, (aliases, seconds) in TIME_DURATION_UNITS.items():
        if unit in aliases:
            return canonical_unit, seconds

    valid_unit_lines = []
    for canonical, (aliases, _) in TIME_DURATION_UNITS.items():
        valid_unit_lines.append(f"  * {canonical}: {', '.join(aliases)}")
    valid_units_str = "\n".join(valid_unit_lines)
    raise ValueError(f"Unknown time unit '{unit}'. Valid units include:\n{valid_units_str}")


def add_age_row_fntr(stage_cfg: DictConfig) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Return a function that adds an age row to the DataFrame for each patient."""
    canonical_unit, seconds_in_unit = normalize_time_unit(stage_cfg.age_unit)
    microseconds_in_unit = int(1e6) * seconds_in_unit

    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        if "subject_id" in df.columns:
            df = df.with_row_count("_idx")
            dob_df = df.filter(pl.col("code") == stage_cfg.DOB_code).select(["subject_id", "time", "_idx"])

            next_event_df = df.with_columns(
                [
                    pl.col("time").shift(-1).alias("next_time"),
                    pl.col("code").shift(-1).alias("next_code"),
                ]
            )

            dob_with_next = dob_df.join(next_event_df, on="_idx", how="left").filter(
                pl.col("next_code").is_not_null()
            )

            age_expr = (pl.col("next_time") - pl.col("time")).dt.total_microseconds() / microseconds_in_unit
            age_expr = age_expr.cast(pl.Float32, strict=False).round(0)

            one_second = pl.duration(seconds=1)
            adjusted_time_expr = (pl.col("next_time") - one_second).alias("time")

            code_expr = pl.format(f"{stage_cfg.age_code}//{{}}", age_expr.cast(pl.Int32)).alias("code")

            age_df = dob_with_next.select(
                [pl.col("subject_id"), adjusted_time_expr, code_expr, age_expr.alias("numeric_value")]
            )
            return age_df

    return fn


def add_age_column(df: pl.LazyFrame, cfg) -> pl.LazyFrame:
    """
    Add an age column to the DataFrame by subtracting the time of birth from the time of each observation.
    """
    canonical_unit, seconds_in_unit = normalize_time_unit(cfg.age_unit)
    microseconds_in_unit = int(1e6) * seconds_in_unit
    dob_times = df.filter(pl.col("code") == cfg.DOB_code).select(["subject_id", "time"])
    df = df.join(dob_times, on="subject_id")
    df = df.filter(pl.col("code") != cfg.DOB_code)
    df = df.with_columns(
        ((pl.col("time") - pl.col("time_right")).dt.total_microseconds() / microseconds_in_unit)
        .cast(pl.Int32, strict=False)
        .alias("age")
    )
    df = df.drop(["time_right"])
    return df


def add_age_fntr(stage_cfg: DictConfig) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Return a function that adds age of the patient."""
    compute_fns = []
    for feature_name, feature_cfg in stage_cfg.items():
        match feature_name:
            case "age_row":
                compute_fns.append(add_new_events_fntr(add_age_row_fntr(feature_cfg), feature_cfg))
            case str() if feature_name in INFERRED_STAGE_KEYS:
                continue
            case _:
                raise ValueError(f"Unknown time-derived measurement: {feature_name}")

        logger.info(f"Adding {feature_name} via config: {OmegaConf.to_yaml(feature_cfg)}")

    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        for compute_fn in compute_fns:
            df = compute_fn(df)
        return df

    return fn


@hydra.main(
    version_base=None, config_path=str(PREPROCESS_CONFIG_YAML.parent), config_name=PREPROCESS_CONFIG_YAML.stem
)
def main(cfg: DictConfig):
    """Adds time-derived measurements to a MEDS cohort as separate observations at each unique time."""
    map_over(cfg, compute_fn=add_age_fntr)


if __name__ == "__main__":  # pragma: no cover
    main()
