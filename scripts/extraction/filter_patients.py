#!/usr/bin/env python

import json
import random
from functools import partial
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from MEDS_polars_functions.filter_patients_by_length import (
    filter_patients_by_num_events,
    filter_patients_by_num_measurements,
)
from MEDS_polars_functions.mapper import wrap as rwlock_wrap
from MEDS_polars_functions.utils import hydra_loguru_init, write_lazyframe


@hydra.main(version_base=None, config_path="../../configs", config_name="extraction")
def main(cfg: DictConfig):
    """Filter patients based on number of events."""

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
    
    input_dir = Path(cfg.stage_cfg.data_input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input cohort directory not found: {input_dir}")

    compute_fns = []
    if cfg.stage_cfg.min_measurements_per_patient:
        logger.info(
            f"Filtering patients with fewer than {cfg.stage_cfg.min_measurements_per_patient} measurements "
            "(observations of any kind)."
        )
        compute_fns.append(
            partial(
                filter_patients_by_num_measurements,
                min_measurements_per_patient=cfg.stage_cfg.min_measurements_per_patient,
            )
        )
    if cfg.stage_cfg.min_events_per_patient:
        logger.info(
            f"Filtering patients with fewer than {cfg.stage_cfg.min_events_per_patient} events "
            "(unique timepoints)."
        )
        compute_fns.append(
            partial(
                filter_patients_by_num_events, min_events_per_patient=cfg.stage_cfg.min_events_per_patient
            )
        )

    for sp in ["train"]:
        in_dir = input_dir / sp
        all_files = sorted(list(in_dir.glob("**/*.parquet")))

        for f in all_files:
            out_fp = Path(cfg.stage_cfg.output_dir) / sp / f.name
            logger.info(f"Filtering {str(f.resolve())} into {str(out_fp.resolve())}")

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
    logger.info("Filtered patients.")


if __name__ == "__main__":
    main()
