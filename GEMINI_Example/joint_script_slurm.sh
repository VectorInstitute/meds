#!/usr/bin/env bash

# This makes the script fail if any internal script fails
set -e

# Function to display help message
function display_help() {
    echo "Usage: $0 <GEMINI_PREMEDS_DIR> <GEMINI_MEDS_DIR> <N_PARALLEL_WORKERS>"
    echo
    echo "This script processes GEMINI data through several steps, handling raw data conversion,"
    echo "sharding events, splitting patients, converting to sharded events, and merging into a MEDS cohort."
    echo "This script uses slurm to process the data in parallel via the 'submitit' Hydra launcher."
    echo
    echo "Arguments:"
    # echo "  GEMINI_RAW_DIR        Directory containing raw MIMIC-IV data files."
    echo "  GEMINI_PREMEDS_DIR    Output directory for pre-MEDS data."
    echo "  GEMINI_MEDS_DIR       Output directory for processed MEDS data."
    echo "  N_PARALLEL_WORKERS     Number of parallel workers for processing."
    echo
    echo "Options:"
    echo "  -h, --help          Display this help message and exit."
    exit 1
}

# Check if the first parameter is '-h' or '--help'
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    display_help
fi

# Check for mandatory parameters
if [ "$#" -ne 3 ]; then
    echo "Error: Incorrect number of arguments provided."
    display_help
fi

# export GEMINI_RAW_DIR="$1"
export GEMINI_PREMEDS_DIR="$1"
export GEMINI_MEDS_DIR="$2"
export N_PARALLEL_WORKERS="$3"

shift 3

# Note we use `--multirun` throughout here due to ensure the submitit launcher is used throughout, so that
# this doesn't fall back on running anything locally in a setting where only slurm worker nodes have
# sufficient computational resources to run the actual jobs.


echo "Trying submitit launching with $N_PARALLEL_WORKERS jobs."
GEMINI_EVENT_CONFIGS=./GEMINI_Example/configs/event_configs.yaml

source ~/myenv/bin/activate

# ./scripts/extraction/shard_events.py \
#     --multirun \
#     worker="range(0,$N_PARALLEL_WORKERS)" \
#     hydra/launcher=submitit_slurm \
#     hydra.launcher.timeout_min=60 \
#     hydra.launcher.cpus_per_task=5 \
#     hydra.launcher.mem_gb=40 \
#     hydra.launcher.partition="gpu" \
#     "hydra.job.env_copy=[PATH]" \
#     input_dir="$GEMINI_PREMEDS_DIR" \
#     cohort_dir="$GEMINI_MEDS_DIR" \
#     event_conversion_config_fp="$GEMINI_EVENT_CONFIGS" \
#     stage=shard_events

# echo "Splitting patients on one worker"
# ./scripts/extraction/split_and_shard_patients.py \
#     --multirun \
#     worker="range(0,1)" \
#     hydra/launcher=submitit_slurm \
#     hydra.launcher.timeout_min=60 \
#     hydra.launcher.cpus_per_task=5 \
#     hydra.launcher.mem_gb=40 \
#     hydra.launcher.partition="gpu" \
#     input_dir="$GEMINI_PREMEDS_DIR" \
#     cohort_dir="$GEMINI_MEDS_DIR" \
#     event_conversion_config_fp="$GEMINI_EVENT_CONFIGS" "$@" \
#     stage=split_and_shard_patients

# echo "Converting to sharded events with $N_PARALLEL_WORKERS workers in parallel"
# ./scripts/extraction/convert_to_sharded_events.py \
#     --multirun \
#     worker="range(0,$N_PARALLEL_WORKERS)" \
#     hydra/launcher=submitit_slurm \
#     hydra.launcher.timeout_min=60 \
#     hydra.launcher.cpus_per_task=5 \
#     hydra.launcher.mem_gb=40 \
#     hydra.launcher.partition="gpu" \
#     input_dir="$GEMINI_PREMEDS_DIR" \
#     cohort_dir="$GEMINI_MEDS_DIR" \
#     event_conversion_config_fp="$GEMINI_EVENT_CONFIGS" "$@" \
#     stage=convert_to_sharded_events

# echo "Merging to a MEDS cohort with $N_PARALLEL_WORKERS workers in parallel"
# ./scripts/extraction/merge_to_MEDS_cohort.py \
#     --multirun \
#     worker="range(0,$N_PARALLEL_WORKERS)" \
#     hydra/launcher=submitit_slurm \
#     hydra.launcher.timeout_min=60 \
#     hydra.launcher.cpus_per_task=5 \
#     hydra.launcher.mem_gb=40 \
#     hydra.launcher.partition="gpu" \
#     input_dir="$GEMINI_PREMEDS_DIR" \
#     cohort_dir="$GEMINI_MEDS_DIR" \
#     event_conversion_config_fp="$GEMINI_EVENT_CONFIGS" "$@" \
#     stage=merge_to_MEDS_cohort


# echo "Filtering rows and cols with $N_PARALLEL_WORKERS workers in parallel"
# ./scripts/extraction/filter_rows_and_cols.py \
#     --multirun \
#     worker="range(0,$N_PARALLEL_WORKERS)" \
#     hydra/launcher=submitit_slurm \
#     hydra.launcher.timeout_min=60 \
#     hydra.launcher.cpus_per_task=5 \
#     hydra.launcher.mem_gb=40 \
#     hydra.launcher.partition="gpu" \
#     input_dir="$GEMINI_PREMEDS_DIR" \
#     cohort_dir="$GEMINI_MEDS_DIR" \
#     event_conversion_config_fp="$GEMINI_EVENT_CONFIGS" "$@" \
#     stage=filter_rows_and_cols
    

# echo "Filtering patients with $N_PARALLEL_WORKERS workers in parallel"
# ./scripts/extraction/filter_patients.py \
#     --multirun \
#     worker="range(0,$N_PARALLEL_WORKERS)" \
#     hydra/launcher=submitit_slurm \
#     hydra.launcher.timeout_min=60 \
#     hydra.launcher.cpus_per_task=5 \
#     hydra.launcher.mem_gb=40 \
#     hydra.launcher.partition="gpu" \
#     input_dir="$GEMINI_PREMEDS_DIR" \
#     cohort_dir="$GEMINI_MEDS_DIR" \
#     event_conversion_config_fp="$GEMINI_EVENT_CONFIGS" "$@" \
#     stage=filter_patients
    
# echo "Adding time intervals cohort with $N_PARALLEL_WORKERS workers in parallel"
# ./scripts/extraction/add_time_interval.py \
#     --multirun \
#     worker="range(0,$N_PARALLEL_WORKERS)" \
#     hydra/launcher=submitit_slurm \
#     hydra.launcher.timeout_min=60 \
#     hydra.launcher.cpus_per_task=5 \
#     hydra.launcher.mem_gb=40 \
#     hydra.launcher.partition="gpu" \
#     input_dir="$GEMINI_PREMEDS_DIR" \
#     cohort_dir="$GEMINI_MEDS_DIR" \
#     event_conversion_config_fp="$GEMINI_EVENT_CONFIGS" "$@" \
#     stage=add_time_interval
    
# echo "Adding REG column with $N_PARALLEL_WORKERS workers in parallel"
# ./scripts/extraction/add_reg.py \
#     --multirun \
#     worker="range(0,$N_PARALLEL_WORKERS)" \
#     hydra/launcher=submitit_slurm \
#     hydra.launcher.timeout_min=60 \
#     hydra.launcher.cpus_per_task=5 \
#     hydra.launcher.mem_gb=40 \
#     hydra.launcher.partition="gpu" \
#     input_dir="$GEMINI_PREMEDS_DIR" \
#     cohort_dir="$GEMINI_MEDS_DIR" \
#     event_conversion_config_fp="$GEMINI_EVENT_CONFIGS" "$@" \
#     stage=add_reg
    
# echo "Adding Embedding columns with $N_PARALLEL_WORKERS workers in parallel"
# ./scripts/extraction/add_embedding_columns.py \
#     --multirun \
#     worker="range(0,$N_PARALLEL_WORKERS)" \
#     hydra/launcher=submitit_slurm \
#     hydra.launcher.timeout_min=60 \
#     hydra.launcher.cpus_per_task=5 \
#     hydra.launcher.mem_gb=40 \
#     hydra.launcher.partition="gpu" \
#     input_dir="$GEMINI_PREMEDS_DIR" \
#     cohort_dir="$GEMINI_MEDS_DIR" \
#     event_conversion_config_fp="$GEMINI_EVENT_CONFIGS" "$@" \
#     stage=add_embedding_columns
    
# echo "Filtering labs with $N_PARALLEL_WORKERS workers in parallel"
# ./scripts/extraction/filter_labs.py \
#     --multirun \
#     worker="range(0,$N_PARALLEL_WORKERS)" \
#     hydra/launcher=submitit_slurm \
#     hydra.launcher.timeout_min=60 \
#     hydra.launcher.cpus_per_task=5 \
#     hydra.launcher.mem_gb=40 \
#     hydra.launcher.partition="gpu" \
#     input_dir="$GEMINI_PREMEDS_DIR" \
#     cohort_dir="$GEMINI_MEDS_DIR" \
#     event_conversion_config_fp="$GEMINI_EVENT_CONFIGS" "$@" \
#     stage=filter_labs
    
echo "Quantizing lab values with $N_PARALLEL_WORKERS workers in parallel"
./scripts/extraction/quantize_lab_values.py \
    --multirun \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    hydra/launcher=submitit_slurm \
    hydra.launcher.timeout_min=60 \
    hydra.launcher.cpus_per_task=5 \
    hydra.launcher.mem_gb=40 \
    hydra.launcher.partition="gpu" \
    input_dir="$GEMINI_PREMEDS_DIR" \
    cohort_dir="$GEMINI_MEDS_DIR" \
    event_conversion_config_fp="$GEMINI_EVENT_CONFIGS" "$@" \
    stage=quantize_lab_values
    
echo "Aggregating sequences with $N_PARALLEL_WORKERS workers in parallel"
./scripts/extraction/generate_sequence.py \
    --multirun \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    hydra/launcher=submitit_slurm \
    hydra.launcher.timeout_min=60 \
    hydra.launcher.cpus_per_task=5 \
    hydra.launcher.mem_gb=40 \
    hydra.launcher.partition="gpu" \
    input_dir="$GEMINI_PREMEDS_DIR" \
    cohort_dir="$GEMINI_MEDS_DIR" \
    event_conversion_config_fp="$GEMINI_EVENT_CONFIGS" "$@" \
    stage=generate_sequence
