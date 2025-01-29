#!/usr/bin/env bash

# This makes the script fail if any internal script fails
set -e

# Function to display help message
function display_help() {
    echo "Usage: $0 <MIMICIV_MEDS_EXTRACT_DIR> <MIMICIV_MEDS_PREP_DIR>"
    echo
    echo "This script processes MIMICIV data through several steps, handling raw data conversion,"
    echo "sharding events, splitting subjects, converting to sharded events, and merging into a MEDS cohort."
    echo
    echo "Arguments:"
    echo "  MIMICIV_MEDS_EXTRACT_DIR                            Output directory for extracted data."
    echo "  MIMICIV_MEDS_PREP_DIR                               Output directory for processed MEDS data."
    echo
    echo "Options:"
    echo "  -h, --help          Display this help message and exit."
    exit 1
}

echo "Unsetting SLURM_CPU_BIND in case you're running this on a slurm interactive node with slurm parallelism"
unset SLURM_CPU_BIND

# Check if the first parameter is '-h' or '--help'
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    display_help
fi

# Check for mandatory parameters
if [ "$#" -lt 2 ]; then
    echo "Error: Incorrect number of arguments provided."
    display_help
fi

export MIMICIV_MEDS_EXTRACT_DIR=$1
export MIMICIV_MEDS_PREP_DIR=$2
shift 2

EVENT_CONVERSION_CONFIG_FP="$(pwd)/configs/event_configs_seq.yaml"
PIPELINE_CONFIG_FP="$(pwd)/configs/preprocess_MIMIC_seq.yaml"
export N_WORKERS=4

# We export these variables separately from their assignment so that any errors during assignment are caught.
export EVENT_CONVERSION_CONFIG_FP
export PIPELINE_CONFIG_FP

export HYDRA_FULL_ERROR=1

if [ -z "$N_WORKERS" ]; then
  echo "Setting N_WORKERS to 1 to avoid issues with the runners."
  export N_WORKERS="1"
fi

export VISIT_ID_COLUMN=genc_id

echo "Running preprocessing pipeline."
MEDS_transform-runner "pipeline_config_fp=$PIPELINE_CONFIG_FP" "$@"
