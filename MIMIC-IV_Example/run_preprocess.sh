#!/usr/bin/env bash

set -e  # Exit script on error

# Function to display help message
display_help() {
    cat <<EOF
Usage: $0 <MIMICIV_MEDS_EXTRACT_DIR> <MIMICIV_MEDS_PREP_DIR> [--qa-data]

This script processes MIMICIV data through multiple steps:

This script processes MIMICIV data through multiple steps:

For sequence processing, the steps are:
  - filter_codes
  - filter_subjects
  - filter_labs
  - filter_meds
  - update_transfers
  - add_age
  - add_cls_token
  - quantize_labs
  - add_time_tokens
  - generate_sequence

For QA processing, the steps are:
  - filter_labs
  - filter_meds
  - quantize_labs

Arguments:
  MIMICIV_MEDS_EXTRACT_DIR   Output directory for extracted data.
  MIMICIV_MEDS_PREP_DIR      Output directory for processed MEDS data.

Options:
  --qa-data                 Use QA-specific configuration files instead of sequential ones.
  -h, --help                Display this help message and exit.
EOF
    exit 1
}

# Unset SLURM settings if running on a Slurm node
unset SLURM_CPU_BIND

# Handle help option
[[ "$1" == "-h" || "$1" == "--help" ]] && display_help

# Check for mandatory parameters
if [[ $# -lt 2 ]]; then
    echo "Error: Incorrect number of arguments provided."
    display_help
fi

export MIMICIV_MEDS_EXTRACT_DIR=$1
export MIMICIV_MEDS_PREP_DIR=$2
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIMIC_DIR="$(realpath "$SCRIPT_DIR/")"
export MIMIC_DIR

# Default to sequential configuration unless --qa-data is passed
SUFFIX="seq"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --qa-data)
            SUFFIX="qa"
            ;;
        *)
            echo "Error: Unknown argument '$1'"
            display_help
            ;;
    esac
    shift
done

# Configuration files
EVENT_CONVERSION_CONFIG_FP="$(pwd)/configs/event_configs_${SUFFIX}.yaml"
export EVENT_CONVERSION_CONFIG_FP

PIPELINE_CONFIG_FP="$(pwd)/configs/preprocess_MIMIC_${SUFFIX}.yaml"
export PIPELINE_CONFIG_FP

# Set default number of workers if not already set
export N_WORKERS=${N_WORKERS:-4}
export VISIT_ID_COLUMN=genc_id
export HYDRA_FULL_ERROR=1

# Run preprocessing pipeline
echo "Running preprocessing pipeline."
MEDS_transform-runner "pipeline_config_fp=$PIPELINE_CONFIG_FP"
