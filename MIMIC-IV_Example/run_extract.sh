#!/usr/bin/env bash

set -e  # Exit script on error

# Function to display help message
display_help() {
    cat <<EOF
Usage: $0 <MIMICIV_RAW_DIR> <MIMICIV_PRE_MEDS_DIR> <MIMICIV_MEDS_EXTRACT_DIR> [options]

This script processes MIMIC-IV data through multiple steps:
  - Batches files (if specified)
  - Converts raw data
  - Shards events
  - Splits subjects
  - Converts to sharded events
  - Merges into a MEDS cohort

Arguments:
  MIMICIV_RAW_DIR          Directory containing raw MIMIC-IV data files.
  MIMICIV_PRE_MEDS_DIR     Output directory for pre-MEDS data.
  MIMICIV_MEDS_EXTRACT_DIR Output directory for processed MEDS data.

Options:
  do_unzip=true|false      (Optional) Unzip CSV files before processing (default: false).
  batch_files              Run batch_files.py before processing (requires extra args).
  --lab_input=<path>       (Required if batch_files is set) Path to labevents CSV.
  --chart_input=<path>     (Required if batch_files is set) Path to chartevents CSV.
  --qa-data               (Optional) If provided, use 'qa' suffixed configuration files.
  -h, --help              Show this help message and exit.
EOF
    exit 0
}

# Unset SLURM settings if running on a Slurm node
unset SLURM_CPU_BIND

# Environment variables
export POLARS_SKIP_CPU_CHECK=1
export HYDRA_FULL_ERROR=1
export N_WORKERS=1

# Check if the first parameter is '-h' or '--help'
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    display_help
fi

# Check for mandatory parameters
if [ "$#" -lt 3 ]; then
    echo "Error: Incorrect number of arguments provided."
    display_help
fi

export MIMICIV_RAW_DIR=$1
export MIMICIV_PRE_MEDS_DIR=$2
export MIMICIV_MEDS_EXTRACT_DIR=$3
shift 3

# Defaults
DO_UNZIP="false"
RUN_BATCH_FILES="false"
LAB_INPUT_FILE=""
LAB_OUTPUT_DIR="$MIMICIV_RAW_DIR/hosp"
CHART_INPUT_FILE=""
CHART_OUTPUT_DIR="$MIMICIV_RAW_DIR/icu"
QA_DATA="false"

# Process optional arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        batch_files)
            RUN_BATCH_FILES="true"
            ;;
        do_unzip=true|do_unzip=false)
            DO_UNZIP="${1#*=}"
            ;;
        --lab_input=*)
            LAB_INPUT_FILE="${1#*=}"
            ;;
        --lab_output=*)
            LAB_OUTPUT_DIR="${1#*=}"
            ;;
        --chart_input=*)
            CHART_INPUT_FILE="${1#*=}"
            ;;
        --chart_output=*)
            CHART_OUTPUT_DIR="${1#*=}"
            ;;
        --qa-data)
            QA_DATA="true"
            ;;
        *)
            echo "Error: Unknown argument '$1'"
            display_help
            ;;
    esac
    shift
done

# Determine file suffix based on qa-data flag
SUFFIX="seq"
if [ "$QA_DATA" == "true" ]; then
    SUFFIX="qa"
fi

# File paths
EVENT_CONVERSION_CONFIG_FP="$(pwd)/configs/event_configs_${SUFFIX}.yaml"
export EVENT_CONVERSION_CONFIG_FP

PIPELINE_CONFIG_FP="$(pwd)/configs/extract_MIMIC.yaml"
export PIPELINE_CONFIG_FP

PRE_MEDS_PY_FP="$(pwd)/pre_MEDS_${SUFFIX}.py"
export PRE_MEDS_PY_FP

BATCH_FILES_PY_FP="$(pwd)/batch_files.py"
export BATCH_FILES_PY_FP

# Function to unzip files
function unzip_files() {
    local gz_files="${MIMICIV_RAW_DIR}/*/*.csv.gz"
    if compgen -G "$gz_files" > /dev/null; then
        echo "Unzipping CSV files..."
        find "$MIMICIV_RAW_DIR" -type f -name "*.csv.gz" -exec gzip -d --force {} +
    else
        echo "No .csv.gz files found to unzip."
    fi
}

# Unzip files if requested
if [ "$DO_UNZIP" == "true" ]; then
    unzip_files
else
    echo "Skipping file unzipping."
fi

# Function to batch files
function run_batch_files() {
    echo "Running batch_files.py..."
    python "$BATCH_FILES_PY_FP" --type both \
        --lab_input_file "$LAB_INPUT_FILE" \
        --lab_output_dir "$LAB_OUTPUT_DIR" \
        --chart_input_file "$CHART_INPUT_FILE" \
        --chart_output_dir "$CHART_OUTPUT_DIR" \
        --rows_per_parquet 10000000
}

# Run batch_files.py if requested
if [ "$RUN_BATCH_FILES" == "true" ]; then
    run_batch_files
fi

# Run pre-MEDS conversion
echo "Running pre-MEDS conversion..."
python "$PRE_MEDS_PY_FP" input_dir="$MIMICIV_RAW_DIR" cohort_dir="$MIMICIV_PRE_MEDS_DIR"

# Ensure N_WORKERS is set
export N_WORKERS="${N_WORKERS:-1}"

# Run extraction pipeline
echo "Running extraction pipeline..."
MEDS_transform-runner "pipeline_config_fp=$PIPELINE_CONFIG_FP"
