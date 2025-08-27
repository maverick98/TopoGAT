#!/bin/bash

# Usage: ./all_datasets_runner.sh topogat
# or     ./all_datasets_runner.sh topogin

MODEL_FAMILY=$1

if [[ -z "$MODEL_FAMILY" ]]; then
  echo "Usage: $0 [topogat|topogin]"
  exit 1
fi

# Define the list of datasets
DATASETS=("MUTAG" "PTC_MR" "ENZYMES" "PROTEINS")

echo "📦 Model family: $MODEL_FAMILY"
echo "📊 Datasets to process: ${DATASETS[*]}"
echo

# Global timer
GLOBAL_START=$(date +%s)

# Create a logs directory
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

for dataset in "${DATASETS[@]}"; do
  echo "--------------------------------------------------------------------------------"
  echo "▶ Running MANOVA for dataset: $dataset with model family: $MODEL_FAMILY"
  
  START_TIME=$(date +%s)

  bash generate_manova_report.sh "$dataset" "$MODEL_FAMILY"

  END_TIME=$(date +%s)
  DURATION=$((END_TIME - START_TIME))

  echo "🕒 Time for $dataset: ${DURATION}s"
  echo "📄 Log: $LOG_DIR/${dataset}_${MODEL_FAMILY}_manova.log"
  echo "--------------------------------------------------------------------------------"
  echo
done

GLOBAL_END=$(date +%s)
TOTAL_TIME=$((GLOBAL_END - GLOBAL_START))

echo "✅ All MANOVA reports generated for model family: $MODEL_FAMILY"
echo "⏱️  Total time taken: ${TOTAL_TIME} seconds"
