#!/bin/bash

DATASET=$1
ARCH=$2

if [[ -z "$DATASET" || -z "$ARCH" ]]; then
    echo "Usage: bash generate_manova_report.sh <DATASET> <topogat|topogin>"
    exit 1
fi

echo "🚀 Starting MANOVA pipeline for $ARCH on $DATASET..."

# Create a directory to store logs
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${DATASET}_${ARCH}_manova.log"

# Start timer
START_TIME=$(date +%s)

# Run the full MANOVA pipeline and save logs
python scripts/run_manova_pipeline.py \
    --dataset "$DATASET" \
    --arch "$ARCH" \
    | tee "$LOG_FILE"

# End timer
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))

echo "✅ All steps complete for $DATASET ($ARCH)!"
echo "🕒 Total time taken: ${ELAPSED_TIME} seconds"
echo "🗂️  Log saved at: $LOG_FILE"
