#!/bin/bash

# Default directory to current folder if not passed
ROOT_DIR=${1:-$(pwd)}
OUTPUT_FILE="project_structure.txt"

echo "Project structure under: $ROOT_DIR"
echo "---------------------------------" | tee "$OUTPUT_FILE"

find "$ROOT_DIR" \
  \( -type d \( -name "__pycache__" -o -name "topogat-env" -o -name "experimental_results" \) -prune \) -o \
  -type f ! -name "*.png" ! -path "*/.*" -print | \
awk -v root="$ROOT_DIR" '
{
  sub(root"/", "", $0)
  n = gsub("/", "/", $0)
  indent = ""
  for (i = 0; i < n; i++) indent = indent "│   "
  print indent "├── " $0
}' | tee -a "$OUTPUT_FILE"

echo -e "\nSaved to $OUTPUT_FILE"
