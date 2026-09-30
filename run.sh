#!/bin/sh
# Пример: ./run.sh q8 context --lengths 1024 2048 4096 8192 16384 32768
set -eu
if [ "$#" -lt 2 ]; then
    echo "Использование: ./run.sh q4|q8 smoke|speed|context [--lengths ...] [--plan]"
    exit 1
fi
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
model=$1
mode=$2
shift 2
cd "$project_dir"
exec ./scripts/python.sh scripts/benchmark.py --model "$model" --mode "$mode" "$@"
