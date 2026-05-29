#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASELINE_DIR="$(cd "${SCRIPT_DIR}/../baseline" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${BASELINE_DIR}:${PYTHONPATH}"

python3 -u "${BASELINE_DIR}/train.py" \
    --dataset_module dataset_weighted \
    --model_module  model_weighted \
    --schema_path "${SCRIPT_DIR}/schema_aligned.json" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    "$@"
