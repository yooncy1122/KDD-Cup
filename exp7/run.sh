#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/../baseline:${SCRIPT_DIR}/../exp1:${PYTHONPATH}"

python3 -u "${SCRIPT_DIR}/train.py" \
    --dataset_module   dataset_reltime \
    --model_module     model7 \
    --schema_path      "${SCRIPT_DIR}/../exp1/schema_aligned.json" \
    --d_model          128 \
    --emb_dim          64 \
    --num_hyformer_blocks 3 \
    --num_heads        8 \
    --user_ns_tokens   4 \
    --item_ns_tokens   3 \
    --ns_groups_json   "" \
    --emb_skip_threshold 1000000 \
    --num_workers      8 \
    "$@"
