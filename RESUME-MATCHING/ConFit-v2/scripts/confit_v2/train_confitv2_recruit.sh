#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-./}"

DATASET_DIR="${DATASET_DIR:-dataset/recruiting_data_v2}"
SAVE_PATH="${SAVE_PATH:-model_checkpoints/confit_v2}"
PRECISION="${PRECISION:-bf16}"
HARD_NEGATIVE_PATH="${HARD_NEGATIVE_PATH:-}"

extra_args=()
if [[ -n "$HARD_NEGATIVE_PATH" ]]; then
  extra_args+=(--hard_negative_path "$HARD_NEGATIVE_PATH")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python runners/trainer/train_confit.py \
--save_path "$SAVE_PATH" \
--resume_data_path "$DATASET_DIR/all_resume_full_desensitized_recruiting_data.csv" \
--job_data_path "$DATASET_DIR/all_jd_full_from_text_recruiting_data.csv" \
--train_label_path "$DATASET_DIR/train_labeled_data.jsonl" \
--num_resume_features 1 \
--num_job_features 1 \
--valid_label_path "$DATASET_DIR/valid_classification_data.jsonl" \
--classification_data_path "$DATASET_DIR/test_classification_data.jsonl" \
--rank_resume_data_path "$DATASET_DIR/rank_resume.json" \
--rank_job_data_path "$DATASET_DIR/rank_job.json" \
--dataset_type recruiting_data_v2 \
--train_batch_size 2 \
--num_hard_negatives 2 \
--val_batch_size 2 \
--gradient_accumulation_steps 4 \
--lr 1e-5 \
--weight_decay 1e-2 \
--num_warmup_steps 0.05 \
--num_suit_encoder_layers 0 \
--num_encoder_layers 0 \
--encode_all True \
--embedding_method mean_pool \
--finegrained_loss noop \
--pretrained_encoder jinaai/jina-embeddings-v2-base-zh \
--do_normalize true \
--temperature 0.05 \
--do_both_rj_hard_neg true \
--precision "$PRECISION" \
"${extra_args[@]}" \
--strategy deepspeed_stage_2 \
--max_epochs 10
