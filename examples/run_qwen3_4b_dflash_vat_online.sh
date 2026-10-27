#!/bin/bash
# Derived from SpecForge (https://github.com/sgl-project/SpecForge)
# MIT License
# Modified for VAT
#
# VAT training of the 5-layer DFlash draft for Qwen3-4B (paper settings),
# with the Hugging Face target model in every trainer process.
# Single node:  bash examples/run_qwen3_4b_dflash_vat_online.sh [NUM_GPUS]
# Multi-node:   NNODES=8 NODE_RANK=<rank> MASTER_ADDR=<host> MASTER_PORT=29500 \
#               bash examples/run_qwen3_4b_dflash_vat_online.sh 8

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
NUM_GPUS=${1:-8}

if [ -n "$NNODES" ]; then
    LAUNCH="--nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
else
    LAUNCH="--standalone"
fi

torchrun \
    $LAUNCH \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path Qwen/Qwen3-4B \
    --target-model-backend hf \
    --draft-config-path $ROOT_DIR/configs/qwen3-4b-dflash.json \
    --train-data-path $ROOT_DIR/cache/dataset/perfectblend_qwen3-4b_regen.jsonl \
    --on-the-fly-loader \
    --output-dir $ROOT_DIR/outputs/qwen3-4b-dflash-vat \
    --num-epochs 3 \
    --batch-size 1 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.045 \
    --max-grad-norm 1.0 \
    --max-length 4096 \
    --chat-template qwen \
    --attention-backend flex_attention \
    --block-size 16 \
    --num-anchors 512 \
    --loss-type vat \
    --loss-decay-gamma 7 \
    --vat-kd-weight 1.0 \
    --vat-kd-temperature 2.0 \
    --vat-accept-head-weight 1.0 \
    --vat-self-cov-weight 0.6 \
    --log-interval 50 \
    --save-interval 5000 \
    --report-to tensorboard
