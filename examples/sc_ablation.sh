#!/bin/bash

LOG_DIR=/home/xuechenhao/smoothquant/log

# No quantization

# CUDA_VISIBLE_DEVICES=2 python smoothquant/ppl_eval.py \
#     --alpha 0.85 \
#     --model_path /datasets/opt/6.7b \
#     --act_scales_path act_scales/opt-6.7b.pt \
#     2>&1 | tee $LOG_DIR/no_quant-no_sc.log &

# # Only quantization

# CUDA_VISIBLE_DEVICES=3 python smoothquant/ppl_eval.py \
#     --alpha 0.85 \
#     --model_path /datasets/opt/6.7b \
#     --act_scales_path act_scales/opt-6.7b.pt \
#     --smooth \
#     --quantize \
#     2>&1 | tee $LOG_DIR/quant-no_sc.log &

# wait

# linear SC errors

CUDA_VISIBLE_DEVICES=2 python smoothquant/ppl_eval.py \
    --alpha 0.85 \
    --model_path /datasets/opt/6.7b \
    --act_scales_path act_scales/opt-6.7b.pt \
    --smooth \
    --quantize \
    --sc_mu 0.0103 \
    --sc_sigma 0.1577 \
    --linear_sc_error \
    2>&1 | tee $LOG_DIR/quant-sc_linear-mac_32.log &

CUDA_VISIBLE_DEVICES=3 python smoothquant/ppl_eval.py \
    --alpha 0.85 \
    --model_path /datasets/opt/6.7b \
    --act_scales_path act_scales/opt-6.7b.pt \
    --smooth \
    --quantize \
    --sc_mu 0.0117 \
    --sc_sigma 0.1551 \
    --linear_sc_error \
    2>&1 | tee $LOG_DIR/quant-sc_linear-mac_64.log &

CUDA_VISIBLE_DEVICES=4 python smoothquant/ppl_eval.py \
    --alpha 0.85 \
    --model_path /datasets/opt/6.7b \
    --act_scales_path act_scales/opt-6.7b.pt \
    --smooth \
    --quantize \
    --sc_mu 0.0133 \
    --sc_sigma 0.1606 \
    --linear_sc_error \
    2>&1 | tee $LOG_DIR/quant-sc_linear-mac_128.log &

CUDA_VISIBLE_DEVICES=5 python smoothquant/ppl_eval.py \
    --alpha 0.85 \
    --model_path /datasets/opt/6.7b \
    --act_scales_path act_scales/opt-6.7b.pt \
    --smooth \
    --quantize \
    --sc_mu 0.0150 \
    --sc_sigma 0.1547 \
    --linear_sc_error \
    2>&1 | tee $LOG_DIR/quant-sc_linear-mac_256.log &

CUDA_VISIBLE_DEVICES=6 python smoothquant/ppl_eval.py \
    --alpha 0.85 \
    --model_path /datasets/opt/6.7b \
    --act_scales_path act_scales/opt-6.7b.pt \
    --smooth \
    --quantize \
    --sc_mu 0.0153 \
    --sc_sigma 0.1596 \
    --linear_sc_error \
    2>&1 | tee $LOG_DIR/quant-sc_linear-mac_512.log &

CUDA_VISIBLE_DEVICES=7 python smoothquant/ppl_eval.py \
    --alpha 0.85 \
    --model_path /datasets/opt/6.7b \
    --act_scales_path act_scales/opt-6.7b.pt \
    --smooth \
    --quantize \
    --sc_mu 0.0238 \
    --sc_sigma 0.1502 \
    --linear_sc_error \
    2>&1 | tee $LOG_DIR/quant-sc_linear-mac_4096.log &

wait


# pv SC errors

CUDA_VISIBLE_DEVICES=2 python smoothquant/ppl_eval.py \
    --alpha 0.85 \
    --model_path /datasets/opt/6.7b \
    --act_scales_path act_scales/opt-6.7b.pt \
    --smooth \
    --quantize \
    --sc_mu 0.0103 \
    --sc_sigma 0.1577 \
    --pv_sc_error \
    2>&1 | tee $LOG_DIR/quant-sc_pv-mac_32.log &

CUDA_VISIBLE_DEVICES=3 python smoothquant/ppl_eval.py \
    --alpha 0.85 \
    --model_path /datasets/opt/6.7b \
    --act_scales_path act_scales/opt-6.7b.pt \
    --smooth \
    --quantize \
    --sc_mu 0.0117 \
    --sc_sigma 0.1551 \
    --pv_sc_error \
    2>&1 | tee $LOG_DIR/quant-sc_pv-mac_64.log &

CUDA_VISIBLE_DEVICES=4 python smoothquant/ppl_eval.py \
    --alpha 0.85 \
    --model_path /datasets/opt/6.7b \
    --act_scales_path act_scales/opt-6.7b.pt \
    --smooth \
    --quantize \
    --sc_mu 0.0133 \
    --sc_sigma 0.1606 \
    --pv_sc_error \
    2>&1 | tee $LOG_DIR/quant-sc_pv-mac_128.log &

CUDA_VISIBLE_DEVICES=5 python smoothquant/ppl_eval.py \
    --alpha 0.85 \
    --model_path /datasets/opt/6.7b \
    --act_scales_path act_scales/opt-6.7b.pt \
    --smooth \
    --quantize \
    --sc_mu 0.0150 \
    --sc_sigma 0.1547 \
    --pv_sc_error \
    2>&1 | tee $LOG_DIR/quant-sc_pv-mac_256.log &

CUDA_VISIBLE_DEVICES=6 python smoothquant/ppl_eval.py \
    --alpha 0.85 \
    --model_path /datasets/opt/6.7b \
    --act_scales_path act_scales/opt-6.7b.pt \
    --smooth \
    --quantize \
    --sc_mu 0.0153 \
    --sc_sigma 0.1596 \
    --pv_sc_error \
    2>&1 | tee $LOG_DIR/quant-sc_pv-mac_512.log &

CUDA_VISIBLE_DEVICES=7 python smoothquant/ppl_eval.py \
    --alpha 0.85 \
    --model_path /datasets/opt/6.7b \
    --act_scales_path act_scales/opt-6.7b.pt \
    --smooth \
    --quantize \
    --sc_mu 0.0238 \
    --sc_sigma 0.1502 \
    --pv_sc_error \
    2>&1 | tee $LOG_DIR/quant-sc_pv-mac_4096.log &

wait



# qk SC errors

CUDA_VISIBLE_DEVICES=2 python smoothquant/ppl_eval.py \
    --alpha 0.85 \
    --model_path /datasets/opt/6.7b \
    --act_scales_path act_scales/opt-6.7b.pt \
    --smooth \
    --quantize \
    --sc_mu 0.0103 \
    --sc_sigma 0.1577 \
    --qk_sc_error \
    2>&1 | tee $LOG_DIR/quant-sc_qk-mac_32.log &

CUDA_VISIBLE_DEVICES=3 python smoothquant/ppl_eval.py \
    --alpha 0.85 \
    --model_path /datasets/opt/6.7b \
    --act_scales_path act_scales/opt-6.7b.pt \
    --smooth \
    --quantize \
    --sc_mu 0.0117 \
    --sc_sigma 0.1551 \
    --qk_sc_error \
    2>&1 | tee $LOG_DIR/quant-sc_qk-mac_64.log &

CUDA_VISIBLE_DEVICES=4 python smoothquant/ppl_eval.py \
    --alpha 0.85 \
    --model_path /datasets/opt/6.7b \
    --act_scales_path act_scales/opt-6.7b.pt \
    --smooth \
    --quantize \
    --sc_mu 0.0133 \
    --sc_sigma 0.1606 \
    --qk_sc_error \
    2>&1 | tee $LOG_DIR/quant-sc_qk-mac_128.log &

CUDA_VISIBLE_DEVICES=5 python smoothquant/ppl_eval.py \
    --alpha 0.85 \
    --model_path /datasets/opt/6.7b \
    --act_scales_path act_scales/opt-6.7b.pt \
    --smooth \
    --quantize \
    --sc_mu 0.0150 \
    --sc_sigma 0.1547 \
    --qk_sc_error \
    2>&1 | tee $LOG_DIR/quant-sc_qk-mac_256.log &

CUDA_VISIBLE_DEVICES=6 python smoothquant/ppl_eval.py \
    --alpha 0.85 \
    --model_path /datasets/opt/6.7b \
    --act_scales_path act_scales/opt-6.7b.pt \
    --smooth \
    --quantize \
    --sc_mu 0.0153 \
    --sc_sigma 0.1596 \
    --qk_sc_error \
    2>&1 | tee $LOG_DIR/quant-sc_qk-mac_512.log &

CUDA_VISIBLE_DEVICES=7 python smoothquant/ppl_eval.py \
    --alpha 0.85 \
    --model_path /datasets/opt/6.7b \
    --act_scales_path act_scales/opt-6.7b.pt \
    --smooth \
    --quantize \
    --sc_mu 0.0238 \
    --sc_sigma 0.1502 \
    --qk_sc_error \
    2>&1 | tee $LOG_DIR/quant-sc_qk-mac_4096.log &

wait


