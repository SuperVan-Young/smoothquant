# # llama-7B
# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python smoothquant/ppl_eval.py \
#     --alpha 0.85 \
#     --model_path /datasets/llama-hf/llama-7b-meta/ \
#     --act_scales_path act_scales/llama-7b-meta.pt \
#     --smooth \
#     --quantize

# # llama-13B
# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python smoothquant/ppl_eval.py \
#     --alpha 0.85 \
#     --model_path /datasets/llama-hf/llama-13b-meta/ \
#     --act_scales_path act_scales/llama-13b-meta.pt \
#     --smooth \
#     --quantize

# # llama-30B
# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python smoothquant/ppl_eval.py \
#     --alpha 0.85 \
#     --model_path /datasets/llama-hf/llama-30b-meta/ \
#     --act_scales_path act_scales/llama-30b-meta.pt \
#     --smooth \
#     --quantize

# # llama-65B
# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python smoothquant/ppl_eval.py \
#     --alpha 0.85 \
#     --model_path /datasets/llama-hf/llama-65b-meta/ \
#     --act_scales_path act_scales/llama-65b-meta.pt \
#     --smooth \
#     --quantize

# opt-6.7B
CUDA_VISIBLE_DEVICES=2 python smoothquant/ppl_eval.py \
    --alpha 0.85 \
    --model_path /datasets/opt/6.7b \
    --act_scales_path act_scales/opt-6.7b.pt \
    --smooth \
    --quantize

# opt-13B
CUDA_VISIBLE_DEVICES=2,3 python smoothquant/ppl_eval.py \
    --alpha 0.85 \
    --model_path /datasets/opt/13b \
    --act_scales_path act_scales/opt-13b.pt \
    --smooth \
    --quantize

# opt-30B
CUDA_VISIBLE_DEVICES=2,3 python smoothquant/ppl_eval.py \
    --alpha 0.85 \
    --model_path /datasets/opt/30b \
    --act_scales_path act_scales/opt-30b.pt \
    --smooth \
    --quantize

# opt-66B
CUDA_VISIBLE_DEVICES=2,3,4,5 python smoothquant/ppl_eval.py \
    --alpha 0.85 \
    --model_path /datasets/opt/66b \
    --act_scales_path act_scales/opt-66b.pt \
    --smooth \
    --quantize

# # Llama-2-7B
# CUDA_VISIBLE_DEVICES=0 python smoothquant/ppl_eval.py \
#     --alpha 0.85 \
#     --model_path meta-llama/Llama-2-7b-hf \
#     --act_scales_path act_scales/llama-2-7b.pt \
#     --smooth \
#     --quantize

# # Llama-2-13B
# CUDA_VISIBLE_DEVICES=0 python smoothquant/ppl_eval.py \
#     --alpha 0.85 \
#     --model_path meta-llama/Llama-2-13b-hf \
#     --act_scales_path act_scales/llama-2-13b.pt \
#     --smooth \
#     --quantize

# # Llama-2-70B
# CUDA_VISIBLE_DEVICES=0,1,2 python smoothquant/ppl_eval.py \
#     --alpha 0.9 \
#     --model_path meta-llama/Llama-2-70b-hf \
#     --act_scales_path act_scales/llama-2-70b.pt \
#     --smooth \
#     --quantize

# # Mistral-7B
# CUDA_VISIBLE_DEVICES=0 python smoothquant/ppl_eval.py \
#     --alpha 0.8 \
#     --model_path mistralai/Mistral-7B-v0.1 \
#     --act_scales_path act_scales/Mistral-7B-v0.1.pt \
#     --smooth \
#     --quantize

# # Mixtral-8x7B
# CUDA_VISIBLE_DEVICES=0,1 python smoothquant/ppl_eval.py \
#     --alpha 0.8 \
#     --model_path mistralai/Mixtral-8x7B-v0.1 \
#     --act_scales_path act_scales/Mixtral-8x7B-v0.1.pt \
#     --smooth \
#     --quantize

# # Falcon-7B
# CUDA_VISIBLE_DEVICES=0 python smoothquant/ppl_eval.py \
#     --alpha 0.6 \
#     --model_path tiiuae/falcon-7b \
#     --act_scales_path act_scales/falcon-7b.pt \
#     --smooth \
#     --quantize

# # Falcon-40B
# CUDA_VISIBLE_DEVICES=0,1 python smoothquant/ppl_eval.py \
#     --alpha 0.7 \
#     --model_path tiiuae/falcon-40b \
#     --act_scales_path act_scales/falcon-40b.pt \
#     --smooth \
#     --quantize

# # Meta-Llama-3-8B
# CUDA_VISIBLE_DEVICES=0 python smoothquant/ppl_eval.py \
#     --alpha 0.85 \
#     --model_path meta-llama/Meta-Llama-3-8B \
#     --act_scales_path act_scales/Meta-Llama-3-8B.pt \
#     --smooth \
#     --quantize

# # Meta-Llama-3-70B
# CUDA_VISIBLE_DEVICES=0,1 python smoothquant/ppl_eval.py \
#     --alpha 0.85 \
#     --model_path meta-llama/Meta-Llama-3-70B \
#     --act_scales_path act_scales/Meta-Llama-3-70B.pt \
#     --smooth \
#     --quantize
