# llama-7B
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python examples/generate_act_scales.py \
    --model-name /datasets/llama-hf/llama-7b-meta/ \
    --output-path act_scales/llama-7b-meta.pt

# llama-13B
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python examples/generate_act_scales.py \
    --model-name /datasets/llama-hf/llama-13b-meta/ \
    --output-path act_scales/llama-13b-meta.pt

# llama-30B
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python examples/generate_act_scales.py \
    --model-name /datasets/llama-hf/llama-30b-meta/ \
    --output-path act_scales/llama-30b-meta.pt

# llama-65B
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python examples/generate_act_scales.py \
    --model-name /datasets/llama-hf/llama-65b-meta/ \
    --output-path act_scales/llama-65b-meta.pt

# opt-6.7B
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python examples/generate_act_scales.py \
    --model-name /datasets/opt/6.7b \
    --output-path act_scales/opt-6.7b.pt

# opt-13B
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python examples/generate_act_scales.py \
    --model-name /datasets/opt/13b \
    --output-path act_scales/opt-13b.pt

# opt-30B
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python examples/generate_act_scales.py \
    --model-name /datasets/opt/30b \
    --output-path act_scales/opt-30b.pt

# opt-66B
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python examples/generate_act_scales.py \
    --model-name /datasets/opt/66b \
    --output-path act_scales/opt-66b.pt