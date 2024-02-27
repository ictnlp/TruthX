export CUDA_VISIBLE_DEVICES=0

ROOT=path_to_truthx_dir
EXP_ROOT=$ROOT/results
model_path=path_to_llm #e.g. Llama-2-7b-chat-hf

python3  $ROOT/scripts/truthfulqa_generation.py \
    --model-path $model_path \
    --fewshot-prompting True \
    --output-file $EXP_ROOT/truthfulqa_generation/Llama-2-7b-chat-hf.jsonl 