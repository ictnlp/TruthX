export CUDA_VISIBLE_DEVICES=0

ROOT=path_to_truthx_dir
EXP_ROOT=$ROOT/results
model_path=path_to_llm # e.g. Llama-2-7b-chat-hf

# two-fold validation
truthx_model1=truthx_models/Llama-2-7b-chat-hf/truthx_model.fold1.pt
truthx_model2=truthx_models/Llama-2-7b-chat-hf/truthx_model.fold2.pt

strength=1.0
layers=10

python3  $ROOT/scripts/truthfulqa_generation_truthx.py \
    --model-path $model_path \
    --truthx-model $truthx_model1 \
    --truthx-model2 $truthx_model2 \
    --two-fold True \
    --data-yaml data/data_fold1.yaml \
    --edit-strength $strength --top-layers $layers  \
    --fewshot-prompting True \
    --output-file $EXP_ROOT/truthfulqa_generation_truthx/Llama-2-7b-chat-hf.truthx.jsonl 