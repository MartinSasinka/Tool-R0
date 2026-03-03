#!/bin/bash
MODEL_PATH=$1
DATA_PATH=$2
OUTPUT_DIR=$3
RUN_NAME=$4
STEPS=$5

export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch \
    --config_file ./configs/deepseed_zero3.yaml --num_processes 4 \
    step3_solver.py \
    --model_name_or_path "$MODEL_PATH" \
    --generated_data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps $STEPS \
    --save_steps 2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-6 \
    --dtype bfloat16 \
    --max_completion_length 2048 \
    --log_completions \
    --num_generations 4 \
    --remove_unused_columns False \
    --loss_type grpo \
    --logging_steps 1 \
    --run_name "$RUN_NAME" \
    --report_to wandb

pkill -f "step3_solver.py"