#!/bin/bash
MODEL_PATH=$1
OUTPUT_DIR=$2
RUN_NAME=$3
STEPS=$4

export CUDA_VISIBLE_DEVICES=0,1,2

accelerate launch \
    --config_file ./configs/deepseed_zero3.yaml --num_processes 3 \
    step1_generator.py \
    --model_name_or_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps $STEPS \
    --save_steps 2 \
    --logging_steps 1 \
    --run_name "$RUN_NAME" \
    --number_of_generated_data 2000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-6 \
    --dtype bfloat16 \
    --max_completion_length 2048 \
    --log_completions \
    --num_generations 4 \
    --remove_unused_columns False \
    --loss_type grpo \
    --report_to wandb

pkill -f "step1_generator.py"
pkill python