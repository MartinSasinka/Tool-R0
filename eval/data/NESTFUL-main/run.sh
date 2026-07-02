echo "#### RUNNING INFERENCE ####"    
python -u src/eval.py \
    --model "ibm-granite/granite-3.1-8b-instruct" \
    --model_name "granite-3.1-8b-instruct" \
    --save_directory "results" \
    --dataset "data_v2/nestful_data.jsonl" \
    --icl_count 3


echo "#### SCORING ####"    
python -u src/scorer.py \
    --model_name "granite-3.1-8b-instruct" \
    --result_file_path "results/nestful_3/granite-3.1-8b-instruct/output.jsonl" \
    --executable_func_dir "data_v2/executable_functions"
