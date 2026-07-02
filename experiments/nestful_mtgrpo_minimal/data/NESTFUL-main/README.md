# NESTFUL
This is the official repository for `NESTFUL`.
- Paper Title: **_NESTFUL: A Benchmark for Evaluating LLMs on Nested Sequences of API Calls_**
- Link: https://arxiv.org/abs/2409.03797v3
- HuggingFace Data Link: https://huggingface.co/datasets/ibm-research/nestful

## Data
We have shared the latest NESTFUL evaluation set under `data_v2` dir.
- `nestful_data.jsonl`: It has 1861 evaluation data for nested sequencing.
- `executable_functions`: Contains the implementation of all the functions in the benchmark.

The `data_v1` directory includes the data for the previous version of the paper - [link](https://arxiv.org/abs/2409.03797v1).
- `executable`: contains data and spec with necessary information to execute them through RapidAPI.
- `non-executable`: contains the nested sequencing data from SGD and GLAIVE that are hand-picked by human annotators from data synthetically generated using an LLM.

## Installation

```bash
# Create a new Conda environment with Python 3.10
conda create --name nestful python=3.10
conda activate nestful

# Install necessary dependencies
pip install vllm==0.6.2
pip install scikit-learn jsonlines
```

## Evaluation
Evaluation consists of two main stages: **Inference Generation** and **Scoring**. You can execute both stages together by running `sh run.sh`. Refer to the configuration section below to modify `run.sh` for different settings or models.

### Inference Generation
```bash
python -u src/eval.py \
    --model "ibm-granite/granite-3.1-8b-instruct" \              # Hugging Face model path or identifier
    --model_name "granite-3.1-8b-instruct" \                     # Used for selecting prompt and output parser
    --save_directory "results" \                                 # Directory to store the output files
    --dataset "data_v2/nestful_data.jsonl" \                     # Path to the dataset file in JSONL format
    --icl_count 3                                                # Number of in-context examples to use (e.g., 3)
```

### Scoring
```bash
python -u src/scorer.py \
    --model_name "granite-3.1-8b-instruct" \                          # Name of the model used for generating predictions
    --result_file_path "results/nestful_3/granite-3.1-8b-instruct/output.jsonl" \  # Path to the model output JSONL file
    --executable_func_dir "data_v2/executable_functions"             # Directory containing executable function definitions
```




