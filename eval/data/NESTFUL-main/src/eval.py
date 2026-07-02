import os, json, argparse
from utils import *
from instruct_data_prep import get_instruct_data
import gc
import torch
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)

def eval_code(args):
    print("### Loading Data...")
    
    data = read_jsonlines(args.dataset)

    for i in range(len(data)):
        data[i]["tools"] = json.dumps(data[i]["tools"])
        data[i]["gold_answer"] = json.dumps(data[i]["gold_answer"])
        data[i]["output"] = json.dumps(data[i]["output"])

    print("### Preparing Instruct Data...")
    instruct_data = get_instruct_data(data, args.model, args.model_name, args.icl_count)

    print("### Loading Model...")
    llm = LLM(
        model=args.model, 
        tensor_parallel_size=torch.cuda.device_count(), 
        disable_custom_all_reduce=True
    )
        
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    prompts = [sample["input"] for sample in instruct_data]

    print("### Starting Generation...")
    response, output_list = [], []
    count_total_batches = -(-len(prompts) // args.batch_size)
    for idx in range(0, len(prompts), args.batch_size):
        print(f"### At batch {idx // args.batch_size + 1} out of {count_total_batches} batches...")
        prompt_batch = prompts[idx : idx + args.batch_size]
        batch_output = llm.generate(prompt_batch, sampling_params)

        for output in batch_output:
            response.append(output.outputs[0].text.strip())

    for idx in range(len(response)):
        temp = instruct_data[idx]
        temp["generated_text"] = response[idx]
        output_list.append(temp)

    # unload model from GPU
    destroy_model_parallel()
    destroy_distributed_environment()
    del llm.llm_engine.model_executor
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    print("### Saving...")
    save_path = os.path.join(args.save_directory, f"nestful_{args.icl_count}", args.model_name, "output.jsonl")
    print(f"### Save Path: {save_path}")
    os.makedirs(os.path.join(args.save_directory, f"nestful_{args.icl_count}", args.model_name), exist_ok=True)
    write_jsonlines(output_list, save_path)

    print(f"### DONE...!!!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--save_directory", type=str, default='results')
    parser.add_argument("--dataset", type=str, default="ibm-research/nestful")
    parser.add_argument("--icl_count", default=3, type=int)  
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    
    args = parser.parse_args()
    print(args)
    eval_code(args)
