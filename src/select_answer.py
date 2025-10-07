import sys
import os
sys.path.append(".")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7" 
os.environ["WORLD_SIZE"] = "8"
os.environ['MASTER_PORT'] = '29203'
import logging
import torch
import transformers
import json
#dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
from torch.utils.data import DataLoader
from dataset import SelectAnswerDataset
from arguments import ModelArguments, DataArguments
from transformers import (
    HfArgumentParser,
)
def collate_fn(batch):
    # utility_prompt, formated_passages_ids, query_id, gt_answer, query
    # utility_prompt, formated_passages_ids, query_id, gt_answer
    utility_prompt = [item[0] for item in batch]
    formated_passages_ids = [item[1] for item in batch]
    query_id = [item[2] for item in batch]
    gt_answer = [item[3] for item in batch]
    query = [item[4] for item in batch]
    return utility_prompt, formated_passages_ids, query_id, gt_answer, query
if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
    if "llama" in model_args.model_name_or_path:
        sampling_params = SamplingParams(temperature=0.0, max_tokens=4096, stop = ["<|eot_id|>"])
    else:
        sampling_params = SamplingParams(temperature=0.0, max_tokens=4096)
    # sampling_params = SamplingParams(temperature=0.0, max_tokens=4096, stop = ["<|eot_id|>"])
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    llm = LLM(
        model=model_args.model_name_or_path,
        # dtype=torch.float16,
        # gpu_memory_utilization=0.8,
        tensor_parallel_size=8
    )
    dataset = SelectAnswerDataset(data_args, tokenizer)
    dataloader = DataLoader(dataset, batch_size=data_args.batch_size, shuffle=False, collate_fn=collate_fn)
    file_w = open(data_args.output_dir, "w", encoding="utf-8")
    for utility_prompts, formated_passages_ids, query_ids, gt_answers, querys in dataloader:  
        ress = []
        outputs = llm.generate(utility_prompts, sampling_params)
        for output in outputs:
            res = output.outputs[0].text
            ress.append(res)
        # print(query_ids)
        # for i in range(len(formated_passages_ids)):
        for i in range(len(query_ids)):
            file_w.write(json.dumps({
            "query_id": query_ids[i],
            "query": querys[i],
            "gt_answer": gt_answers[i],
            "answer_output": ress[i],
            })+"\n")