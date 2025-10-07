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
from dataset import PointEncodeDataset
from arguments import ModelArguments, DataArguments
from transformers import (
    HfArgumentParser,
)
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
        dtype=torch.float16,
        # gpu_memory_utilization=0.8,
        tensor_parallel_size=8
    )
    # llm = LLM(model=model_args.model_name_or_path, tensor_parallel_size=8)
    dataset = PointEncodeDataset(data_args, tokenizer)
    # dataloader = DataLoader(dataset, batch_size=data_args.batch_size, shuffle=False, collate_fn=collate_fn)
    file_w = open(data_args.output_dir, "w", encoding="utf-8")
    for utility_prompts, formated_passages_ids, query_id, gt_answer, query, passages in tqdm(dataset):  # 使用tqdm显示进度条 
        ress = []
        outputs = llm.generate(utility_prompts, sampling_params)
        for output in outputs:
            res = output.outputs[0].text
            ress.append(res)
        utility_score = []
        selected_passages = []
        for index, point in enumerate(ress):
            if "utilityjudgment:no" in ''.join(point.lower().split()):
                utility_score.append(0)
            elif "utilityjudgment:yes" in ''.join(point.lower().split()):
                utility_score.append(1)
                selected_passages.append(passages[index])
            else:
                utility_score.append(0)
        # for i in range(len(formated_passages_ids)):
        file_w.write(json.dumps({
        "query_id": query_id,
        "query": query,
        "gt_answer": gt_answer,
        "formated_passages_ids": formated_passages_ids,
        "point_output": ress,
        "utility_score": utility_score,
        "select_passages": selected_passages
        })+"\n")