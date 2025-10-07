import sys
import os
import logging
import torch
import json
from tqdm import tqdm
import transformers
#dataset
from vllm import LLM, SamplingParams
from torch.utils.data import DataLoader
from dataset import QueryLiklihoodEncodeDataset
from arguments import ModelArguments, DataArguments
from transformers import (
    HfArgumentParser,
)
import math
def collate_fn(batch):
    # messages, messages_no, query_id, gt_answer, query
    messages = [item[0] for item in batch]
    messages_no = [item[1] for item in batch]
    query_ids = [item[2] for item in batch]
    gt_answers = [item[3] for item in batch]
    querys = [item[4] for item in batch]
    return messages, messages_no, query_ids, gt_answers, querys
def calculate_likelihood(log_probs):
    # 计算总log probability
    total_log_prob = sum(log_probs)
    # 计算并返回likelihood
    likelihood = math.exp(total_log_prob)
    return likelihood

if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
    # if "llama" in model_args.model_name_or_path:
    #     sampling_params = SamplingParams(temperature=0.0, max_tokens=4096, stop = ["<|eot_id|>"], prompt_logprobs=True)
    # else:
    #     sampling_params = SamplingParams(temperature=0.0, max_tokens=4096, prompt_logprobs=True)
    sampling_params = SamplingParams(temperature=0.0, prompt_logprobs=1, max_tokens=1)
    # sampling_params = SamplingParams(temperature=0.0, max_tokens=4096, stop = ["<|eot_id|>"])
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    llm = LLM(
        model=model_args.model_name_or_path,
        tensor_parallel_size=8,
    )
    dataset = QueryLiklihoodEncodeDataset(data_args, tokenizer)
    dataloader = DataLoader(dataset, batch_size=data_args.batch_size, shuffle=False, collate_fn=collate_fn)
    file_w = open(data_args.output_dir, "w", encoding="utf-8")
    for batch in tqdm(dataloader):
        messages, messages_no, query_ids, gt_answers, querys = batch
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        inputs_token = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        inputs_no_token = tokenizer.apply_chat_template(
            messages_no,
            tokenize=True,
            add_generation_prompt=True,
        )
        log_probses = []
        log_probs_tokens = []
        likelihood = []
        # print(inputs)
        outputs = llm.generate(inputs, sampling_params)
        for index, output in enumerate(outputs):
            # print(inputs_token[index])
            # print(inputs_no_token[index])
            # print(output.prompt_logprobs)
            answer_tokens = inputs_token[index][len(inputs_no_token[index]):]
            # print(output.prompt_logprobs)
            logs = output.prompt_logprobs[-len(answer_tokens):]
            # print(logs)
            # assert 1 > 2
            log_probs = [logs[i][answer_tokens[i]].logprob for i in range(len(answer_tokens)-1)]
            log_probses.append(log_probs)
            likelihood.append(calculate_likelihood(log_probs))
            log_probs_token = [logs[i][answer_tokens[i]].decoded_token for i in range(len(answer_tokens)-1)]
            log_probs_tokens.append(log_probs_token)
            # all_log_probs = [logs[i][answer_tokens[i]].logprob for i in range(len(answer_tokens))]
        start_idx = 0
        query_results = []
        for i in range(len(query_ids)):
            query_results.append({
                "query_id": query_ids[i],
                "query": querys[i],
                "gt_answer": gt_answers[i],
                "likelihood_all": log_probses[i],  # 使用收集的结果
                "likelihood": likelihood[i],  # 使用收集的结果
                "likelihood_tokens": log_probs_tokens[i],  # 使用收集的结果
            })
        for result in query_results:
            # print(result)
            file_w.write(json.dumps(result) + "\n")
        
        
        
        
       