import sys
import os
import json
import torch
from tqdm import tqdm
sys.path.append(".")
from transformers import AutoTokenizer, AutoModelForCausalLM
from arguments import ModelArguments, DataArguments
from dataset import PointEncodeDataset
from transformers import HfArgumentParser
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import gather_object

def find_subsequence(sequence, target1, target2):
    """在序列中查找目标子序列"""
    sequence = sequence.cpu().tolist()
    target_len1 = len(target1)
    target_len2 = len(target2)
    max_len = max(target_len1, target_len2)
    for i in range(len(sequence) - max_len + 1):
        if sequence[i:i+target_len1] == target1:
            return i, target1
        if sequence[i:i+target_len2] == target2:
            return i, target2
    return -1, -1

def calculate_entropy_batch(utility_prompts, tokenizer, model, accelerator, model_anme):
    # tokenizer.padding_side = "left"  # 显式设置
    """批量处理prompts并计算熵"""
    # 批量编码prompts
    inputs = tokenizer(
        utility_prompts, 
        return_tensors="pt", 
        padding=True,
        return_attention_mask=True,
        truncation=True,
        max_length=2048  # 防止过长的序列
    )
    
    # 将输入移动到模型所在设备
    # inputs = accelerator.prepare({k: v for k, v in inputs.items()})
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    # print(inputs["input_ids"].shape)
    # assert 1 > 2
    
    # input_lengths = inputs["attention_mask"].sum(dim=1)  # 实际输入长度
    input_lengths = inputs["input_ids"].size(1)
    batch_size = inputs["input_ids"].size(0)
    
    # 获取原始模型（解除DDP包装）
    unwrapped_model = accelerator.unwrap_model(model)
    
    # 批量生成
    outputs = unwrapped_model.generate(
        **inputs,
        max_new_tokens=128,
        return_dict_in_generate=True,
        output_logits=True,
        do_sample=False,
        # eos_token_id=None,  # Disable early stopping
        pad_token_id=tokenizer.eos_token_id,
    )
    
    # 获取logits和生成序列
    logits = outputs.logits  # 元组 (max_new_tokens, batch_size, vocab_size)
    sequences = outputs.sequences  # (batch_size, total_len)
    max_new_tokens = len(logits)
    
    # 目标token ID qwen
    # target_tokens1 = [19407, 25]
    # target_tokens2= [9407, 25, 364]
    # llama 3
    if "llama" in model_anme.lower():
        target_tokens1 = [19971, 25]    # "judgment:"
        target_tokens2 = [19971, 794, 330]  # "judgment\": \""
    # qwen3-8B
    if "qwen" in model_anme.lower():
        target_tokens1 =[19199, 19407, 25]
        target_tokens2= [19407, 25]
    
    batch_results = []
    for i in range(batch_size):
        # 提取当前样本的生成序列（去掉输入部分）
        input_len = input_lengths
        generated_tokens = sequences[i, input_len:].clone().detach()
        
        # 查找目标序列
        start_pos, matched_target = find_subsequence(generated_tokens, target_tokens1, target_tokens2)
        end_pos = start_pos + len(matched_target)
        
        if start_pos == -1:
            print("generated_tokens: ", generated_tokens, "target_tokens1: ", target_tokens1, "target_tokens2: ", target_tokens2)
            # assert 1 > 2
            batch_results.append({
                "generated_text": tokenizer.decode(generated_tokens),
                "target_token_entropy": -1,
                "target_token": "",
                "message": "Fail: Target not found"
            })
            continue
        
        # 检查end_pos是否超出logits范围
        if end_pos >= max_new_tokens:
            print(f"Warning: end_pos ({end_pos}) >= max_new_tokens ({max_new_tokens}), using last available position")
            # assert 1 > 2
            return {
                "generated_text": tokenizer.decode(generated_tokens),
                "target_token_entropy": -1,
                "target_token": "",
                "message": f"Fail: Target position ({end_pos}) out of logits range ({max_new_tokens})"
            }
            continue
        
        # 获取目标位置对应的logits
        # print("len(logits): ", len(logits))
        # print("len(logits[0]): ", len(logits[0]))
        # print("len(logits[0][0]): ", len(logits[0][0]))
        # print("end_pos: ", end_pos)
        target_logits = logits[end_pos][i]  # (max_new_tokens, batch_size, vocab_size)
        token_id = generated_tokens[end_pos].item()
        token_str = tokenizer.decode([token_id])
        
        # 计算熵
        probs = torch.nn.functional.softmax(target_logits, dim=-1)
        probs = probs + 1e-10
        probs = probs / probs.sum()
        entropy = -torch.sum(probs * torch.log2(probs)).item()
        
        batch_results.append({
            "generated_text": tokenizer.decode(generated_tokens),
            "target_token_entropy": entropy,
            "target_token": token_str,
            "message": "Success"
        })
    
    return batch_results

def collate_fn(batch):
    """将多个样本的utility_prompts合并为一个大batch"""
    # 解压batch数据
    utility_prompts_batch = []
    metadata_batch = []
    
    for item in batch: # batch=4
        utility_prompts, formated_passages_ids, query_id, gt_answer, query, passages = item
        utility_prompts_batch.extend(utility_prompts)
        metadata_batch.append({
            "formated_passages_ids": formated_passages_ids,
            "query_id": query_id,
            "gt_answer": gt_answer,
            "query": query,
            "passages": passages,
            "num_prompts": len(utility_prompts)  # 记录每个查询的prompt数量
        })
    
    return utility_prompts_batch, metadata_batch

if __name__ == '__main__':
    # 初始化Accelerator
    accelerator = Accelerator()
    
    parser = HfArgumentParser((ModelArguments, DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.padding_side = "left"
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, 
        # attn_implementation="sdpa",
    )
    model.eval()
    
    # 创建数据集
    dataset = PointEncodeDataset(data_args, tokenizer)
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset, 
        batch_size=data_args.batch_size, 
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # 准备模型和数据加载器进行分布式处理
    model, dataloader = accelerator.prepare(model, dataloader)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(data_args.output_dir), exist_ok=True)
    
    # 主进程负责写文件
    if accelerator.is_main_process:
        file_w = open(data_args.output_dir, "w", encoding="utf-8")
    
    # 处理每个批次
    for batch in tqdm(dataloader, disable=not accelerator.is_main_process):
        utility_prompts_batch, metadata_batch = batch
        
        # 批量计算熵
        batch_results = calculate_entropy_batch(
            utility_prompts_batch, 
            tokenizer, 
            model,
            accelerator,
            model_args.model_name_or_path, 
        )
        all_batch_results = gather_object(batch_results)  # 关键修改：收集所有进程的结果
        all_metadata_batch= gather_object(metadata_batch)  # 关键修改：收集所有进程的结果
        start_idx = 0
        query_results = []
        for meta in all_metadata_batch:
            num_prompts = meta["num_prompts"]
            end_idx = start_idx + num_prompts
            
            # 使用收集到的所有结果
            query_results.append({
                "query_id": meta["query_id"],
                "query": meta["query"],
                "gt_answer": meta["gt_answer"],
                "formated_passages_ids": meta["formated_passages_ids"],
                "entropies": all_batch_results[start_idx:end_idx],  # 使用收集的结果
                "passages": meta["passages"]
            })
            start_idx = end_idx
        
        # 只在主进程写入结果
        if accelerator.is_main_process:
            for result in query_results:
                file_w.write(json.dumps(result, ensure_ascii=False) + "\n")
            file_w.flush()
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        file_w.close()