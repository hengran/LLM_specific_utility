import sys
import os
import json
import torch
import numpy as np
from tqdm import tqdm
sys.path.append(".")
from transformers import AutoTokenizer, AutoModelForCausalLM
from arguments import ModelArguments, DataArguments
from dataset import ListAttentionEncodeDataset, AnswerDataset
from transformers import HfArgumentParser
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import gather_object

import numpy as np
from contextlib import contextmanager

@contextmanager
def full_print():
    # 保存原始设置
    opts = np.get_printoptions()
    # 临时设置
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    try:
        yield
    finally:
        # 恢复原始设置
        np.set_printoptions(**opts)

def calculate_attention_weights_batch(tokenizer, model, accelerator, batch):
    # 确保使用左padding
    utility_prompts, formated_passages_ids, query_ids, gt_answers, queries, formated_passages = batch
    tokenizer.padding_side = "left"  # 显式设置
    
    inputs = tokenizer(
        utility_prompts,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
        truncation=True,
        max_length=4096
    )
    
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_lengths = inputs["attention_mask"].sum(dim=1)  # 实际内容长度
    total_seq_len = inputs["input_ids"].size(1)  # 含padding的总长度
    batch_size = inputs["input_ids"].size(0)
    
    unwrapped_model = accelerator.unwrap_model(model)
    
    # First get model outputs with attention during generation
    outputs = unwrapped_model.generate(
        **inputs,
        max_new_tokens=32,
        return_dict_in_generate=True,
        output_attentions=True,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    batch_results = []
    all_attentions = outputs.attentions
    
    for i in range(batch_size):
        cur_input_len = input_lengths[i].item()
        pad_len = total_seq_len - cur_input_len  # 左padding长度
        
        # Get tokens for this sample
        content_tokens = tokenizer.convert_ids_to_tokens(
            inputs["input_ids"][i]
        )
        
        # Find passage spans in the tokenized input
        passage_spans = []
        passage_attention_weights = []  # Store attention weights for each passage
        
        for idx in range(20):
            passage_text = f"{formated_passages[i][idx]}\n"
            passage_tokens = tokenizer.tokenize(passage_text)[:-1]
            
            # Find passage position in content_tokens
            found = False
            for pos in range(len(content_tokens) - len(passage_tokens) + 1):
                if content_tokens[pos:pos+len(passage_tokens)] == passage_tokens:
                    # Adjust for padding offset
                    actual_start = pos
                    actual_end = pos + len(passage_tokens)
                    passage_spans.append((actual_start, actual_end))
                    found = True
                    break
            
            if not found:
                print(f"Warning: Passage {idx+1} not found in tokens")
                passage_spans.append((0, 0))  # Fallback to avoid crashes
        
        # Process generation results
        gen_ids = outputs.sequences[i, total_seq_len:]  # Skip input sequence
        generated_answer = tokenizer.decode(gen_ids, skip_special_tokens=True)
        num_gen_tokens = len(gen_ids)
        
        # Calculate attention weights for each generated token
        token_passage_weights = []
        
        if num_gen_tokens > 0 and all_attentions is not None:
            for step in range(num_gen_tokens):
                # Get attention weights for current generation step
                step_attn = all_attentions[-1][step][i]  # Shape: (num_heads, seq_len, seq_len)
                
                # Current sequence length including generated tokens so far
                current_seq_len = total_seq_len + step + 1
                
                # Get attention from current token to all previous tokens
                # Use the last position (current generated token) as query
                token_attn = step_attn[:, -1, :current_seq_len]  # Shape: (num_heads, current_seq_len)
                
                # Average across attention heads
                avg_attn = token_attn.mean(dim=0).cpu().float().numpy()  # Shape: (current_seq_len,)
                
                # Calculate attention weight for each passage
                passage_weights = np.zeros(20)
                
                for k, (start, end) in enumerate(passage_spans):
                    if start < end and end <= len(avg_attn):
                        # Extract attention weights for this passage's tokens
                        passage_attn = avg_attn[start:end]
                        # Sum attention weights for all tokens in this passage
                        passage_weights[k] = passage_attn.sum()
                        
                        # Alternative: use average attention weight
                        # passage_weights[k] = passage_attn.mean()
                
                # Normalize passage weights to sum to 1 (optional)
                total_weight = passage_weights.sum()
                if total_weight > 0:
                    normalized_passage_weights = passage_weights / total_weight
                else:
                    normalized_passage_weights = passage_weights
                
                token_passage_weights.append({
                    'step': step,
                    'generated_token': tokenizer.decode([gen_ids[step]], skip_special_tokens=True),
                    'raw_passage_weights': passage_weights.tolist(),
                    'normalized_passage_weights': normalized_passage_weights.tolist()
                })
        
        # Calculate overall passage attention (average across all generation steps)
        if token_passage_weights:
            overall_passage_weights = np.mean([step_data['raw_passage_weights'] 
                                             for step_data in token_passage_weights], axis=0)
            overall_normalized_weights = overall_passage_weights / overall_passage_weights.sum() \
                                       if overall_passage_weights.sum() > 0 else overall_passage_weights
        else:
            overall_passage_weights = np.zeros(20)
            overall_normalized_weights = np.zeros(20)
        
        # Store results
        result = {
            "query_id": query_ids[i],
            "query": queries[i],
            "generated_answer": generated_answer,
            "formated_passages_ids": formated_passages_ids[i],
            "gt_answer": gt_answers[i],
            "passage_spans": passage_spans,
            "token_passage_weights": token_passage_weights,  # Per-token attention weights
            "overall_passage_weights": overall_passage_weights.tolist(),  # Average across all tokens
            "overall_normalized_passage_weights": overall_normalized_weights.tolist(),
            "passage_attention_summary": {
                f"passage_{j}": {
                    "passage_id": formated_passages_ids[i][j] if j < len(formated_passages_ids[i]) else None,
                    "raw_weight": float(overall_passage_weights[j]),
                    "normalized_weight": float(overall_normalized_weights[j]),
                    "rank": int(np.argsort(overall_passage_weights)[::-1].tolist().index(j) + 1)
                } for j in range(20)
            }
        }
        
        batch_results.append(result)
    
    return batch_results

def collate_fn(batch):
    # utility_prompt, formated_passages_ids, query_id, gt_answer, query, formated_passages
    utility_prompt = [item[0] for item in batch]
    formated_passages_ids = [item[1] for item in batch]
    query_id = [item[2] for item in batch]
    gt_answer = [item[3] for item in batch]
    query = [item[4] for item in batch]
    formated_passages = [item[5] for item in batch]
    return utility_prompt, formated_passages_ids, query_id, gt_answer, query, formated_passages

if __name__ == '__main__':
    accelerator = Accelerator()
    parser = HfArgumentParser((ModelArguments, DataArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        attn_implementation="eager"
    )
    model.eval()
    
    dataset = ListAttentionEncodeDataset(data_args, tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=data_args.batch_size, 
        shuffle=False,
        collate_fn=collate_fn
    )
    
    model, dataloader = accelerator.prepare(model, dataloader)
    os.makedirs(os.path.dirname(data_args.output_dir), exist_ok=True)
    
    if accelerator.is_main_process:
        file_w = open(data_args.output_dir, "w", encoding="utf-8")
    
    for batch in tqdm(dataloader, disable=not accelerator.is_main_process):
        batch_results = calculate_attention_weights_batch(
            tokenizer, 
            model,
            accelerator,
            batch
        )
        
        all_results = gather_object(batch_results)
        
        # Only main process writes to file
        if accelerator.is_main_process:
            for result in all_results:
                file_w.write(json.dumps(result, ensure_ascii=False) + "\n")
            file_w.flush()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        file_w.close()
        print("Attention weight calculation completed!")