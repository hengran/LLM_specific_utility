
# Datasets
We use the four datasets from KILT (NQ, HotpotQA, TriviaQA, FEVER), MS MARCO, and 2WikiQA. 
Queries, human-annotated passages, the corpus, and ground-truth answers of KILT are provided by [DPR](https://github.com/facebookresearch/DPR/issues/186).   
Queries, human-annotated passages, the corpus, and ground-truth answers of MS MARCO can be downloaded from the [official code](https://microsoft.github.io/msmarco/).    
The queries and ground-truth answer of 2WikiQA can be downloaded from the [official code](https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?e=1).   

# Retrieval
We use the retrieval code from [Tevatron](https://github.com/texttron/tevatron/tree/main). 
```
cd tevatron-main-bge/src
sh test.sh 
```
The top-20 results are saved in datasets/. Due to the memory limitation, we will upload the retrieval results to HuggingFace for each dataset after the anonymity period.  

# Gold Utility for Specific LLM


For each dataset and specific LLM: 
```
 model_name_or_path="Qwen3-8B"
 dataset_path="utility-alignment/datasets/nq/Qwen3-8B_top_200_passages.jsonl"
 output_dir="utility-alignment/results/pseudo_answer/nq/Qwen3-8B_point_answer_performance_em.jsonl"
 python pointwise_performance_answerem.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 64 --output_dir $output_dir --topk 20
```
Then, 
```
model_name_or_path="Qwen3-8B"
dataset_path="utility-alignment/datasets/nq/Qwen3-8B_top_200_passages.jsonl"
output_dir="utility-alignment/results/pseudo_answer/nq/Qwen3-8B_answer_withoutput_passage.jsonl"
python without_passage_answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 64 --output_dir $output_dir --topk 0

```
Lastly, compute the performance difference between w/ passage and w/o passage using the evaluation/has_answer_point.py and evaluation/gold_utility_computing.py.  
The gold_utility results are saved in results/gold_utility/. 
Finllay,  for the final LLM-Specific-Utility-Benchmark can be found at [LLM-Specific-Utility-Benchmark](https://modelscope.cn/datasets/hengranzhang/LLM-Specific-Utility-Benchmark/settings)

# LLM-specific Utility Judgment Methods
Running codes of all methods are shown in src/answer_test.sh 

# Evaluation
Two evaluation types are in evaluation/ranking-evaluation.py  and evaluation/set-evaluation.py. 
The judgment results are saved in results/main_experiments. Due to the memory limitation, we will upload all the judgment results to HuggingFace for each dataset after the anonymity period. 


