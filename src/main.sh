#!/bin/bash


# all experiment here


for model_name in "Llama-3.1-8B-Instruct" "Qwen3-14B" "Qwen3-8B" "Qwen3-4B";
do
model_name_or_path="/home/gomall/models/"$model_name
for data_type in "hotpotqa" "2wikiqa";
do
# pointwise verberlized
dataset_path="/home/gomall/work/utility-alignment/datasets/"$data_type"/top_200_passages.jsonl"
output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_verberlized.jsonl"
python point_run_verberlized.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 64 --output_dir $output_dir --topk 20 --enable_thinking False

python dataloader/point_verbarlized.py $dataset_path $output_dir "/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_verberlized_final.jsonl"

model_name_or_path=$model_name_or_path
dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_verberlized_final.jsonl"
output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_verberlized_final_answer.jsonl"
python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --topk 20 --enable_thinking False

# listwise verberlized
dataset_path="/home/gomall/work/utility-alignment/datasets/"$data_type"/top_200_passages.jsonl"
output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_list_verberlized.jsonl"
python listwise_verberlized.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --topk 20 --enable_thinking False

dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_list_verberlized.jsonl"
output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_list_verberlized_answer.jsonl"
python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --enable_thinking False 

# # attention weight
model_name_or_path=$model_name_or_path
dataset_path="/home/gomall/work/utility-alignment/datasets/"$data_type"/top_200_passages.jsonl"
output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_listwise_attention_weight.jsonl"
accelerate launch --num_processes=2 listwise_attention.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --output_dir $output_dir --topk 20  --batch_size 1 --enable_thinking False
# python script.py <top_200_passages_file> <listwise_attention_weight> <listwise_attention_weight_reverse>
python dataloader/listwise_attention.py $dataset_path $output_dir "/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_listwise_attention_weight_reverse.jsonl"
 
model_name_or_path=$model_name_or_path
dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_listwise_attention_weight_reverse.jsonl"
output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_listwise_attention_weight_reverse_answer_20.jsonl"
python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 64 --output_dir $output_dir --topk 20 --enable_thinking False

model_name_or_path=$model_name_or_path
dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_listwise_attention_weight_reverse.jsonl"
output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_listwise_attention_weight_reverse_answer_10.jsonl"
python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 64 --output_dir $output_dir --topk 10 --enable_thinking False

model_name_or_path=$model_name_or_path
dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_listwise_attention_weight_reverse.jsonl"
output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_listwise_attention_weight_reverse_answer_5.jsonl"
python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 64 --output_dir $output_dir --topk 5  --enable_thinking False


#probility_top20
dataset_path="/home/gomall/work/utility-alignment/datasets/"$data_type"/top_200_passages.jsonl"
output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_prob_no_cot.jsonl"
accelerate launch --num_processes=8 pointwise_prob.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 1 --output_dir $output_dir --topk 20 --enable_thinking False

python dataloader/point_probility.py  $output_dir "/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_prob_no_cot_final.jsonl"

model_name_or_path=$model_name_or_path
dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_prob_no_cot_final.jsonl"
output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_prob_no_cot_final_answer.jsonl"
python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --topk 20 --selected_filed "sorted_all_passage_scores" --enable_thinking False
#probility_top10
model_name_or_path=$model_name_or_path
dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_prob_no_cot_final.jsonl"
output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_prob_no_cot_final_answer_10.jsonl"
python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --topk 10 --selected_filed "sorted_all_passage_scores" --enable_thinking False
#probility_top5
model_name_or_path=$model_name_or_path
dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_prob_no_cot_final.jsonl"
output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_prob_no_cot_final_answer_5.jsonl"
python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --topk 5 --selected_filed "sorted_all_passage_scores" --enable_thinking False

#entropy_no_reverve
dataset_path="/home/gomall/work/utility-alignment/datasets/"$data_type"/top_200_passages.jsonl"
output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_entropy_no_cot.jsonl"
accelerate launch --num_processes=8 point_entropy_run.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 1 --output_dir $output_dir --topk 20
python dataloader/point_entropy.py $output_dir "/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_entropy_no_cot_reverseF_final.jsonl"

dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_entropy_no_cot_reverseF_final.jsonl"
output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_entropy_no_cot_reverseF_final_answer_20.jsonl"
python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 64 --output_dir $output_dir --topk 20 --selected_filed "sorted_all_passage_scores" --enable_thinking False

dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_entropy_no_cot_reverseF_final.jsonl"
output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_entropy_no_cot_reverseF_final_answer_10.jsonl"
python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 64 --output_dir $output_dir --topk 10 --selected_filed "sorted_all_passage_scores" --enable_thinking False

dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_entropy_no_cot_reverseF_final.jsonl"
output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_entropy_no_cot_reverseF_final_answer_5.jsonl"
python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 64 --output_dir $output_dir --topk 5 --selected_filed "sorted_all_passage_scores" --enable_thinking False



# #generate sepudo answer
# model_name_or_path=$model_name_or_path
# dataset_path="/home/gomall/work/utility-alignment/datasets/"$data_type"/top_200_passages.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_all_answer.jsonl"
# python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --topk 20  --enable_thinking False

# model_name_or_path=$model_name_or_path
# dataset_path="/home/gomall/work/utility-alignment/datasets/"$data_type"/top_200_passages.jsonl"
# answer_file="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_all_answer.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_list_answer_utility_verberlized.jsonl" 
# python list_answer_utility_verberlized.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --topk 20 --answer_file $answer_file --gt_answer False  --enable_thinking False

# model_name_or_path=$model_name_or_path
# dataset_path="/home/gomall/work/utility-alignment/datasets/"$data_type"/top_200_passages.jsonl"
# answer_file="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_all_answer.jsonl"
# gt_answer_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_gt_answer.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_verberlized_gt_answer.jsonl"
# python pointwise_verberlized_answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 64 --output_dir $output_dir --topk 20 --answer_file $answer_file --gt_answer True --gt_answer_path $gt_answer_path --enable_thinking False
# # answer generation: point_answer_verberlized_gt_answer
# dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_verberlized_gt_answer.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_verberlized_gt_answer_final_answer.jsonl"
# python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --selected_filed "select_passages" --enable_thinking False


# model_name_or_path=$model_name_or_path
# dataset_path="/home/gomall/work/utility-alignment/datasets/"$data_type"/top_200_passages.jsonl"
# answer_file="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_all_answer.jsonl"
# gt_answer_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_gt_answer.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_list_answer_utility_verberlized_gt_answer.jsonl" 
# python list_answer_utility_verberlized.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --topk 20 --answer_file $answer_file  --gt_answer True --gt_answer_path $gt_answer_path --enable_thinking False
# # answer generation: listwise_answer_verberlized_gt_answer
# model_name_or_path="/home/gomall/models/Llama-3.1-8B-Instruct"
# dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_list_answer_utility_verberlized_gt_answer.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_list_answer_utility_verberlized_gt_answer_final_answer.jsonl"
# python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --selected_filed "selected_passage" --enable_thinking False

# # Likelihood
# dataset_path="/home/gomall/work/utility-alignment/datasets/"$data_type"/top_200_passages.jsonl"
# answer_file="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_all_answer.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_pointwise_likelihood.jsonl" 
# python pointwise_likelihood.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --topk 20 --answer_file $answer_file --gt_answer False  --enable_thinking False


# # answer generation: listwise_answer_verberlized
# dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_list_answer_utility_verberlized.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_list_answer_utility_verberlized_answer.jsonl"
# python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --enable_thinking False 
# # pointwise_answer_verberlized
# dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_verberlized.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_verberlized_answer.jsonl"
# python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --selected_filed "select_passages" --topk 20 --enable_thinking False
done
done

# Performance: EM
# with the passages
# dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_top_200_passages.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em.jsonl"
# python pointwise_performance_answerem.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 64 --output_dir $output_dir --topk 20
# dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em_results.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em_results_answer_pseudo_em_weight.jsonl"
# python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --selected_filed "pseudo_em_weight" --topk 20
# dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em_results.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em_results_answer_pseudo_em_weight_cut_1.jsonl"
# python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --selected_filed "pseudo_em_weight" --topk 20 --cut 1 


# model_name_or_path="/home/gomall/models/Llama-3.1-8B-Instruct"
# dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em_results.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em_results_answer_gt_em_weight_cut1.jsonl"
# python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --selected_filed "gt_em_weight" --topk 20 --cut 1

# model_name_or_path="/home/gomall/models/Llama-3.1-8B-Instruct"
# dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em_results.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em_results_answer_pseudo_f1_weight.jsonl"
# python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --selected_filed "pseudo_f1_weight" --topk 20

# model_name_or_path="/home/gomall/models/Llama-3.1-8B-Instruct"
# dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em_results.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em_results_answer_gt_em_weight.jsonl"
# python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --selected_filed "gt_em_weight" --topk 20

# model_name_or_path="/home/gomall/models/Llama-3.1-8B-Instruct"
# dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em_results.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em_results_answer_gt_f1_weight.jsonl"
# python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --selected_filed "gt_f1_weight" --topk 20


# model_name_or_path="/home/gomall/models/Llama-3.1-8B-Instruct"
# dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em_results.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em_results_answer_pseudo_em_weight_top10.jsonl"
# python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --selected_filed "pseudo_em_weight" --topk 10

# model_name_or_path="/home/gomall/models/Llama-3.1-8B-Instruct"
# dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em_results.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em_results_answer_pseudo_f1_weight_top10.jsonl"
# python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --selected_filed "pseudo_f1_weight" --topk 10

# model_name_or_path="/home/gomall/models/Llama-3.1-8B-Instruct"
# dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em_results.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em_results_answer_gt_em_weight_top10.jsonl"
# python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --selected_filed "gt_em_weight" --topk 10

# model_name_or_path="/home/gomall/models/Llama-3.1-8B-Instruct"
# dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em_results.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em_results_answer_gt_f1_weight_top10.jsonl"
# python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --selected_filed "gt_f1_weight" --topk 10


# model_name_or_path="/home/gomall/models/Llama-3.1-8B-Instruct"
# dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em_results.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em_results_answer_pseudo_em_weight_top5.jsonl"
# python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --selected_filed "pseudo_em_weight" --topk 5

# model_name_or_path="/home/gomall/models/Llama-3.1-8B-Instruct"
# dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em_results.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em_results_answer_pseudo_f1_weight_top5.jsonl"
# python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --selected_filed "pseudo_f1_weight" --topk 5

# model_name_or_path="/home/gomall/models/Llama-3.1-8B-Instruct"
# dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em_results.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em_results_answer_gt_em_weight_top5.jsonl"
# python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --selected_filed "gt_em_weight" --topk 5

# model_name_or_path="/home/gomall/models/Llama-3.1-8B-Instruct"
# dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em_results.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em_results_answer_gt_f1_weight_top5.jsonl"
# python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --selected_filed "gt_f1_weight" --topk 5




# # with the passages
# model_name_or_path="/home/gomall/models/Llama-3.1-8B-Instruct"
# dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_top_200_passages.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em.jsonl"
# python pointwise_performance_answerem.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 64 --output_dir $output_dir --topk 20

# without all the passages
# model_name_or_path="/home/gomall/models/Llama-3.1-8B-Instruct"
# dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_top_200_passages.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_answer_withoutput_passage.jsonl"
# python without_passage_answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 64 --output_dir $output_dir --topk 0



#likelihood
# model_name_or_path="/home/gomall/models/Llama-3.1-8B-Instruct"
# dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_top_200_passages.jsonl"
# output_dir_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_pointwise_likelihood.jsonl"
# answer_file="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_gt_answer.jsonl"
# python pointwise_likelihood.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 10 --output_dir $output_dir_path --answer_file $answer_file --topk 20

# model_name_or_path="/home/gomall/models/Llama-3.1-8B-Instruct"
# dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_dev_query_pos_qrel.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_ground_truth_passages_answer.jsonl"
# python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --selected_filed "pos_ids" --topk 20


# model_name_or_path="/home/gomall/models/Llama-3.1-8B-Instruct"
# dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_top_200_passages.jsonl"
# output_dir_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_pointwise_likelihood.jsonl"
# answer_file="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_all_answer.jsonl"
# python pointwise_likelihood.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 10 --output_dir $output_dir_path --answer_file $answer_file --topk 20 --gt_answer false

# model_name_or_path="/home/gomall/models/Llama-3.1-8B-Instruct"
# dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_top_200_passages.jsonl"
# output_dir_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_pointwise_likelihood_gt_answer.jsonl"
# answer_file="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_gt_answer.jsonl"
# python pointwise_likelihood.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 10 --output_dir $output_dir_path --answer_file $answer_file --topk 20 --gt_answer true


# model_name_or_path="/home/gomall/models/Llama-3.1-8B-Instruct"
# dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_top_200_passages.jsonl"
# output_dir_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_pointwise_likelihood.jsonl"
# answer_file="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_all_answer.jsonl"
# python pointwise_likelihood.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 1 --output_dir $output_dir_path --answer_file $answer_file --topk 20 --gt_answer false
# python merge.py


# model_name_or_path="/home/gomall/models/Llama-3.1-8B-Instruct"
# dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_pointwise_likelihood_final.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_pointwise_likelihood_final_answer_top20.jsonl"
# python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --selected_filed "sorted_all_passage_scores" --topk 20

# model_name_or_path="/home/gomall/models/Llama-3.1-8B-Instruct"
# dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_pointwise_likelihood_final.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_pointwise_likelihood_final_answer_top10.jsonl"
# python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --selected_filed "sorted_all_passage_scores" --topk 10

# model_name_or_path="/home/gomall/models/Llama-3.1-8B-Instruct"
# dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_pointwise_likelihood_final.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_pointwise_likelihood_final_answer_top5.jsonl"
# python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --selected_filed "sorted_all_passage_scores" --topk 5

# model_name_or_path="/home/gomall/models/Llama-3.1-8B-Instruct"
# dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_pointwise_likelihood_gt_answer_final.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_pointwise_likelihood_gt_answer_final_answer_top20.jsonl"
# python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --selected_filed "sorted_all_passage_scores" --topk 20

# model_name_or_path="/home/gomall/models/Llama-3.1-8B-Instruct"
# dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_pointwise_likelihood_gt_answer_final.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_pointwise_likelihood_gt_answer_final_answer_top10.jsonl"
# python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --selected_filed "sorted_all_passage_scores" --topk 10

# model_name_or_path="/home/gomall/models/Llama-3.1-8B-Instruct"
# dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_pointwise_likelihood_gt_answer_final.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_pointwise_likelihood_gt_answer_final_answer_top5.jsonl"
# python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --selected_filed "sorted_all_passage_scores" --topk 5



# model_name_or_path="/home/gomall/models/Llama-3.1-8B-Instruct"
# dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em_results_divergence.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em_results_divergence_em_cut_bye1.jsonl"
# python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --selected_filed "pseudo_em_weight_bye_pmi" --topk 20 --cut 1 --score "bye_div"


# model_name_or_path="/home/gomall/models/Llama-3.1-8B-Instruct"
# dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em_results_divergence.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em_results_divergence_em_cut_pmi1.jsonl"
# python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --selected_filed "pseudo_em_weight_bye_pmi" --topk 20 --cut 1 --score "pmi_div"




# model_name_or_path="/home/gomall/models/Llama-3.1-8B-Instruct"
# dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em_results_divergence.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em_results_divergence_f1_cut_bye0.5.jsonl"
# python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --selected_filed "pseudo_f1_weight_bye_pmi" --topk 20 --cut 0.5 --score "bye_div"


# model_name_or_path="/home/gomall/models/Llama-3.1-8B-Instruct"
# dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em_results_divergence.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em_results_divergence_f1_cut_bye0.3.jsonl"
# python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --selected_filed "pseudo_f1_weight_bye_pmi" --topk 20 --cut 0.3 --score "bye_div"


# model_name_or_path="/home/gomall/models/Llama-3.1-8B-Instruct"
# dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em_results_divergence.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_point_answer_performance_em_results_divergence_f1_cut_pmi1.jsonl"
# python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --selected_filed "pseudo_f1_weight_bye_pmi" --topk 20 --cut 1 --score "pmi_div"


# model_name_or_path="/home/gomall/models/Llama-3.1-8B-Instruct"
# dataset_path="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_top_200_passages.jsonl"
# answer_file="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_all_answer.jsonl"
# output_dir="/home/gomall/work/utility-alignment/results/"$data_type"/"$model_name"_list_answer_utility_verberlized_first.jsonl" 
# python list_answer_utility_verberlized.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --topk 10 --answer_file $answer_file


# model_name_or_path="/home/gomall/models/Llama-3.1-8B-Instruct"
# dataset_path="/root/paddlejob/workspace/env_run/output/utility-alignment/datasets/nq/top_200_passages.jsonl"
# answer_file="/root/paddlejob/workspace/env_run/output/utility-alignment/datasets/nq/all_answer.jsonl"
# output_dir="/root/paddlejob/workspace/env_run/output/utility-alignment/datasets/nq/list_answer_utility_verberlized_last.jsonl" 
# python list_answer_utility_verberlized.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --topk -10 --answer_file $answer_file


