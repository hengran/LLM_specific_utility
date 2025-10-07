
##LLM-specific Utility
for model_name in "llama31-8b-instruct" "Qwen3-14B" "Qwen3-8B" "Qwen3-32B"; 
do
    model_name_or_path="model/"$model_name
    for data_type in  "nq";
    do
    mkdir "utility_alignment/results/main_experiments/pseudo_answer/"$data_type
    dataset_path="utility_alignment/datasets/"$data_type"/top_200_passages.jsonl"
    output_dir="utility_alignment/results/main_experiments/pseudo_answer/"$data_type"/"$model_name"_point_verberlized_answer_thinking_judge_based_on_think_False.jsonl"
    answer_file="utility_alignment/results/pseudo_answer/"$data_type"/"$model_name"-all-passages.jsonl"
    python pointwise_verberlized_answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 64 --output_dir $output_dir --topk 20 --enable_thinking False --answer_file $answer_file --gt_answer False --judge_based_on_think False
    dataset_path="utility_alignment/datasets/"$data_type"/top_200_passages.jsonl"
    output_dir="utility_alignment/results/main_experiments/pseudo_answer/"$data_type"/"$model_name"_list_verberlized_answer.jsonl"
    answer_file="utility_alignment/results/pseudo_answer/"$data_type"/"$model_name"-all-passages.jsonl"
    python listwise_verbalized_answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 128 --output_dir $output_dir --topk 20 --enable_thinking False --answer_file $answer_file --gt_answer False --judge_based_on_think False
    done
done

for model_name in "llama31-8b-instruct" "Qwen3-14B" "Qwen3-8B" "Qwen3-32B";
do
    model_name_or_path="model/"$model_name
    for data_type in "nq" "2wikiqa" "hotpotqa" "msmarco" "triviaqa";
    do
    mkdir output_dir="utility_alignment/results/main_experiments/"$data_type"/"
    # pointwise verberlized
    dataset_path="utility_alignment/datasets/"$data_type"/top_200_passages.jsonl"
    output_dir="utility_alignment/results/main_experiments/"$data_type"/"$model_name"_point_verberlized.jsonl"
    python point_run_verberlized.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 64 --output_dir $output_dir --topk 20 --enable_thinking False

    # listwise verberlized
    dataset_path="utility_alignment/datasets/"$data_type"/top_200_passages.jsonl"
    output_dir="utility_alignment/results/main_experiments/"$data_type"/"$model_name"_list_verberlized.jsonl"
    python listwise_verberlized.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --topk 20 --enable_thinking False

    # pointwise verberlized with pseudo-answer
    dataset_path="utility_alignment/datasets/"$data_type"/top_200_passages.jsonl"
    output_dir="utility_alignment/results/main_experiments/pseudo_answer/"$data_type"/"$model_name"_point_verberlized_answer_thinking_judge_based_on_think_False.jsonl"
    answer_file="utility_alignment/results/pseudo_answer/"$data_type"/"$model_name"-all-passages.jsonl"
    python pointwise_verberlized_answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 64 --output_dir $output_dir --topk 20 --enable_thinking False --answer_file $answer_file --gt_answer False --judge_based_on_think False
    dataset_path="utility_alignment/datasets/"$data_type"/top_200_passages.jsonl"
    output_dir="utility_alignment/results/main_experiments/pseudo_answer/"$data_type"/"$model_name"_list_verberlized_answer2.jsonl"
    answer_file="utility_alignment/results/pseudo_answer/"$data_type"/"$model_name"-all-passages.jsonl"
    python listwise_verbalized_answer2.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 128 --output_dir $output_dir --topk 20 --enable_thinking False --answer_file $answer_file --gt_answer False --judge_based_on_think False

    # # attention weight
    model_name_or_path=$model_name_or_path
    dataset_path="utility_alignment/datasets/"$data_type"/top_200_passages.jsonl"
    output_dir="utility_alignment/results/main_experiments/"$data_type"/"$model_name"_listwise_attention_weight.jsonl"
    accelerate launch --num_processes=2 listwise_attention.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --output_dir $output_dir --topk 20  --batch_size 1 --enable_thinking False

    dataset_path="utility-alignment/datasets/"$data_type"/top_200_passages.jsonl"
    answer_file="utility-alignment/results/"$data_type"/"$model_name"_all_answer.jsonl"
    output_dir="utility-alignment/results/"$data_type"/"$model_name"_pointwise_likelihood.jsonl" 
    python pointwise_likelihood.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --topk 20 --answer_file $answer_file --gt_answer False --enable_thinking False

    done
done
### RAG 
for model in "llama31-8b-instruct";
do 
    for dataset_type in "nq" "hotpotqa";
    do 
    model_name_or_path=""$model
    dataset_path="utility_alignment/results/gt_likelihood/"$dataset_type"/"$model"_utility_passages_filter.jsonl"
    output_dir="utility_alignment/results/gt_likelihood/"$dataset_type"/"$model"_utility_passages_filter_answer.jsonl"
    python answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --output_dir $output_dir --topk 20 --selected_filed "utility_passages" --enable_thinking False
    done
done
