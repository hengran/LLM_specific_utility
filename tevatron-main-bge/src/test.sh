model_path="output/bge-m3"
q_max_len=64
p_max_len=512
query_prefix=''
passage_prefix=''
lora_model_save_path=$model_path"/encode-triviqa"
corpus_path=$model_path"/encode"
corpus_path='data/kilt_data/corpus_passage.jsonl'
# corpus_path='data/msmarco-pass/corpus.jsonl'
dev_query_path="utility-alignment/datasets/triviaqa/triviaqa-dev.jsonl"
encode_path=$lora_model_save_path

mkdir -p $lora_model_save_path
mkdir $encode_path
echo "====== encode query"
query_prefix=''
passage_prefix=''

CUDA_VISIBLE_DEVICES=1 python -m tevatron.retriever.driver.encode_bge_m3 \
  --output_dir=temp \
  --model_name_or_path $model_path \
  --normalize \
  --encode_is_query \
  --fp16 \
  --per_device_eval_batch_size 128 \
  --passage_max_len $p_max_len \
  --pooling eos \
  --append_eos_token \
  --query_prefix "" \
  --passage_prefix "" \
  --query_max_len $q_max_len \
  --dataset_path $dev_query_path \
  --encode_output_path $encode_path/dev_query_emb.pkl
for s in 0 1 2 3 4 5 6 7;
do
gpuid=$s
CUDA_VISIBLE_DEVICES=$s python -m tevatron.retriever.driver.encode_bge_m3 \
  --output_dir=temp \
  --model_name_or_path $model_path \
  --normalize \
  --fp16 \
  --per_device_eval_batch_size 1024 \
  --pooling eos \
  --append_eos_token \
  --query_prefix "" \
  --passage_prefix "" \
  --passage_max_len $p_max_len \
  --dataset_path $corpus_path \
  --query_max_len $q_max_len \
  --dataset_number_of_shards 8 \
  --dataset_shard_index ${s} \
  --encode_output_path $encode_path/corpus_emb.${s}.pkl   &
  # 等待最后一个循环完成
  if [ "$s" == "7" ]; then
      wait
  fi
done
sleep 10s
echo "====== Search the Corpus"
echo $encode_path
set -f && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m tevatron.retriever.driver.search \
--query_reps $encode_path/dev_query_emb.pkl \
--passage_reps $corpus_path'/corpus_emb.*.pkl' \
--depth 200 \
--batch_size 256 \
--save_text \
--save_ranking_to $encode_path/dev_rank.txt
