import numpy as np
import json
import math
import re
from sklearn.metrics import ndcg_score
hotpotqa_filter = set()
import numpy as np

import pandas as pd
import tempfile
import os
import copy
from typing import Dict, Tuple
import pytrec_eval
import argparse 
from operator import itemgetter
def cal_mrr(qrels, results, k_values):
    mrr = {}
    for k in k_values:
        runs_topk = {query: dict(sorted(docs.items(), key=lambda x: x[1], reverse=True)[:k]) for query, docs in results.items()}
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'recip_rank'})
        scores = evaluator.evaluate(runs_topk)
        mrr[f"MRR@{k}"] = scores
    return mrr

def trec_eval(qrels: Dict[str, Dict[str, int]],
              results: Dict[str, Dict[str, float]],
              k_values: Tuple[int] = (10, 50, 100, 200, 1000)) -> Dict[str, float]:
    ndcg, _map, recall, mrr = {}, {}, {}, {}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        mrr[f"MRR@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string})
    scores = evaluator.evaluate(results)

    all_mrr = cal_mrr(qrels, results, k_values)

    for query_id in scores:
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            mrr[f"MRR@{k}"] += all_mrr[f"MRR@{k}"][query_id]["recip_rank"]

    def _normalize(m: dict) -> dict:
        return {k: round(v / len(scores), 5) for k, v in m.items()}

    ndcg = _normalize(ndcg)
    _map = _normalize(_map)
    recall = _normalize(recall)
    mrr = _normalize(mrr)

    all_metrics = {}
    for mt in [ndcg, _map, recall, mrr]:
        all_metrics.update(mt)

    return all_metrics


def remove_duplicate(response):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
        else:
            print('duplicate')
    return new_response


def clean_response(response: str):
    new_response = ''
    for c in response:
        if not c.isdigit():
            new_response += ' '
        else:
            try:
                new_response += str(int(c))
            except:
                new_response += ' '
    new_response = new_response.strip()
    return new_response


class EvalFunction:
    @staticmethod
    def receive_responses(rank_results, responses, cut_start=0, cut_end=100):
        print('receive_responses', len(responses), len(rank_results))
        for i in range(len(responses)):
            response = responses[i]
            response = clean_response(response)
            response = [int(x) - 1 for x in response.split()]
            response = remove_duplicate(response)
            cut_range = copy.deepcopy(rank_results[i]['hits'][cut_start: cut_end])
            original_rank = [tt for tt in range(len(cut_range))]
            response = [ss for ss in response if ss in original_rank]
            response = response + [tt for tt in original_rank if tt not in response]
            for j, x in enumerate(response):
                rank_results[i]['hits'][j + cut_start] = {
                    'content': cut_range[x]['content'], 'qid': cut_range[x]['qid'], 'docid': cut_range[x]['docid'],
                    'rank': cut_range[j]['rank'], 'score': cut_range[j]['score']}
        return rank_results

    @staticmethod
    def trunc(qrels, run):
        qrels = get_qrels_file(qrels)
        # print(qrels)
        run = pd.read_csv(run, delim_whitespace=True, header=None)
        qrels = pd.read_csv(qrels, delim_whitespace=True, header=None)
        run[0] = run[0].astype(str)
        qrels[0] = qrels[0].astype(str)

        qrels = qrels[qrels[0].isin(run[0])]
        temp_file = tempfile.NamedTemporaryFile(delete=False).name
        qrels.to_csv(temp_file, sep='\t', header=None, index=None)
        return temp_file

    @staticmethod
    def main(args_qrel, args_run):

        # args_qrel = EvalFunction.trunc(args_qrel, args_run)

        assert os.path.exists(args_qrel)
        assert os.path.exists(args_run)

        with open(args_qrel, 'r') as f_qrel:
            qrel = pytrec_eval.parse_qrel(f_qrel)

        with open(args_run, 'r') as f_run:
            run = pytrec_eval.parse_run(f_run)

        all_metrics = trec_eval(qrel, run, k_values=(1, 5, 10, 20))
        # print(all_metrics)
        return all_metrics
def calculate_likelihood(log_probs):
    # 计算总log probability
    total_log_prob = sum(log_probs)
    # 计算并返回likelihood
    likelihood = math.exp(total_log_prob)
    return likelihood

def extract_numbers(s):
    pattern = r'\[(\d+)\]'
    numbers_str = re.findall(pattern, s)
    numbers = []
    seen = set()  # 用于跟踪已出现的数字
    for num_str in numbers_str:
        num = int(num_str)
        # 检查数字是否在1-20范围内且未出现过
        if 1 <= num <= 20 and num not in seen:
            numbers.append(num)
            seen.add(num)
    numbers = [num-1 for num in numbers]
    return numbers
def pearson_correlation_numpy(x, y):
    """
    使用NumPy计算两个列表的皮尔逊相关系数
    """
    return np.corrcoef(x, y)[0, 1]
with open("utility_alignment/datasets/hotpotqa/postive.jsonl", "r", encoding="utf-8") as file_r:
    for line in file_r:
        js = json.loads(line)
        hotpotqa_filter.add(js["query_id"])
print("\tNDCG@5 \t NDCG@10 \t MRR@5 \t MRR@10 \t Recall@5 \t Recall@10")
for data_type in ["triviaqa"]:
# for data_type in ["2wikiqa", "triviaqa", "2wikiqa", "nq", "hotpotqa"]:
    for model_type in ["llama31-8b-instruct", "Qwen3-8B", "Qwen3-14B", "Qwen3-32B"]:
    # for model_type in ["Qwen3-8B", "llama31-8b-instruct"]:
        # reference
        query_docids = {}
        reference_lielihoods = {}
        true_relevance = []
        predicted_scores = []
        correlations = []
        qrel_file = "utility_alignment/results/pointwise_answer/"+data_type+"/utility_passages/"+model_type+"_utility_passages_qrel.txt"
        with open(qrel_file, "w", encoding="utf-8") as file_w:
            with open("utility_alignment/results/pointwise_answer/"+data_type+"/utility_passages/"+model_type+"_utility_passages.jsonl", "r", encoding="utf-8") as file_r:
                for line in file_r:
                    js = json.loads(line)
                    query_id = js["query_id"]
                    if "hotpotqa" in data_type:
                        if query_id not in hotpotqa_filter:
                            continue
                    if "triviaqa" in data_type:
                        query_id = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', query_id)
                    utility_passages = js["utility_passages"]
                    utility_passages_ids = [passage["docid"] for passage in utility_passages]
                    for utility_id in utility_passages_ids:
                        file_w.write(query_id+" 0 "+utility_id+" 1\n")
        # likelihood candidate
        trec_file = "utility_alignment/results/main_experiments/"+data_type+"/"+model_type+"_likelihood_trec.txt"
        with open(trec_file, "w", encoding="utf-8") as file_w:
            with open("utility_alignment/results/main_experiments/"+data_type+"/"+model_type+"_likelihood.jsonl", "r", encoding="utf-8") as file_r:
                for line in file_r:
                    js = json.loads(line)
                    query_id = js["query_id"]
                    if "hotpotqa" in data_type:
                        if query_id not in hotpotqa_filter:
                            continue
                    likelihoods = []
                    if "<think>" in js["likelihood_tokens"][0]:
                        likelihood_alls = js["likelihood_all"]
                        for likelihood_all in likelihood_alls:
                            likelihood = calculate_likelihood(likelihood_all[4:-1])
                            likelihoods.append(likelihood)
                    else:
                        likelihoods = js["likelihood"]
                    formated_passages_ids = js["formated_passages_ids"][:20]
                    passgae_lieklihood = []
                    for i in range(len(likelihoods)):
                        passgae_lieklihood.append({"passage_id": formated_passages_ids[i], "likelihood": likelihoods[i]})
                    passgae_lieklihood = sorted(passgae_lieklihood, key=itemgetter("likelihood"), reverse=True)  
                    # 8813f87c0bdd11eba7f7acde48001122 Q0 3901230 1 0.6575829982757568 dense 
                    for index, passage in enumerate(passgae_lieklihood):
                        passage_id = passage["passage_id"]
                        likelihood = passage["likelihood"]
                        file_w.write(f"{query_id} Q0 {passage_id} {index} {likelihood} likelihood\n")

        # ranking_verbarlized candidate
        read_ids = set()
        root = "utility_alignment/results/main_experiments/"
        trec_file = root+data_type+"/"+model_type+"_rank_list_verberlized_trec.txt"
        # trec_file = root+data_type+"/"+model_type+"_rank_answer_list_verberlized_trec.txt"
        with open(trec_file, "w", encoding="utf-8") as file_w:
            if not os.path.exists(root+data_type+"/"+model_type+"_rank_list_verberlized.jsonl"):
                continue
            with open(root+data_type+"/"+model_type+"_rank_list_verberlized.jsonl", "r", encoding="utf-8") as file_r:
            # with open(root+data_type+"/"+model_type+"_rank_answer_list_verberlized.jsonl", "r", encoding="utf-8") as file_r:
                for line in file_r:
                    js = json.loads(line)
                    query_id = js["query_id"]
                    if "hotpotqa" in data_type:
                        if query_id not in hotpotqa_filter:
                            continue
                    query_id = js["query_id"]
                    if "triviaqa" in data_type:
                        query_id = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', query_id)
                    # query_id = query_id.replace("\n", "")
                    if query_id in read_ids:
                        continue
                    rank_list = extract_numbers(js["listwise_output"])
                    read_ids.add(query_id)
                    formated_passages_ids = js["formated_passages_ids"][:20]
                    for index, rank in enumerate(rank_list):
                        passage_id = formated_passages_ids[rank]
                        likelihood = len(formated_passages_ids)-index
                        line = f"{query_id} Q0 {passage_id} {index} {likelihood} verbarlized\n"
                        file_w.write(line)
                        if len(line.strip().split()) != 6:
                            print(js)
                        assert len(line.strip().split()) == 6
                    
        # attention candidate 
        read_ids = set()
        trec_file = "utility_alignment/results/main_experiments/"+data_type+"/"+model_type+"_attention_trec.txt"
        with open(trec_file, "w", encoding="utf-8") as file_w:
            with open("utility_alignment/results/main_experiments/"+data_type+"/"+model_type+"_listwise_attention_weight.jsonl", "r", encoding="utf-8") as file_r:
                for line in file_r:
                    js = json.loads(line)
                    query_id = js["query_id"]
                    if "hotpotqa" in data_type:
                        if query_id not in hotpotqa_filter:
                            continue
                    if "triviaqa" in data_type:
                        query_id = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', query_id)
                    if query_id in read_ids:
                        continue
                    read_ids.add(query_id)
                    passage_attention_summary = js["passage_attention_summary"]
                    # 提取所有 passage 数据
                    passages = passage_attention_summary.values()
                    sorted_passages = sorted(passages, key=lambda x: x["normalized_weight"], reverse=True)
                    # assert len(set([passages["passage_id"] for passage in sorted_passages])) == 20
                    sorted_passage_ids = [passage["passage_id"] for passage in sorted_passages]
                    assert len(set(sorted_passage_ids)) == 20
                    for index, passage in enumerate(sorted_passages):
                        passage_id = passage["passage_id"]
                        likelihood = passage["normalized_weight"]
                        file_w.write(f"{query_id} Q0 {passage_id} {index} {likelihood} attention\n")
        all_metrics = EvalFunction.main(qrel_file, trec_file)
        print(data_type, model_type, len(read_ids), "\t", 100*all_metrics["NDCG@5"], "\t", 100*all_metrics["NDCG@10"], "\t", 100*all_metrics["MRR@5"], "\t", 100*all_metrics["MRR@10"], "\t", 100*all_metrics["Recall@5"], "\t", 100*all_metrics["Recall@10"])
