import numpy as np
import json
import math
from sklearn.metrics import ndcg_score
hotpotqa_filter = set()
import numpy as np

import pandas as pd
import tempfile
import os
import copy
from typing import Dict, Tuple
def calculate_likelihood(log_probs):
    # 计算总log probability
    total_log_prob = sum(log_probs)
    # 计算并返回likelihood
    likelihood = math.exp(total_log_prob)
    return likelihood

def get_point_results(point_outputs):
    results = []
    for point in point_outputs:
        if ": yes" in point.lower() or " yes" in point.lower():
            results.append(1)
        else:
            results.append(0)
    return results

def pearson_correlation_numpy(x, y):
    """
    使用NumPy计算两个列表的皮尔逊相关系数
    """
    return np.corrcoef(x, y)[0, 1]

with open("utility_alignment/datasets/hotpotqa/postive.jsonl", "r", encoding="utf-8") as file_r:
    for line in file_r:
        js = json.loads(line)
        hotpotqa_filter.add(js["query_id"])

def calculate_set(selected_passage_ids, reference_ids):
    same = [_id for _id in selected_passage_ids if _id in reference_ids]
    precision = len(same)/len(selected_passage_ids) if len(selected_passage_ids) !=0 else 0
    recall = len(same)/len(reference_ids)
    return precision, recall
# for data_type in ["nq", "hotpotqa", "2wikiqa", "triviaqa"]:
# for data_type in ["fever", "nq", "hotpotqa", "2wikiqa", "triviaqa"]:
for data_type in ["msmarco"]:
    for model_type in ["llama31-8b-instruct", "Qwen3-8B","Qwen3-14B", "Qwen3-32B"]:
    # for model_type in ["Qwen3-8B", "Qwen3-32B"]:
        # for method_type in ["_point_verberlized", "_list_verberlized", "pseudo_answer_point", "pseudo_answer_list"]:
        for method_type in ["_list_verberlized", "pseudo_answer_list"]:
            query_docids = {}
            reference_lielihoods = {}
            true_relevance = []
            predicted_scores = []
            correlations = []
            query_utility_dict = {}
            with open("utility_alignment/results/gold_utility/"+data_type+"/"+model_type+"_utility_passages.jsonl", "r", encoding="utf-8") as file_r:
                for line in file_r:
                    js = json.loads(line)
                    query_id = js["query_id"]
                    if "hotpotqa" in data_type:
                        if query_id not in hotpotqa_filter:
                            continue
                    utility_passages = js["utility_passages"]
                    utility_passages_ids = [str(passage["docid"]) for passage in utility_passages]
                    query_utility_dict[query_id] = utility_passages_ids
            # candidate
            pres = []
            recalls = []
            accs = []
            nums_rel_0 = []
            nums_non_0 = []
            root = "utility_alignment/results/main_experiments/"
            if method_type =="pseudo_answer_list":
                file_path = root+"pseudo_answer/"+data_type+"/"+model_type+"_list_verberlized.jsonl"
            elif method_type =="pseudo_answer_point":
                file_path = root+"pseudo_answer/"+data_type+"/"+model_type+"_point_verberlized.jsonl"
            else:
                file_path = root+data_type+"/"+model_type+method_type+".jsonl"
            with open(file_path, "r", encoding="utf-8") as file_r:
                for line in file_r:
                    js = json.loads(line)
                    query_id = js["query_id"]
                    if "hotpotqa" in data_type:
                        if query_id not in hotpotqa_filter:
                            continue
                    if "selected_passage" not in js:
                        point_outputs = get_point_results(js["point_output"])
                        selected_passage_ids = [_id for index, _id in enumerate(js["formated_passages_ids"]) if point_outputs[index] == 1]
                    else:
                        selected_passage_ids = [passage["docid"] for passage in js["selected_passage"]]
                    selected_passage_ids = list(set(selected_passage_ids))
                    if query_id not in query_utility_dict:
                        continue
                    reference_ids = query_utility_dict[query_id]
                    if len(reference_ids) == 0:
                        nums_rel_0.append(len(selected_passage_ids))
                        if len(selected_passage_ids) == 0:
                            accs.append(1)
                        else:
                            accs.append(0)
                    else:
                        nums_non_0.append(len(selected_passage_ids))
                        pre, recall = calculate_set(selected_passage_ids, reference_ids)
                        pres.append(pre)
                        recalls.append(recall)
            # print(len(accs), " ",len(recalls))
            nums_rel = sum(nums_rel_0)/len(nums_rel_0)
            nums_none = sum(nums_non_0)/len(nums_non_0)
            precision = sum(pres)/len(pres)
            recall = sum(recalls)/len(recalls)
            acc = sum(accs)/len(accs)
            f1 = 2 * (precision*recall)/(precision + recall)
            print(data_type, "\t", len(recalls), "\t", model_type, "\t", method_type, "\t", precision*100,"\t", recall*100, "\t", f1*100, "\t", nums_rel, "\t",nums_none)
