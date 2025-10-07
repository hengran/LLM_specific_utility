import json
from operator import itemgetter
with open("../datasets/nq/pointwise_likelihood_gt_answer_final.jsonl", "w", encoding="utf-8") as file_w:
    with open("../datasets/nq/pointwise_likelihood_gt_answer.jsonl", "r", encoding="utf-8") as file_r:
        for line in file_r:
            js = json.loads(line)
            passages = js["passages"]
            likelihood = js["likelihood"]
            attention_weight = []
            for i, passage in enumerate(passages):
                attention_weight.append({"docid": passages[i]["docid"], "text": passages[i]["text"], "title": passages[i]["title"], "likelihood": likelihood[i]*10000000000}) 
            sorted_all_passage_scores = sorted(attention_weight, key=itemgetter("likelihood"), reverse=True)
            file_w.write(json.dumps({
                    "query_id": js["query_id"],
                    "query": js["query"],
                    "gt_answer": js["gt_answer"],
                    "sorted_all_passage_scores": sorted_all_passage_scores
                })+"\n")
import json
from operator import itemgetter
with open("../datasets/nq/pointwise_likelihood_final.jsonl", "w", encoding="utf-8") as file_w:
    with open("../datasets/nq/pointwise_likelihood.jsonl", "r", encoding="utf-8") as file_r:
        for line in file_r:
            js = json.loads(line)
            passages = js["passages"]
            likelihood = js["likelihood"]
            attention_weight = []
            for i, passage in enumerate(passages):
                attention_weight.append({"docid": passages[i]["docid"], "text": passages[i]["text"], "title": passages[i]["title"], "likelihood": likelihood[i]*10000000000}) 
            sorted_all_passage_scores = sorted(attention_weight, key=itemgetter("likelihood"), reverse=True)
            file_w.write(json.dumps({
                    "query_id": js["query_id"],
                    "query": js["query"],
                    "gt_answer": js["gt_answer"],
                    "sorted_all_passage_scores": sorted_all_passage_scores
                })+"\n")

                                     
        