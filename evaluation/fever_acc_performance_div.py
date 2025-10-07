import json
hotpotqa_filter = set()
with open("utility_alignment/datasets/hotpotqa/postive.jsonl", "r", encoding="utf-8") as file_r:
    for line in file_r:
        js = json.loads(line)
        hotpotqa_filter.add(js["query_id"])
for data_type in ["fever"]:
    for model_name in ["llama31-8b-instruct",  "Qwen3-8B", "Qwen3-14B", "Qwen3-32B"]:
        query_id_scores = {}
        if model_name == "llama31-8b-instruct":
            with open("utility_alignment/results/pseudo_answer/"+data_type+"/"+model_name+"_no_passages_results.jsonl", "r", encoding="utf-8") as file_r:
                for line in file_r:
                    js = json.loads(line)
                    query_id = js["query_id"]
                    query_id_scores[query_id] = js["acc"]
        else:
            with open("utility_alignment/results/pseudo_answer/"+data_type+"/"+model_name+"_no_passages_results.jsonl", "r", encoding="utf-8") as file_r:
                for line in file_r:
                    js = json.loads(line)
                    query_id = js["query_id"]
                    query_id_scores[query_id] = js["acc"]
        all_nums = []
        original_nums = []
        with open("utility_alignment/results/gold_utility/"+data_type+"/"+model_name+"_utility_passages.jsonl", "w", encoding="utf-8") as file_w:
            with open("utility_alignment/results/gold_utility/"+data_type+"/"+model_name+"_answer_with_point_passage_results.jsonl", "r", encoding="utf-8") as file_r: 
                for line in file_r:
                    js = json.loads(line)
                    query_id = js["query_id"]
                    if "hotpotqa" in data_type:
                        if query_id not in hotpotqa_filter:
                            continue
                    utility_passages =  [passage for passage in js["acc"] if passage["utility_score"]-query_id_scores[query_id] > 0]
                    passage_score = [passage["utility_score"]-query_id_scores[query_id] for passage in js["acc"]]
                    utility = [score for score in passage_score if score >0]
                    original = [passage["utility_score"]==1 for passage in js["acc"]]
                    all_nums.append(len(utility))
                    original_nums.append(sum(original))
                    file_w.write(json.dumps({
                        "query_id": query_id,
                        "query": js["query"],
                        "utility_passages": utility_passages,
                        "gt_answer": []
                    })+"\n")
            print(data_type)
            print(model_name, " avg: ", sum(all_nums)/len(all_nums))
            print(model_name, " original_nums: ", sum(original_nums)/len(original_nums))
