import random
import re 
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import List, Tuple
from arguments import DataArguments
import json
import ast
#####################################################################
def format_query(query: str) -> str:
    return f'{query.strip()}'.strip()

def format_passage(text: str, title: str = '') -> str:
    return f'{title.strip()} {text.strip()}'.strip()
#####################################################################
query_type = {}
for data_type in ["nq", "hotpotqa", "msmarco", "triviaqa", "fever", "2wikiqa"]:
    with open(f"datasets/{data_type}/top_200_passages.jsonl", "r", encoding="utf-8") as file_r:
        for line in file_r:
            js = json.loads(line)
            query_type[js["query_id"]] = data_type
def get_direct_judge_point_utility(question, passage):
    return [
            {'role': 'user', 'content': f"Question: {question} \n Passage: {passage} \n Determine if the passage has utility based on two strict criteria: \n 1. Usefulness: The passage should not only be relevant to the question but also be useful in generating a correct, reasonable, and perfect answer to the question.\n 2. Novelty: Is the useful information new to you? This means it must NOT be part of your pre-existing knowledge.\n"+"Let's think step by step. The final format of the output is: 'Utility judgment: Yes/No.'"},
           ]
def get_direct_judge_point_utility_no_cot(question, passage):
    return [
            {'role': 'user', 'content': f"Question: {question} \n Passage: {passage} \n Determine if the passage has utility based on two strict criteria: \n 1. Usefulness: The passage should not only be relevant to the question but also be useful in generating a correct, reasonable, and perfect answer to the question.\n 2. Novelty: Is the useful information new to you? This means it must NOT be part of your pre-existing knowledge.\n"+"Directly output your response. The format of the output is: 'Utility judgment: Yes/No.'"},
           ]
def get_direct_judge_point_utility_no_cot_prob(question, passage):
    return [
            {'role': 'user', 'content': f"Question: {question} \n Passage: {passage} \n Determine if the passage has utility based on two strict criteria: \n 1. Usefulness: The passage should not only be relevant to the question but also be useful in generating a correct, reasonable, and perfect answer to the question.\n 2. Novelty: Is the useful information new to you? This means it must NOT be part of your pre-existing knowledge.\n"+"Directly output your response. The format of the output is: 'Utility judgment: Yes/No.'"},
            {'role': 'assistant', 'content': 'Utility judgment: '}
           ]

def get_direct_judge_point_utility_no_cot_answer(question, passage, answer):
    return [
            {'role': 'user', 'content': f"Question: {question} \n Passage: {passage} \n The reference answer: {answer} Determine if the passage has utility based on two strict criteria: \n 1. Usefulness: The passage should not only be relevant to the question but also be useful in generating a correct, reasonable, and perfect answer to the question.\n 2. Novelty: Is the useful information new to you? This means it must NOT be part of your pre-existing knowledge.\n"+"Directly output your response. The format of the output is: 'Utility judgment: Yes/No.'"},
           ]
def get_direct_judge_point_utility_prob(question, passage):
    return [
            {'role': 'user', 'content': f"Question: {question} \n Passage: {passage} \n Determine if the passage has utility based on two strict criteria: \n 1. Usefulness: The passage should not only be relevant to the question but also be useful in generating a correct, reasonable, and perfect answer to the question.\n 2. Novelty: Is the useful information new to you? This means it must NOT be part of your pre-existing knowledge.\n"+"The final format of the output is: 'Utility judgment: Yes/No.'"},
            {'role': 'assistant', 'content': 'Utility judgment: '}
           ]
def get_prefix_direct_judge_list_utility(query, num):
    return [
            {'role': 'user',
             'content': f"I will provide you with {num} passages, each indicated by number identifier []. \nSelect the passages that have utility for youself in answering the question: {query}."},
            {'role': 'assistant', 'content': 'Okay, please provide the passages.'}]

def get_post_direct_judge_list_utility(query, instruct):
    return f"Question: {query}.\n\n Determine if the passage has utility based on two strict criteria: \n 1. Usefulness: The passage should not only be relevant to the question but also be useful in generating a correct, reasonable, and perfect answer to the question.\n 2. Novelty: Is the useful information new to you? This means it must NOT be part of your pre-existing knowledge.\n"+instruct

def get_direct_judge_list_utility(question, instruct, passages):
    messages = get_prefix_direct_judge_list_utility(question, len(passages))
    rank = 0
    for content in passages:
        rank += 1
        messages.append({'role': 'user', 'content': f"[{rank}] {content}"})
        messages.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
    messages.append({'role': 'user', 'content': get_post_direct_judge_list_utility(question, instruct)})
    return messages

def get_direct_rank_list_utility(question, instruct, passages):
    num = len(passages)
    messages = [
            {'role': 'user',
             'content': f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on the passages utility for youself in answering the question: {question}."},
            {'role': 'assistant', 'content': 'Okay, please provide the passages.'}]
    rank = 0
    for content in passages:
        rank += 1
        messages.append({'role': 'user', 'content': f"[{rank}] {content}"})
        messages.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
    messages.append({'role': 'user', 'content': f"Question: {question}.\n\n Determine if the passage has utility based on two strict criteria: \n 1. Usefulness: The passage should not only be relevant to the question but also be useful in generating a correct, reasonable, and perfect answer to the question.\n 2. Novelty: Is the useful information new to you? This means it must NOT be part of your pre-existing knowledge.\n"+instruct})
    return messages

def get_direct_rank_list_utility_with_answer(question, instruct, passages, answer):
    num = len(passages)
    messages = [
            {'role': 'user',
             'content': f"I will provide you with {num} passages, each indicated by number identifier []. I will also give you a reference answer. \nRank the passages based on the passages' utility for youself in answering the question: {question}."},
            {'role': 'assistant', 'content': 'Okay, please provide the passages and the reference answer.'}]
    rank = 0
    for content in passages:
        rank += 1
        messages.append({'role': 'user', 'content': f"[{rank}] {content}"})
        messages.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
    messages.append({'role': 'user', 'content': f"Question: {question}.\n\n Reference answer: {answer}\n\n Determine if the passage has utility based on two strict criteria: \n 1. Usefulness: The passage should not only be relevant to the question but also be useful in generating a correct, reasonable, and perfect answer to the question.\n 2. Novelty: Is the useful information new to you? This means it must NOT be part of your pre-existing knowledge.\n"+instruct})
    return messages
# =========================================================
def get_direct_judge_list_utility_with_answer(question, instruct, passages, answer):
    messages = get_prefix_direct_judge_list_utility_answer(question, len(passages))
    rank = 0
    for content in passages:
        rank += 1
        messages.append({'role': 'user', 'content': f"[{rank}] {content}"})
        messages.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
    messages.append({'role': 'user', 'content': get_post_direct_judge_list_utility_with_answer(question, instruct, answer)})
    return messages
def get_prefix_direct_judge_list_utility_answer(query, num):
    return [
            {'role': 'user',
             'content': f"I will provide you with {num} passages, each indicated by number identifier []. I will also give you a reference answer. \nSelect the passages that have utility for youself in answering the question: {query}."},
            {'role': 'assistant', 'content': 'Okay, please provide the passages and the reference answer.'}]
def get_post_direct_judge_list_utility_with_answer(query, instruct, answer):
    return f"Question: {query}.\n\n Reference answer: {answer}\n\n Determine if the passage has utility based on two strict criteria: \n 1. Usefulness: The passage should not only be relevant to the question but also be useful in generating a correct, reasonable, and perfect answer to the question.\n 2. Novelty: Is the useful information new to you? This means it must NOT be part of your pre-existing knowledge.\n"+instruct
#####################################################################
def get_utility_select_file(file_name):
    query_utility_ids = {}
    with open(file_name, "r", encoding="utf-8") as file:
        for line in file:
            js = json.loads(line)
            query_id = js["query_id"]
            new_numbers = []
            deterministic_probability = [entropy["deterministic_probability"] for entropy in js["entropies"]]
            for i in range(len(deterministic_probability)):
                if deterministic_probability[i] > 0.5:
                    new_numbers.append(i)
            query_utility_ids[query_id] = new_numbers
    return query_utility_ids
            
            

def direct_generate_answer_prompt(question):
    return [
            {'role': 'user', 'content': f"Answer the following question based on your internal knowledge with one or few words without the explanation. Question: {question}\n\n Answer:"},]
def direct_generate_answer_prompt_with_answer(question, answer):
    return [
            {'role': 'user', 'content': f"Answer the following question based on your internal knowledge with one or few words without the explanation. Question: {question}\n\n Answer:"},
            {'role': 'assistant', 'content': f'{answer}'},]

def generate_answer_prompt_passages(question, passages):
    if isinstance(passages, list):

        if len(passages) == 0:
            if query_type[question] in ["nq", "hotpotqa", "2wikiqa"]:
                return generate_answer_prompt_passages_without_passage(question)
            else:
                return direct_generate_answer_prompt(question)
            # return [ 
            # {'role': 'user', 'content': f"Answer the following question based on your internal knowledge with one or few words without the source.\n Question: {question}\n\n Answer:"},]
        pas = '\n'.join(passages)
    else:
        pas = passages
    return [ 
            {'role': 'user', 'content': f"Information: \n{pas}\n Answer the following question based on the given information or your internal knowledge  with one or few words without the source.\n Question: {question}\n\n Answer:"},]


def generate_answer_prompt_passages_without_passage(question):
    return [ 
            {'role': 'user', 'content': f"Answer the following question based on your internal knowledge with one or few words without the source.\n Question: {question}\n\n Answer:"},]

def generate_answer_prompt_passages_attention(question, passages):
    if isinstance(passages, list):
        pas = '\n'.join(passages)
    else:
        pas = passages
    return [ 
            {'role': 'user', 'content': f"Information: \n{pas}\n Answer the following question based on the given information or your internal knowledge  with one or few words without the source.\n Question: {question}\n\n Answer:"},]
    # all_passages = ""
    # for i in range(len(passages)):
    #     all_passages += f"Passage {i+1}: {passages[i]}\n"
    # return [ 
    #         {'role': 'user', 'content': f"Given {len(passages)} passages: \n{all_passages}\n, answer the following question based on the given passages or your internal knowledge  with one or few words without the source.\n Question: {question}\n\n Answer:"},]
def select_answers_prompt(query, answers):
    return [
        {'role': 'user', 'content': f"Query: {query}\n Candidate answers: {answers}\n Based on the given query and the candidate set of answers, select a consistent answer as the final answer. The final answer:"},
    ]
def generate_answer_prompt_passages_with_answer(question, passages, answer):
    if isinstance(passages, list):
        pas = '\n'.join(passages)
    else:
        pas = passages
    return [
        {'role': 'user', 'content': f"Information: \n{pas}\n Answer the following question based on the given information or your internal knowledge  with one or few words without the source.\n Question: {question}\n\n Answer:"},
        {'role': 'assistant', 'content': f'{answer}'},
    ]

class PointEncodeDatasetProb(Dataset):
    def __init__(self, data_args: DataArguments, tokenizer):
        print(data_args)
        self.data_args = data_args
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )
        self.tokenizer = tokenizer
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[int]]:
        _hashed_seed = hash(item)
        group = self.train_data[item]
        query_id = group['query_id']
        query = group['query']
        if "answer" in group:
            gt_answer = group["answer"]
        elif "gt_answer" in group:
            gt_answer = group["gt_answer"]
        else:
            gt_answer = group["answers"]
        passages = group["passages"]
        formated_passages = []
        formated_passages_ids = []
        group_negatives = group["passages"][:self.data_args.topk]
        utility_prompts = []

        for passage in group_negatives:
            formated_passages.append(format_passage(passage["text"], passage["title"]))
            formated_passages_ids.append(passage["docid"])

        for passage in formated_passages:
            messages = get_direct_judge_point_utility_no_cot_prob(query, passage)
            utility_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            utility_prompts.append(utility_prompt)

        return utility_prompts, formated_passages_ids, query_id, gt_answer, query, passages

class PointAnswerEncodeDataset(Dataset):
    def __init__(self, data_args: DataArguments, tokenizer):
        print(data_args)
        self.data_args = data_args
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )
        self.tokenizer = tokenizer
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        self.answer_data = {}
        if self.data_args.gt_answer: 
            with open(self.data_args.gt_answer_path, "r", encoding="utf-8") as file_r:
                for line in file_r:
                    js = json.loads(line)
                    assert type(js["gt_answer"]) == list
                    self.answer_data[js["query_id"]] = js["gt_answer"][0]
        else:
            if self.data_args.answer_file != None:
                with open(self.data_args.answer_file, "r", encoding="utf-8") as file_r:
                    for line in file_r:
                        js = json.loads(line)
                        if "</think>" in  js["answer_output"]:
                            if self.data_args.judge_based_on_think:
                                self.answer_data[js["query_id"]] = js["answer_output"]
                            else:
                                self.answer_data[js["query_id"]] = js["answer_output"].split("</think>")[1]
                        else:
                            self.answer_data[js["query_id"]] = js["answer_output"]
                            
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[int]]:
        _hashed_seed = hash(item)
        group = self.train_data[item]
        query_id = group['query_id']
        query = group['query']
        if "answer" in group:
            gt_answer = group["answer"]
        elif "gt_answer" in group:
            gt_answer = group["gt_answer"]
        else:
            gt_answer = group["answers"]
        pseudo_answer =  self.answer_data[query_id]
        passages = group["passages"]
        formated_passages = []
        formated_passages_ids = []
        group_negatives = group["passages"][:self.data_args.topk]
        utility_prompts = []

        for passage in group_negatives:
            formated_passages.append(format_passage(passage["text"], passage["title"]))
            formated_passages_ids.append(passage["docid"])

        for passage in formated_passages:
            messages = get_direct_judge_point_utility_no_cot_answer(query, passage, pseudo_answer)
            utility_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.data_args.enable_thinking)
            utility_prompts.append(utility_prompt)

        return utility_prompts, formated_passages_ids, query_id, gt_answer, query, passages

    
class PointPerformanceEncodeDataset(Dataset):
    def __init__(self, data_args: DataArguments, tokenizer):
        print(data_args)
        self.data_args = data_args
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )
        self.tokenizer = tokenizer
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[int]]:
        _hashed_seed = hash(item)
        group = self.train_data[item]
        query_id = group['query_id']
        query = group['query']
        if "answer" in group:
            gt_answer = group["answer"]
        elif "gt_answer" in group:
            gt_answer = group["gt_answer"]
        else:
            gt_answer = group["answers"]
        passages = group["passages"]
        formated_passages = []
        formated_passages_ids = []
        group_negatives = group["passages"][:self.data_args.topk]
        utility_prompts = []

        for passage in group_negatives:
            formated_passages.append(format_passage(passage["text"], passage["title"]))
            formated_passages_ids.append(passage["docid"])

        for passage in formated_passages:
            messages = generate_answer_prompt_passages(query, passage)
            utility_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.data_args.enable_thinking)
            utility_prompts.append(utility_prompt)

        return utility_prompts, formated_passages_ids, query_id, gt_answer, query, passages
    
class PointEncodeDataset(Dataset):
    def __init__(self, data_args: DataArguments, tokenizer):
        print(data_args)
        self.data_args = data_args
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )
        self.tokenizer = tokenizer
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[int]]:
        _hashed_seed = hash(item)
        group = self.train_data[item]
        query_id = group['query_id']
        query = group['query']
        if "answer" in group:
            gt_answer = group["answer"]
        elif "gt_answer" in group:
            gt_answer = group["gt_answer"]
        else:
            gt_answer = group["answers"]
        passages = group["passages"]
        formated_passages = []
        formated_passages_ids = []
        group_negatives = group["passages"][:self.data_args.topk]
        utility_prompts = []

        for passage in group_negatives:
            formated_passages.append(format_passage(passage["text"], passage["title"]))
            formated_passages_ids.append(passage["docid"])

        for passage in formated_passages:
            messages = get_direct_judge_point_utility_no_cot(query, passage)
            utility_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.data_args.enable_thinking)
            utility_prompts.append(utility_prompt)

        return utility_prompts, formated_passages_ids, query_id, gt_answer, query, passages

class ListUtilityEncodeDataset(Dataset):
    def __init__(self, data_args: DataArguments, tokenizer):
        print(data_args)
        self.data_args = data_args
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )
        self.tokenizer = tokenizer
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        self.utility_instruct = """
        Directly output the passages you selected that have utility for yourself in answering the question. The format of the output is: 'My selection:[[i],[j],...].'. Only response the selection results, do not say any word or explain. 
        """
        self.answer_data = {}
        if self.data_args.gt_answer: 
            with open(self.data_args.gt_answer_path, "r", encoding="utf-8") as file_r:
                for line in file_r:
                    js = json.loads(line)
                    assert type(js["gt_answer"]) == list
                    self.answer_data[js["query_id"]] = js["gt_answer"][0]
        else:  
            if self.data_args.answer_file != None:  
                with open(self.data_args.answer_file, "r", encoding="utf-8") as file_r:
                    for line in file_r:
                        js = json.loads(line)
                        if "</think>" in  js["answer_output"]:
                            if self.data_args.judge_based_on_think:
                                self.answer_data[js["query_id"]] = js["answer_output"]
                            else:
                                self.answer_data[js["query_id"]] = js["answer_output"].split("</think>")[1]
                        else:
                            self.answer_data[js["query_id"]] = js["answer_output"]
                    # self.answer_data[js["query_id"]] = ast.literal_eval(js["gt_answer"])self.answer_data[js["query_id"]]
                    
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[int]]:
        _hashed_seed = hash(item)
        group = self.train_data[item]
        query_id = group['query_id']
        query = group['query']
        if "answer" in group:
            gt_answer = group["answer"]
        elif "gt_answer" in group:
            gt_answer = group["gt_answer"]
        else:
            gt_answer = group["answers"]
        pseudo_answer =  self.answer_data[query_id]
        passages = group["passages"]
        formated_passages = []
        formated_passages_ids = []
        group_negatives = group["passages"][:self.data_args.topk]
        utility_prompts = []

        for passage in group_negatives:
            formated_passages.append(format_passage(passage["text"], passage["title"]))
            formated_passages_ids.append(passage["docid"])

        messages = get_direct_judge_list_utility_with_answer(query, self.utility_instruct, group_negatives, pseudo_answer)
        utility_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.data_args.enable_thinking)

        return utility_prompt, formated_passages_ids, query_id, gt_answer, query, group_negatives

class ListAttentionEncodeDataset(Dataset):
    def __init__(self, data_args: DataArguments, tokenizer):
        print(data_args)
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[int]]:
        _hashed_seed = hash(item)
        group = self.train_data[item]
        query_id = group['query_id']
        query = group['query']
        if "answer" in group:
            gt_answer = group["answer"]
        elif "gt_answer" in group:
            gt_answer = group["gt_answer"]
        else:
            gt_answer = group["answers"]
        if "selected_passage" in group:
            passages = group["selected_passage"]
        elif "sorted_all_passage_scores" in group: 
            passages = group["sorted_all_passage_scores"][:self.data_args.topk]
        else:
            passages = group["passages"][:self.data_args.topk]
        formated_passages = []
        formated_passages_ids = []
        for passage in passages:
            text = passage["text"]
            # if len(text.split(" ")) > 200:
            #     text = " ".join(text.split(" ")[:200])
            if len(text.split(" ")) > 100:
                text = " ".join(text.split(" ")[:100])
            formated_passages.append(format_passage(text, passage["title"]))
            formated_passages_ids.append(passage["docid"])
        messages = generate_answer_prompt_passages(query, formated_passages)
        # messages = get_direct_judge_list_utility(query, self.utility_instruct, formated_passages)
        utility_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.data_args.enable_thinking)
        
        
        return utility_prompt, formated_passages_ids, query_id, gt_answer, query, formated_passages

class ListRankEncodeDataset(Dataset):
    def __init__(self, data_args: DataArguments, tokenizer):
        print(data_args)
        self.data_args = data_args
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )
        self.tokenizer = tokenizer
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        self.utility_instruct = """
        Directly output the ranked the passages in descending order of utility for yourself in answering the question. The format of the output is: '[i]>[j]>...'. Only response the ranked results, do not say any word or explain. 
        """
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[int]]:
        _hashed_seed = hash(item)
        group = self.train_data[item]
        query_id = group['query_id']
        query = group['query']
        if "answer" in group:
            gt_answer = group["answer"]
        elif "gt_answer" in group:
            gt_answer = group["gt_answer"]
        else:
            gt_answer = group["answers"]
        passages = group["passages"]
        formated_passages = []
        formated_passages_ids = []
        group_negatives = group["passages"][:self.data_args.topk]
        utility_prompts = []

        for passage in group_negatives:
            formated_passages.append(format_passage(passage["text"], passage["title"]))
            formated_passages_ids.append(passage["docid"])

        messages = get_direct_rank_list_utility(query, self.utility_instruct, group_negatives)
        utility_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.data_args.enable_thinking)

        return utility_prompt, formated_passages_ids, query_id, gt_answer, query, group_negatives


class ListRankWithAnswerEncodeDataset(Dataset):
    def __init__(self, data_args: DataArguments, tokenizer):
        print(data_args)
        self.data_args = data_args
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )
        self.tokenizer = tokenizer
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        self.utility_instruct = """
        Directly output the ranked the passages in descending order of utility for yourself in answering the question. The format of the output is: '[i]>[j]>...'. Only response the ranked results, do not say any word or explain. 
        """
        self.answer_data = {}
        if self.data_args.gt_answer: 
            with open(self.data_args.gt_answer_path, "r", encoding="utf-8") as file_r:
                for line in file_r:
                    js = json.loads(line)
                    assert type(js["gt_answer"]) == list
                    self.answer_data[js["query_id"]] = js["gt_answer"][0]
        else:  
            if self.data_args.answer_file != None:  
                with open(self.data_args.answer_file, "r", encoding="utf-8") as file_r:
                    for line in file_r:
                        js = json.loads(line)
                        if "</think>" in  js["answer_output"]:
                            if self.data_args.judge_based_on_think:
                                self.answer_data[js["query_id"]] = js["answer_output"]
                            else:
                                self.answer_data[js["query_id"]] = js["answer_output"].split("</think>")[1]
                        else:
                            self.answer_data[js["query_id"]] = js["answer_output"]
                    # self.answer_data[js["query_id"]] = ast.literal_eval(js["gt_answer"])self.answer_data[js["query_id"]]
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[int]]:
        _hashed_seed = hash(item)
        group = self.train_data[item]
        query_id = group['query_id']
        query = group['query']
        if "answer" in group:
            gt_answer = group["answer"]
        elif "gt_answer" in group:
            gt_answer = group["gt_answer"]
        else:
            gt_answer = group["answers"]
        passages = group["passages"]
        formated_passages = []
        formated_passages_ids = []
        group_negatives = group["passages"][:self.data_args.topk]
        utility_prompts = []
        answer = self.answer_data[query_id]
        for passage in group_negatives:
            formated_passages.append(format_passage(passage["text"], passage["title"]))
            formated_passages_ids.append(passage["docid"])

        messages = get_direct_rank_list_utility_with_answer(query, self.utility_instruct, group_negatives, answer)
        utility_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.data_args.enable_thinking)

        return utility_prompt, formated_passages_ids, query_id, gt_answer, query, group_negatives


class ListEncodeDataset(Dataset):
    def __init__(self, data_args: DataArguments, tokenizer):
        print(data_args)
        self.data_args = data_args
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )
        self.tokenizer = tokenizer
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        self.utility_instruct = """
        Directly output the passages you selected that have utility for yourself in answering the question. The format of the output is: 'My selection:[[i],[j],...].'. Only response the selection results, do not say any word or explain. 
        """
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[int]]:
        _hashed_seed = hash(item)
        group = self.train_data[item]
        query_id = group['query_id']
        query = group['query']
        if "answer" in group:
            gt_answer = group["answer"]
        elif "gt_answer" in group:
            gt_answer = group["gt_answer"]
        else:
            gt_answer = group["answers"]
        passages = group["passages"]
        formated_passages = []
        formated_passages_ids = []
        group_negatives = group["passages"][:self.data_args.topk]
        utility_prompts = []

        for passage in group_negatives:
            formated_passages.append(format_passage(passage["text"], passage["title"]))
            formated_passages_ids.append(passage["docid"])

        messages = get_direct_judge_list_utility(query, self.utility_instruct, group_negatives)
        utility_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.data_args.enable_thinking)

        return utility_prompt, formated_passages_ids, query_id, gt_answer, query, group_negatives

class SelectAnswerDataset(Dataset):
    def __init__(self, data_args: DataArguments, tokenizer):
        print(data_args)
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[int]]:
        _hashed_seed = hash(item)
        group = self.train_data[item]
        query_id = group['query_id']
        query = group['query']
        answers = gruop["answer_outputs"]
        gt_answer = group["gt_answer"]
        messages = select_answers_prompt(query, answers)
        utility_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.data_args.enable_thinking)
        formated_passages_ids = []
        return utility_prompt, formated_passages_ids, query_id, gt_answer, query

class AnswerWithoutPassageDataset(Dataset):
    def __init__(self, data_args: DataArguments, tokenizer):
        print(data_args)
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[int]]:
        _hashed_seed = hash(item)
        group = self.train_data[item]
        query_id = group['query_id']
        query = group['query']
        if "answer" in group:
            gt_answer = group["answer"]
        elif "gt_answer" in group:
            gt_answer = group["gt_answer"]
        else:
            gt_answer = group["answers"]
        if "selected_passage" in group:
            passages = group["selected_passage"]
        elif "sorted_all_passage_scores" in group: 
            passages = group["sorted_all_passage_scores"][:self.data_args.topk]
        else:
            passages = group["passages"][:self.data_args.topk]
        formated_passages = []
        formated_passages_ids = []
        for passage in passages:
            formated_passages.append(format_passage(passage["text"], passage["title"]))
            formated_passages_ids.append(passage["docid"])
        messages = direct_generate_answer_prompt(query)
        # messages = get_direct_judge_list_utility(query, self.utility_instruct, formated_passages)
        utility_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.data_args.enable_thinking)
        
        return utility_prompt, formated_passages_ids, query_id, gt_answer, query

class AnswerDataset(Dataset):
    def __init__(self, data_args: DataArguments, tokenizer):
        print(data_args)
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[int]]:
        _hashed_seed = hash(item)
        group = self.train_data[item]
        query_id = group['query_id']
        query = group['query']
        if "answer" in group:
            gt_answer = group["answer"]
        elif "gt_answer" in group:
            gt_answer = group["gt_answer"]
        else:
            gt_answer = group["answers"]
        passages = group[self.data_args.selected_filed][:self.data_args.topk]
        if  self.data_args.cut != None:
             passages = [passage for passage in passages if passage[self.data_args.score] >= self.data_args.cut]
        formated_passages = []
        formated_passages_ids = []
        for passage in passages: 
            formated_passages.append(format_passage(passage["text"], passage["title"]))
            formated_passages_ids.append(passage["docid"])
        messages = generate_answer_prompt_passages(query, formated_passages)
        utility_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.data_args.enable_thinking)
        return utility_prompt, formated_passages_ids, query_id, gt_answer, query


class EncodeDataset(Dataset):
    def __init__(self, data_args: DataArguments, tokenizer):
        print(data_args)
        self.data_args = data_args
        self.utility_instruct = """
        Directly output the passages you selected that have utility to yourself. The format of the output is: 'My selection:[[i],[j],...].'. Only response the selection results, do not say any word or explain. 
        """
        self.utility_instruct_tool = """
        Please first generate an answer to the question based on the passages, and then output the passages you selected that have utility in answering the question. The format of the output is: 'Answer: [...], My selection:[[i],[j],...].'. Only response the answer and the selection results, do not say any word or explanation. 
    """
        self.relevance_instruct = """
        Directly output the passages you selected that are relevant to the question. The format of the output is: 'My selection:[[i],[j],...].'. Only response the selection results, do not say any word or explain. 
        """
        self.tokenizer = tokenizer
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        if self.data_args.utility_select_file != '':
            self.utility_select = get_utility_select_file(self.data_args.utility_select_file)
        else:
            self.utility_select = None



    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[int]]:
        _hashed_seed = hash(item)
        group = self.train_data[item]
        query_id = group['query_id']
        query = group['query']
        if "answer" in group:
            gt_answer = group["answer"]
        elif "gt_answer" in group:
            gt_answer = group["gt_answer"]
        else:
            gt_answer = group["answers"]
        passages = group["passages"]
        formated_passages = []
        formated_passages_ids = []
        group_negatives = group["passages"][:self.data_args.topk]
        if self.utility_select != None:
            for id in self.utility_select[query_id]:
                formated_passages.append(format_passage(group_negatives[id]["text"], group_negatives[id]["title"]))
                formated_passages_ids.append(group_negatives[id]["docid"])
        else:
            for passage in group_negatives:
                formated_passages.append(format_passage(passage["text"], passage["title"]))
                formated_passages_ids.append(passage["docid"])
        # messages = generate_answer_prompt_passages(query, formated_passages)
        messages = get_direct_judge_list_utility(query, self.utility_instruct, formated_passages)
        utility_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.data_args.enable_thinking)
        
        return utility_prompt, formated_passages_ids, query_id, gt_answer, query
    
class QueryLiklihoodEncodeDataset(Dataset):
    def __init__(self, data_args: DataArguments, tokenizer):
        print(data_args)
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )

        self.answer_data = {}
        if self.data_args.gt_answer: 
            with open(self.data_args.gt_answer_path, "r", encoding="utf-8") as file_r:
                for line in file_r:
                    js = json.loads(line)
                    assert type(js["gt_answer"]) == list
                    self.answer_data[js["query_id"]] = js["gt_answer"][0]
        else:
            if self.data_args.answer_file != None:
                with open(self.data_args.answer_file, "r", encoding="utf-8") as file_r:
                    for line in file_r:
                        js = json.loads(line)
                        if "</think>" in  js["answer_output"]:
                            if self.data_args.judge_based_on_think:
                                self.answer_data[js["query_id"]] = js["answer_output"]
                            else:
                                self.answer_data[js["query_id"]] = js["answer_output"].split("</think>")[1]
                        else:
                            self.answer_data[js["query_id"]] = js["answer_output"]
        # self.id_answers = get_answers()
    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self, item) -> Tuple[str, List[int]]:
        _hashed_seed = hash(item)
        group = self.train_data[item]
        query_id = group['query_id']
        query = group['query']
        if "answer" in group:
            gt_answer = group["answer"]
        elif "gt_answer" in group:
            gt_answer = group["gt_answer"]
        else:
            gt_answer = group["answers"]
        messages = direct_generate_answer_prompt_with_answer(query, self.answer_data[query_id])
        messages_no = direct_generate_answer_prompt(query)

        return messages, messages_no, query_id, gt_answer, query

    
class LiklihoodEncodeDataset(Dataset):
    def __init__(self, data_args: DataArguments, tokenizer):
        print(data_args)
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )

        self.answer_data = {}
        if self.data_args.gt_answer: 
            with open(self.data_args.gt_answer_path, "r", encoding="utf-8") as file_r:
                for line in file_r:
                    js = json.loads(line)
                    assert type(js["gt_answer"]) == list
                    self.answer_data[js["query_id"]] = js["gt_answer"][0]
        else:
            if self.data_args.answer_file != None:
                with open(self.data_args.answer_file, "r", encoding="utf-8") as file_r:
                    for line in file_r:
                        js = json.loads(line)
                        if "</think>" in  js["answer_output"]:
                            if self.data_args.judge_based_on_think:
                                self.answer_data[js["query_id"]] = js["answer_output"]
                            else:
                                self.answer_data[js["query_id"]] = js["answer_output"].split("</think>")[1]
                        else:
                            self.answer_data[js["query_id"]] = js["answer_output"]
        # self.id_answers = get_answers()
    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self, item) -> Tuple[str, List[int]]:
        _hashed_seed = hash(item)
        group = self.train_data[item]
        query_id = group['query_id']
        query = group['query']
        if "answer" in group:
            gt_answer = group["answer"]
        elif "gt_answer" in group:
            gt_answer = group["gt_answer"]
        else:
            gt_answer = group["answers"]
        formated_passages = []
        formated_passages_ids = []
        group_negatives = group["passages"][:self.data_args.topk]
        utility_prompts = []

        for passage in group_negatives:
            formated_passages.append(format_passage(passage["text"], passage["title"]))
            formated_passages_ids.append(passage["docid"])
        messages = []
        messages_no = []
        for passage in formated_passages:
            messages.append(generate_answer_prompt_passages_with_answer(query, passage, self.answer_data[query_id]))
            messages_no.append(generate_answer_prompt_passages(query, passage))
        return messages, messages_no,  formated_passages_ids, query_id, gt_answer, query, group_negatives

   