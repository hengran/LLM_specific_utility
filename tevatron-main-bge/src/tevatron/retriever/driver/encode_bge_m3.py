import logging
import os
import pickle
import sys
from contextlib import nullcontext

import numpy as np
from tqdm import tqdm

import torch

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import (
    HfArgumentParser,
)

from tevatron.retriever.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments
from tevatron.retriever.dataset import EncodeDataset
from tevatron.retriever.collator import BGECollator
from tevatron.retriever.modeling import EncoderOutput, DenseModel
from FlagEmbedding import BGEM3FlagModel

logger = logging.getLogger(__name__)

# from FlagEmbedding import BGEM3FlagModel

# model = BGEM3FlagModel('BAAI/bge-m3',  
#                        use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

# sentences_1 = ["What is BGE M3?", "Defination of BM25"]
# sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.", 
#                "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

# embeddings_1 = model.encode(sentences_1, 
#                             batch_size=12, 
#                             max_length=8192, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
#                             )['dense_vecs']
# embeddings_2 = model.encode(sentences_2)['dense_vecs']
# similarity = embeddings_1 @ embeddings_2.T
# print(similarity)
# # [[0.6265, 0.3477], [0.3499, 0.678 ]]

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if training_args.local_rank > 0 or training_args.n_gpu > 1:
        raise NotImplementedError('Multi-GPU encoding is not supported.')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )


    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )
    # if tokenizer.pad_token_id is None:
    #     tokenizer.pad_token_id = tokenizer.eos_token_id
    # tokenizer.padding_side = 'right'

    # if training_args.bf16:
    #     torch_dtype = torch.bfloat16
    # elif training_args.fp16:
    #     torch_dtype = torch.float16
    # else:
    #     torch_dtype = torch.float32
    
    # model = DenseModel.load(
    #     model_args.model_name_or_path,
    #     pooling=model_args.pooling,
    #     normalize=model_args.normalize,
    #     lora_name_or_path=model_args.lora_name_or_path,
    #     cache_dir=model_args.cache_dir,
    #     torch_dtype=torch_dtype
    # )
    model = BGEM3FlagModel(
        model_args.model_name_or_path,
        # devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"],   # if you don't have GPUs, you can use ["cpu", "cpu"]
        pooling_method='cls',
        cache_dir=os.getenv('HF_HUB_CACHE', None),
    )

    encode_dataset = EncodeDataset(
        data_args=data_args,
    )

    encode_collator = BGECollator()
    encode_loader = DataLoader(
        encode_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=encode_collator,
        shuffle=False,
        drop_last=False,
        # num_workers=training_args.dataloader_num_workers,
    )
    encoded = []
    lookup_indices = []
    # model = model.to(training_args.device)
    # model.eval()

    for (batch_ids, batch) in tqdm(encode_loader):
        lookup_indices.extend(batch_ids)
        with torch.cuda.amp.autocast() if training_args.fp16 or training_args.bf16 else nullcontext():
            with torch.no_grad():
                # for k, v in batch:
                    # batch[k] = v.to(training_args.device)
                if data_args.encode_is_query:
                    model_output = model.encode(batch, batch_size= training_args.per_device_eval_batch_size, max_length=data_args.query_max_len)['dense_vecs']
                    encoded.append(model_output)
                else:
                    model_output = model.encode(batch, batch_size = training_args.per_device_eval_batch_size, max_length=data_args.passage_max_len)['dense_vecs']
                    encoded.append(model_output)

    encoded = np.concatenate(encoded)
    assert len(encoded) == len(lookup_indices)
    with open(data_args.encode_output_path, 'wb') as f:
        pickle.dump((encoded, lookup_indices), f)


if __name__ == "__main__":
    main()
