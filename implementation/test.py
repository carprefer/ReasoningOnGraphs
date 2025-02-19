import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

outputPath = f"../output/myRoG/trained_model3"

tokenizer = AutoTokenizer.from_pretrained(outputPath, use_fast=False)
model = AutoPeftModelForCausalLM.from_pretrained(outputPath, torch_dtype=torch.bfloat16)
model.to('cuda')

tokens_to_check = ['<SEP>','<PATH>','</PATH>', '<PAD>']

def check_tokens_in_tokenizer(tokenizer, tokens):
    for token in tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != tokenizer.unk_token_id:
            print(f"Token '{token}' is in the tokenizer with ID {token_id}.")
        else:
            print(f"Token '{token}' is not in the tokenizer.")
check_tokens_in_tokenizer(tokenizer, tokens_to_check)

embedding_layer = model.get_input_embeddings()
embedding_size = embedding_layer.embedding_dim
print(f"The embedding size is {embedding_size}.")