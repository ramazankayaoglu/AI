import torch
from master_model import MasterModel
from master_tokenizer import MasterTokenizer
import master_tokenizer

master_tokenizer = MasterTokenizer("tokenizer.json")

prompt = "the capital of united"

tokens = master_tokenizer.encode(prompt)

print(tokens.shape)

torch.manual_seed(1)    


master_model = MasterModel(vocab_size=len(master_tokenizer.vocab), embedding_dim= 4, num_heads=4, context_length=32)

sentence_meanings_with_attention_context = master_model(tokens)
print(sentence_meanings_with_attention_context)