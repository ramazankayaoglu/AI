from json import decoder
import torch
from master_model import MasterModel
from master_tokenizer import MasterTokenizer
import master_tokenizer

master_tokenizer = MasterTokenizer("tokenizer.json")

prompt = "the capital of united"

tokens = master_tokenizer.encode(prompt)

torch.manual_seed(1)    


master_model = MasterModel(vocab_size=len(master_tokenizer.vocab), embedding_dim= 4, num_heads=4, context_length=32, num_layers = 3)

out = master_model(tokens)
#print(out)
#print(out.shape)

#print(master_model)
#print(out[0])

import torch

#probs = torch.softmax(out[-1], dim = -1)
#print(probs)

#max_prob, max_index = torch.max(probs, dim = -1)
#print(max_prob, max_index)


from transformers import AutoTokenizer, AutoModelForCausalLM

q_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
q_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

q_tokens = q_tokenizer.encode(prompt)

#print(q_tokens)

q_tokens2 = q_tokenizer.tokenize(prompt)
#print(q_tokens2)

q_out = q_model(torch.tensor([q_tokens]))
#print(q_out)


#print(q_out.logits.shape)

last_token = q_out.logits[0, -1, :]

probs = torch.softmax(last_token, dim = -1)
max_prob, max_index = torch.max(probs, dim = -1)

print(max_prob, max_index)

print(q_tokenizer.decode(max_index))