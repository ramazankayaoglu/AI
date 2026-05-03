import torch

from master_model import MasterModel
from master_tokenizer import MasterTokenizer
import master_tokenizer

master_tokenizer = MasterTokenizer("tokenizer.json")

prompt = "the capital of united"

tokens = master_tokenizer.encode(prompt)

print(tokens.shape)