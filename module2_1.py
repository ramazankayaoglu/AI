from master_model import MasterModel
from master_tokenizer import MasterTokenizer
import torch

master_tokenizer = MasterTokenizer("tokenizer.json")

prompt = "the capital of united states and the capital of france"

tokens = master_tokenizer.encode(prompt).long()

torch.manual_seed(1)
master_model = MasterModel(
    vocab_size=len(master_tokenizer.vocab),
    embedding_dim=4,
    context_length=32
)
"""
master_model(tokens)
print(master_model(tokens))
print(master_model)
"""

sentence_meanings = master_model(tokens)
print(sentence_meanings.shape)

"""
from transformers import  AutoModelForCausalLM
gemma_model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it")
print(gemma_model)"""


from plot_tokens import plot_tokens

master_sentences = [
    {
    "words" : sentence_meanings.detach().numpy(),
    "labels": master_tokenizer.tokenize(prompt),
    "color": "red"
    }
]

#print(u_tokenizer.tokenize(prompt))

plot_tokens(master_sentences, "Models Context Space")