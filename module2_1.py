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
#print(sentence_meanings.shape)

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

#plot_tokens(master_sentences, "Models Context Space")

#print(sentence_meanings)


the_position = [-1.5256, -0.7502,  0.6540,  1.6095]
capital_position = [ 0.9326, -0.2774, -0.4988,  1.4560]

hardness_distance = abs(the_position[0] - capital_position[0])
brightness_distance = abs(the_position[1] - capital_position[1])
redness_distance = abs(the_position[2] - capital_position[2])
blueness_distance = abs(the_position[3] - capital_position[3])

#print(hardness_distance, brightness_distance, redness_distance, blueness_distance)

total_distance = hardness_distance + brightness_distance + redness_distance + blueness_distance #manhattan distance olarak geçer


apple = [-1.5256, -0.7502, -0.6540, -1.6095]
real_apple = [0.5, -0.7502, -0.6540, -1.6095]

def is_apple(position, real_position):
    dist1 = position[0] - real_position[0]

    print(dist1)

    #burada eğer değer doğru değilse tuning yani eğitim vererek doğru değere ulaştırmaya çalışıyor
    if dist1 > 0:
        apple[0] -= 0.5
    else:
        apple[0] += 0.5

    return dist1 > 0 and dist1 < 0.5


for i in range(1,5):
    print(is_apple(apple, real_apple))
    if(is_apple(apple, real_apple) == True):
        print(is_apple(apple, real_apple))
    else:
        is_apple(apple, real_apple)