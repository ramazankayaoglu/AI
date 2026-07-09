from turtle import forward
import torch
import torch.nn as nn

from master_layer_normalization import  MasterLayerNormalization
from master_decoder_block import MasterDecoderBlock
from master_embedding import MasterEmbedding

class MasterModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, context_length, num_layers):
        super().__init__()
        self.embedding = MasterEmbedding(vocab_size, embedding_dim, context_length)
        self.layers = nn.Sequential(*[MasterDecoderBlock(embedding_dim, num_heads, context_length) for _ in range(num_layers)])

        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x) #dictionary meanings of the tokens(words)
        
        x = self.layers(x)
        x = self.lm_head(x)

        return x
