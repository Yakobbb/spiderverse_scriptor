'''
This project follows the "Let's build GPT: from scratch, in code, spelled out
guide by Andrej Karpathy. Comments were written for my own understanding of the
material. Moreover, I train this model on the scripts of the Spiderverse Movies, rather
than Shakespeare.
'''

import torch
import torch.nn as nn
from torch.nn import functional as F

with open("intothespiderverse_script.txt", 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i,ch in enumerate(chars) } # Maps char to sorted int index 
itos = {i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # Takes string, outputs list of ints
decode = lambda l: ''.join(itos[i] for i in l)

'''
GPT uses something called tiktoken, with roughly 50,000 sub-word level tokens
'''

# Split data into training + testing set
data = torch.tensor(encode(text), dtype=torch.long)
n = int(.9 * len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8 # Defines maximum context length
batch_size = 4 # Number of independent sequences processed in parallel
# In total, we get 32 training samples

def get_batch(split):
    # Create small batch of data using inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # 4 random nums

    x = torch.stack([data[i:i+block_size] for i in ix]) # Creates rows, resulting in 4 x 8 tensor
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')

# We'll use a very simple neural network for training: the bigram language model

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Tokens read off of the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets):
        # idx and targets are both (B, T) tensors of integers
        logits = self.token_embedding_table(idx) # (B, T, C)

        return logits

m = BigramLanguageModel(vocab_size)
out = m(xb, yb)