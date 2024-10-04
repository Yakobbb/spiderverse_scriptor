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
    
    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensors of integers
        logits = self.token_embedding_table(idx) # (B, T, C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # Loss function
        return logits, loss

    def generate(self, idx, max_new_tokens):
        #idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)
            # sample from the distribution using multinomial
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
