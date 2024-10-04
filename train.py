import torch

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

block_size = 8 # Length of chunks that will be plugged into transformer
