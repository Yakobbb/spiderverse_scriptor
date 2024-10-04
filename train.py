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
x = torch.rand(5, 3)
print(x)
