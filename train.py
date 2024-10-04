with open("intothespiderverse_script.txt", 'r', encoding='utf-8') as f:
    text = f.read()

print(len(text))
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i,ch in enumerate(chars) }
itos = {i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] #Takes string, outputs list of ints
decode = lambda l: ''.join(itos[i] for i in l)

print(encode("hii there"))
print(decode(encode("hii there")))