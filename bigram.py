import torch
import torch.nn as nn
from torch.nn import functional as F


# ======================================== HyperParams ===============================================
batch_size     = 32              # number of indep sequences to be processed in parralel
context_length = 8               # number of previous chars used to predict the following one
max_steps      = 5_000          # max number of steps to complete in training
eval_freq      = max_steps // 100 # how often to evaluate
lr             = 1e-3
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device}")
eval_iters = 200
num_embeddings = 32
# --------------------------------------------------------------------------------------------------------

# 1:23:34
with open("input.txt", "r") as f:
    text = f.read()

chars = sorted(list(set(text))) # total num of chars
vocab_size = len(chars)         # num of unique chars
print(f"Dataset length: {len(text)} /|=|\ Unique chars: {vocab_size}")

charToNum = { char:ind for ind, char in enumerate(chars) } # translate char into number
numToChar = { ind:char for ind, char in enumerate(chars) } # translate number into char

# --------------- word Encoder/Decoder, in practice, use subword tokeniser ---------------
encode = lambda s: [charToNum[c] for c in s]          # take a string, output a list of ints
decode = lambda l: "".join([numToChar[i] for i in l]) # take list of ints, output a string

data = torch.tensor(encode(text), dtype=torch.long) # turn training dataset into tensor of encodings
n = int(0.9*len(data))
train_data = data[:n] # 90% tarin
val_data = data[n:]  # 10% val
    

def get_batch(split): # produce a random batch (data loading)
    data = train_data if split == "train" else val_data
    indices = torch.randint(len(data) - context_length, (batch_size,)) # generate batch size random offsets
    x = torch.stack([data[i:i+context_length] for i in indices])     # inp  - stack them as rows
    y = torch.stack([data[i+1:i+context_length+1] for i in indices]) # pred - offset by one to allow for prediction
    x, y = x.to(device), y.to(device)
    return x,y


@torch.no_grad()
def estimate_loss(): # evaluate losses
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



class Head(nn.Module):
    """ Single-head self-attention"""
    def  __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(num_embeddings, head_size, bias=False)
        self.query = nn.Linear(num_embeddings, head_size, bias=False)
        self.value = nn.Linear(num_embeddings, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(context_length, context_length))) # creating custom parameter for masking

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        v = self.value(x)   # (B,T,C)

        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5                      # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        out = wei @ v # (B,T,C)
        return out
    

class FeedForward(nn.Module):
    def __init__(self, num_embeddings):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_embeddings, num_embeddings),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range((num_heads))]) # replicate individual attention heads num_heads times
    
    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1) # concat individual heads 
    

class BigramLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, num_embeddings) # just a 2-d lookup table, like chess
        self.position_embedding_table = nn.Embedding(context_length, num_embeddings)
        self.self_attention_head      = MultiHeadAttention(4, num_embeddings//4)    # divide by 4 because using 4 self attention heads
        self.ffwd                     = FeedForward(num_embeddings)
        self.language_modelling_head  = nn.Linear(num_embeddings, vocab_size)

    def forward(self, indices, targets=None):
        B, T = indices.shape

        token_emb = self.token_embedding_table(indices) # (B, T, C) - Batch x Time x Channel tensor - batch_size x context_length x vocab_size
        pos_emb   = self.position_embedding_table(torch.arange(T, device=device)) # positional encoding 
        x = token_emb + pos_emb
        x = self.self_attention_head(x) # self-attention
        x = self.ffwd(x) # just a feed-forward network
        logits = self.language_modelling_head(x) # (B,T, vocab_size)


        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits  = logits.view( B*T, C) # just because of how PyTorch works
            targets = targets.view(B*T   )
            loss = F.cross_entropy(logits, targets) # expected: -ln(1/65) ≈ 4.17438 because weights start off roughly uniform

        return logits, loss # values before softmax
    
    def generate(self, indices, max_new_tokens):
        for _ in range(max_new_tokens):
            cropped_indices = indices[:, -context_length:]     # never pass more than context_length elements
            logits, _ = self(cropped_indices)                  # get the predictions, loss is ignored
            logits = logits[:, -1, :]                          # focus only on the last time step                          (B,  C)
            probs = F.softmax(logits, dim=-1) # (B, C)         # get probabilities from softmax                            (B,  C)
            ind_next = torch.multinomial(probs, num_samples=1) # sample the next predicted character from the distribution (B,  1)
            indices = torch.cat((indices, ind_next), dim=1)    # append sampled index (predicted char) to running sequence (B,T+1)
        return indices
    

# =========================================== Train ==================================================
model = BigramLM()
model = model.to(device)
optimiser = torch.optim.AdamW(model.parameters(), lr=lr)

for step in range(max_steps):
    if step % eval_freq == 0 or step == max_steps-1: # just printing stuff
        losses = estimate_loss()
        print(f"{step//eval_freq:2d}/{max_steps//eval_freq:2d}: Train loss = {losses['train']:.4f}, Val loss = {losses['val']:.4f}")

    xb, yb = get_batch("train") # sample a batch of data

    # evaluate data
    logits, loss = model(xb,yb)
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    optimiser.step()


print(loss.item())
context = torch.zeros((1,1), dtype=torch.long, device=device)
gen = decode(model.generate(context, max_new_tokens=500)[0].tolist())
with open("gen.txt", "w") as f:
    f.write(gen)