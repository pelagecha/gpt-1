import torch
import torch.nn as nn
from torch.nn import functional as F
import os

# ======================================== HyperParams ===============================================
DATASET = "1984"
batch_size     = 64              # number of indep sequences to be processed in parralel
context_length = 256              # number of previous chars used to predict the following one
max_steps      = 5_000          # max number of steps to complete in training
# eval_freq      = max_steps // 100 # how often to evaluate
eval_freq      = 10
lr             = 3e-4
device = "cuda" if torch.cuda.is_available() else ("cpu" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device}")
eval_iters     = 200
num_embeddings = 384
num_heads = 6
num_layers = 6
dropout = 0.2
# --------------------------------------------------------------------------------------------------------

# 1:23:34

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, f"data/{DATASET}.txt")

with open(file_path, "r") as f:  # Move one level up and into the data directory
    text = f.read()


chars = sorted(list(set(text))) # total num of chars
vocab_size = len(chars)         # num of unique chars
print(f"Dataset length: {len(text)}, Unique chars: {vocab_size}")

charToNum = { char:ind for ind, char in enumerate(chars) } # translate char into number
numToChar = { ind:char for ind, char in enumerate(chars) } # translate number into char

# --------------- word Encoder/Decoder, in practice, use subword tokeniser ---------------
encode = lambda s: [charToNum[c] for c in s]          # take a string, output a list of ints
decode = lambda l: "".join([numToChar[i] for i in l]) # take list of ints, output a string

data = torch.tensor(encode(text), dtype=torch.long) # turn training dataset into tensor of encodings
n = int(0.9*len(data))
train_data = data[:n] # 90% tarin
val_data = data[n:]  # 10% val
    
def get_batch(split):
    data = train_data if split == "train" else val_data
    indices = torch.randint(len(data) - context_length, (batch_size,))
    x = data[indices.unsqueeze(1) + torch.arange(context_length)]  # slice the input in one go
    y = data[indices.unsqueeze(1) + torch.arange(1, context_length + 1)]
    x, y = x.to(device), y.to(device)
    return x, y

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

class LayerNorm: # literally batchnorm, but in call change xmean, xvar dim from 0 to 1
    def __init__(self, dim, eps=1e-5):
        self.eps = eps                 # epsilon to prevent division by 0
        self.gamma = torch.ones(dim)   # batchnorm weight
        self.beta = torch.zeros(dim)   # batchnorm bias

    def __call__(self, x): # forward passif self.training:
        xmean = x.mean(1, keepdim=True) # 1 instead of 0 in batchnorm
        xvar  = x.var(1, keepdim=True)  # 1 instead of 0 in batchnorm
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalise to unit variance
        self.out = self.gamma * xhat + self.beta
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]


class Head(nn.Module):
    """ Single-head self-attention"""
    def  __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(num_embeddings, head_size, bias=False)
        self.query = nn.Linear(num_embeddings, head_size, bias=False)
        self.value = nn.Linear(num_embeddings, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(context_length, context_length))) # creating custom parameter for masking

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        v = self.value(x)   # (B,T,C)

        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5                      # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei) # dropout
        
        out = wei @ v # (B,T,C)
        return out
    

class FeedForward(nn.Module):
    def __init__(self, num_embeddings):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_embeddings, 4 * num_embeddings),
            nn.ReLU(),
            nn.Linear(4 * num_embeddings, num_embeddings), # projection layer back into the residual pathway
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range((num_heads))]) # replicate individual attention heads num_heads times
        self.projection = nn.Linear(num_embeddings, num_embeddings) # projection layer back into the residual pathway
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concat individual heads
        out = self.projection(out)
        out = self.dropout(out) # added dropout
        return out
    

class Block(nn.Module):
    """ Transformer block """
    def __init__(self, num_embeddings, num_heads):
        super().__init__()
        head_size = num_embeddings // num_heads
        self.self_attention = MultiHeadAttention(num_heads, head_size) # self attention
        self.ffwd           = FeedForward(num_embeddings)  # feed forward NN
        self.ln1            = nn.LayerNorm(num_embeddings) # layer normalisation
        self.ln2            = nn.LayerNorm(num_embeddings) # layer normalisation
        
    def forward(self, x):    
        x = x + self.self_attention(self.ln1(x)) # residual connections
        x = x + self.ffwd(self.ln2(x)) # also now has layer norm
        return x

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size,     num_embeddings) # word embedding
        self.position_embedding_table = nn.Embedding(context_length, num_embeddings) # posiional encoding
        self.blocks = nn.Sequential(*[Block(num_embeddings, num_heads=num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(num_embeddings) # final layer normalisation
        self.language_model_head  = nn.Linear(num_embeddings, vocab_size)

    def forward(self, indices, targets=None):
        B, T = indices.shape

        token_emb = self.token_embedding_table(indices) # (B, T, C) - Batch x Time x Channel tensor - batch_size x context_length x vocab_size
        pos_emb   = self.position_embedding_table(torch.arange(T, device=device)) # positional encoding 
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x) # final layer norm
        logits = self.language_model_head(x) # (B,T, vocab_size)


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

model = Transformer()
model = model.to(device)
optimiser = torch.optim.AdamW(model.parameters(), lr=lr)

def main():
    for step in range(max_steps):
        if step == 0:
            print("Starting the training process\n")
        if step % eval_freq == 0 or step == max_steps-1: # just printing stuff
            losses = estimate_loss()
            print(f"{step//eval_freq:2d}/{max_steps//eval_freq:2d}: Train loss = {losses['train']:.4f}, Val loss = {losses['val']:.4f}")

        xb, yb = get_batch("train") # sample a batch of data

        # evaluate data
        logits, loss = model(xb,yb)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

    file_path = os.path.join(current_dir, f"models/{DATASET}.pth")
    torch.save(model.state_dict(), file_path)
    print(loss.item())
    # gen(MODEL=DATASET, text_length=500, terminal=True, writefile=True, timestamp=True)

if __name__ == "__main__":
    main()
