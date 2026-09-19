import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim

class Head(nn.Module):
    def __init__(self, d_model, d_qkv, seq_length):
        super().__init__()
        self.query = nn.Linear(d_model, d_qkv, bias=False)
        self.key = nn.Linear(d_model, d_qkv, bias=False)
        self.value = nn.Linear(d_model, d_qkv, bias=False)

        self.register_buffer('mask', torch.tril(torch.ones(seq_length, seq_length)))

        self.d_qkv = d_qkv

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        T = x.size(-2)

        A = query @ key.transpose(-2, -1)
        A = A.masked_fill(self.mask[:T, :T] == 0, -math.inf)
        A /= math.sqrt(self.d_qkv)

        probs = F.softmax(A, dim = -1)
        return probs @ value
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, seq_length):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_qkv = d_model // num_heads

        self.heads = nn.ModuleList([
            Head(d_model, self.d_qkv, seq_length) for _ in range(num_heads)
        ])

        self.output = nn.Linear(d_model, d_model, bias=False)
    def forward(self, x):
        outputs = [h(x) for h in self.heads]
        C = torch.concat(outputs, dim=-1)
        return self.output(C)
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, seq_length, d_ff):
        super().__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.ReLU = nn.ReLU()
        self.l2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.l1(x)
        x = self.ReLU(x)
        x = self.l2(x)
        return x

class Block(nn.Module):
    def __init__(self, d_model, seq_length, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, seq_length)
        self.ffn = FeedForwardNetwork(d_model, seq_length, 4*d_model)
        self.ln1 = nn.RMSNorm(d_model)
        self.ln2 = nn.RMSNorm(d_model)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model = 128, seq_length = 256, num_heads= 16, layers = 4):
        super().__init__()
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.pos_embedding = nn.Embedding(num_embeddings=seq_length, embedding_dim=d_model)
        self.blocks = nn.Sequential(
            *[Block(d_model, seq_length, num_heads) for _ in range(layers)]
        )

        self.norm = nn.RMSNorm(d_model)
        self.lmhead = nn.Linear(d_model, vocab_size)
        self.seq_length = seq_length

    def forward(self, idx, targets=None):
        token_embds = self.token_embedding(idx)


        T = idx.shape[-1]

        pos_embs = self.pos_embedding(torch.arange(T))
        embds = token_embds + pos_embs
        embds = self.blocks(embds)
        embds = self.norm(embds)
        logits = self.lmhead(embds)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):
            idx_cond = idx[-self.seq_length:]
            logits, _ = self(idx_cond)
            logits = logits[-1, :]
            probs = F.softmax(logits, dim=-1)
            sample = torch.multinomial(probs, num_samples=1)
            idx = torch.concat((idx, sample), dim=-1)
        return idx
    

vocab_size = 100
model = Transformer(vocab_size=vocab_size)
x = torch.randint(vocab_size, (256, ))
y = torch.randint(vocab_size, (256, ))

optimizer = optim.Adam(model.parameters(), lr=0.001)
for _ in range(100):
    logits, loss = model(x,y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print(loss)

    
print(model.generate(torch.tensor([99, 92, 13, 1, 18, 23, 4, 10, 13]), 10))
    




    










        