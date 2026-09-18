import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

        A = query @ key.transpose(-2, -1)
        A /= math.sqrt(self.d_qkv)
        A = A.masked_fill(self.mask == 0, -math.inf)

        probs = F.softmax(A, dim=-1)
        return probs @ value
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, seq_length):
        super().__init__()
        assert d_model % num_heads == 0
        d_qkv = d_model // num_heads
        self.heads = nn.ModuleList([
            Head(d_model, d_qkv, seq_length) for i in range(num_heads)
        ])

        self.output = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        out = [h(x) for h in self.heads]
        C = torch.concat(out, dim=-1)
        return self.output(C)

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff)
        self.ReLU = nn.ReLU()
        self.W2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.W1(x)
        x = self.ReLU(x)
        x = self.W2(x)
        return x

class Block(nn.Module):
    def __init__(self, d_model, num_heads, seq_length):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, seq_length)
        self.ffn = FeedForwardNetwork(d_model, 4 * d_model)

        self.rmsn1 = nn.RMSNorm(d_model)
        self.rmsn2 = nn.RMSNorm(d_model)
    def forward(self, x):
        x = x + self.attn(self.rmsn1(x))
        return x + self.ffn(self.rmsn2(x))
    
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model = 128, num_heads=16, num_layers = 4, seq_length = 32):
        super().__init__()
        
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.positional_embedding = nn.Embedding(num_embeddings=seq_length, embedding_dim=d_model)

        self.transformer_blocks = nn.Sequential(
            *[Block(d_model=d_model, num_heads=num_heads, seq_length=seq_length) for _ in range(num_layers)]
        )
        self.RMSNorm = nn.RMSNorm(d_model)

        self.lmhead = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        #idx:(T,)
        #targets:(T,)
        T = idx.shape
        token_emb = self.token_embedding(idx)

        positions = torch.tensor(torch.arange(T))
        pos_emb = self.positional_embedding(positions)

        emb = token_emb + pos_emb

        emb = self.transformer_blocks(emb)
        emb = self.RMSNorm(emb)
        logits = self.lmhead(emb)

        loss = None

        if targets:
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    










        