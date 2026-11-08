import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Head(nn.Module):
    def __init__(self, d_model, d_kqv, L):
        super().__init__()
        self.q = nn.Linear(d_model, d_kqv, bias=False)
        self.k = nn.Linear(d_model, d_kqv, bias=False)
        self.v = nn.Linear(d_model, d_kqv, bias=False)

        self.register_buffer('mask', torch.tril(torch.ones(L, L)))
        self.d_kqv = d_kqv
    def forward(self, x):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        s = Q @ K.transpose(-2, -1)
        masked_s = s.masked_fill(self.mask == 0, -math.inf)
        softmax_s = F.softmax(masked_s/math.sqrt(self.d_kqv), dim=-1)
        return softmax_s @ V
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, L):
        super().__init__()
        assert d_model % n_head == 0
        self.d_kqv = d_model // n_head
        self.heads = nn.ModuleList([Head(d_model, self.d_kqv, L) for _ in range(n_head)])
        self.output = nn.Linear(d_model, d_model, bias=False)
    def forward(self, x):
        outs = []
        for head in self.heads:
            outs.append(head(x))
        out_concat = torch.concat(outs, dim=-1)
        return self.output(out_concat)
    
class FF(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.f1 = nn.Linear(d_model, 4 * d_model)
        self.relu = nn.ReLU()
        self.f2 = nn.Linear(4 * d_model, d_model)

    def forward(self, x):
        x = self.f1(x)
        x = self.relu(x)
        return self.f2(x)
    
class Block(nn.Module):
    def __init__(self, d_model, n_head, L):
        super().__init__()
        self.attention1 = MultiHeadAttention(n_head, d_model, L)
        self.rms1 = nn.RMSNorm(normalized_shape=d_model)
        self.dense = FF(d_model)
        self.rms2 = nn.RMSNorm(normalized_shape=d_model)
        
    def forward(self, x):
        x = x + self.attention1(self.rms1(x))
        x = x + self.dense(self.rms2(x))
        return x
    
class Transformer(nn.Module):
    def __init__(self, d_model, layer_count, L, vocab_size, n_head):
        super().__init__()
        self.vocab_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.pos_embedding = nn.Embedding(num_embeddings=L, embedding_dim=d_model)

        blocks = [Block(d_model, n_head, L) for _ in range(layer_count)]
        self.layers = nn.Sequential(*blocks)
        self.norm = nn.RMSNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)
        self.out.weight = self.vocab_embedding.weight
    def forward(self, x):
        v_embedding = self.vocab_embedding(x)
        p_embedding = self.pos_embedding(x)
        embedding = v_embedding + p_embedding
        embedding = self.layers(embedding)
        embedding = self.norm(embedding)
        logits = self.out(embedding)
        return logits







        