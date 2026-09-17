import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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

        A = query @ key.T
        A = A.masked_fill(self.mask == 0, -math.inf)
        A = A / math.sqrt(self.d_qkv)

        probs = F.softmax(A, dim = 1)
        O = probs @ value
        return O
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, seq_length):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_kqv = d_model // num_heads
        self.heads = nn.ModuleList([
            Head(d_model, self.d_kqv, seq_length) for _ in range(num_heads)
        ])

        self.output = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        individual_heads = [h(x) for h in self.heads]
        C = torch.concat(individual_heads, dim=1)
        return self.output(C)
    
class FeedForwardNetwork(nn.Module):
    super().__init__()
    





        