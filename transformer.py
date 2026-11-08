import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Head(nn.Module):
    def __init__(self, d_model, d_kqv, L):
        super.__init__()
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






        