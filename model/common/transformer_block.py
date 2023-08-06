import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super().__init__()
        self.d_model = d_model

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        # print ('xshape', x.shape, seq_len)
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False).cuda()
        return x


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # print ('attmask score ',  scores.shape)
    # print (scores)

    if mask is not None:
        mask = mask.unsqueeze(1).unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    # print (scores.shape)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)
    # print ('output', scores.shape, v.shape)
    output = torch.matmul(scores, v)
    # print ('attoutput', output.shape)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model, bias=True)
        self.v_linear = nn.Linear(d_model, d_model, bias=True)
        self.k_linear = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model, bias=True)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)

        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # print ('kqv',k.shape, q.shape, v.shape)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # print ('scores.shape',scores.shape)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)
        # print ('concat', concat.shape)

        output = self.out(concat)
        # print ('output', output.shape)

        return output


class MultiHeadAttentionV2(MultiHeadAttention):
    def forward(self, q, k, v, mask=None):
        bs, dim = q.size()
        out = super().forward(q, k, v, mask)
        return torch.reshape(out, [bs, dim])


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=16, dropout=0.1):
        super().__init__()
        # We set d_ff as a default to 2048

        self.linear_1 = nn.Linear(d_model, d_ff, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(self.size), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        # print("===> x before norm: {}".format(x.size()))
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class TransformerBlockOri(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.heads = heads
        self.d_ff = d_ff
        self.dropout = dropout

        self.attention = MultiHeadAttention(heads, d_model, dropout)
        self.norm1 = Norm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm2 = Norm(d_model)

    def forward(self, x):
        # print("===> x after proj: {}".format(x.size()))
        x = x + self.norm1(self.attention(x, x, x))
        # print("===> x after layer1: {}".format(x.size()))
        x = x + self.norm2(self.ffn(x))
        # print("===> x after layer2: {}".format(x.size()))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model_in, d_model, heads, d_ff, dropout=0.1):
        super().__init__()

        self.d_model_in = d_model_in
        self.d_model = d_model
        self.heads = heads
        self.d_ff = d_ff
        self.dropout = dropout

        self.proj = nn.Linear(d_model_in, d_model)
        self.attention = MultiHeadAttentionV2(heads, d_model, dropout)
        self.norm1 = Norm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm2 = Norm(d_model)

    def forward(self, x):
        x = self.proj(x)
        # print("===> x after proj: {}".format(x.size()))
        x = x + self.norm1(self.attention(x, x, x))
        # print("===> x after layer1: {}".format(x.size()))
        x = x + self.norm2(self.ffn(x))
        # print("===> x after layer2: {}".format(x.size()))
        return x
