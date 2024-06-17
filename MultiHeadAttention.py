import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        """
        Args:
            d_model:      dimension of embeddings
            n_heads:      number of self attention heads
            dropout:      probability of dropout occurring
        """
        super().__init__()
        assert d_model % n_heads == 0  # ensure an even num of heads
        self.d_model = d_model  # 512 dim
        self.n_heads = n_heads  # 8 heads
        self.d_key = d_model // n_heads  # assume d_value equals d_key

        self.Wq = nn.Linear(d_model, d_model)  # query weights
        self.Wk = nn.Linear(d_model, d_model)  # key weights
        self.Wv = nn.Linear(d_model, d_model)  # value weights
        self.Wo = nn.Linear(d_model, d_model)  # output weights

        self.dropout = nn.Dropout(p=dropout)  # initialize dropout layer

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None):
        """
        Args:
           query:         query vector         (batch_size, q_length, d_model)
           key:           key vector           (batch_size, k_length, d_model)
           value:         value vector         (batch_size, s_length, d_model)
           mask:          mask for decoder

        Returns:
           output:        attention values     (batch_size, q_length, d_model)
           attn_probs:    softmax scores       (batch_size, n_heads, q_length, k_length)
        """
        assert isinstance(key, torch.Tensor), "Expected key to be a tensor"
        assert isinstance(query, torch.Tensor), "Expected query to be a tensor"
        assert isinstance(value, torch.Tensor), "Expected value to be a tensor"

        batch_size = key.size(0)

        # calculate query, key, and value tensors
        Q = self.Wq(query)  # (32, 10, 512) x (512, 512) = (32, 10, 512)
        K = self.Wk(key)  # (32, 10, 512) x (512, 512) = (32, 10, 512)
        V = self.Wv(value)  # (32, 10, 512) x (512, 512) = (32, 10, 512)

        # split each tensor into n-heads to compute attention

        # query tensor
        Q = Q.view(batch_size, -1, self.n_heads, self.d_key).permute(0, 2, 1, 3)
        # (32, 10, 8, 64) -> (32, 8, 10, 64) = (batch_size, n_heads, q_length, d_key)

        # key tensor
        K = K.view(batch_size, -1, self.n_heads, self.d_key).permute(0, 2, 1, 3)
        # (32, 10, 8, 64) -> (32, 8, 10, 64) = (batch_size, n_heads, k_length, d_key)

        # value tensor
        V = V.view(batch_size, -1, self.n_heads, self.d_key).permute(0, 2, 1, 3)
        # (32, 10, 8, 64) -> (32, 8, 10, 64) = (batch_size, n_heads, v_length, d_key)

        # computes attention
        # scaled dot product -> QK^{T}
        scaled_dot_prod = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(self.d_key)
        # (32, 8, 10, 64) x (32, 8, 64, 10) -> (32, 8, 10, 10) = (batch_size, n_heads, q_length, k_length)

        # fill those positions of product as (-1e10) where mask positions are 0
        if mask is not None:
            scaled_dot_prod = scaled_dot_prod.masked_fill(mask == 0, -1e4)

        # apply softmax
        attn_probs = torch.softmax(scaled_dot_prod, dim=-1)

        # multiply by values to get attention
        A = torch.matmul(self.dropout(attn_probs), V)  # (32, 8, 10, 10) x (32, 8, 10, 64) -> (32, 8, 10, 64)
        # (batch_size, n_heads, q_length, k_length) x (batch_size, n_heads, v_length, d_key) -> (batch_size, n_heads, q_length, d_key)

        # reshape attention back to (32, 10, 512)
        A = A.permute(0, 2, 1, 3).contiguous()
        # (32, 8, 10, 64) -> (32, 10, 8, 64)

        A = A.view(batch_size, -1, self.n_heads * self.d_key)
        # (32, 10, 8, 64) -> (32, 10, 8*64) -> (32, 10, 512) = (batch_size, q_length, d_model)

        # push through the final weight layer
        output = self.Wo(A)  # (32, 10, 512) x (512, 512) = (32, 10, 512)

        return output, attn_probs  # return attn_probs for visualization of the scores
