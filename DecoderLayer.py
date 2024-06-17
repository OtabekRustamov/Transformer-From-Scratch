
# torch packages
import torch.nn as nn
from torch import Tensor
from MultiHeadAttention import MultiHeadAttention
from PositionwiseFeedForward import PositionwiseFeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float):
        """
        Args:
            d_model:      dimension of embeddings
            n_heads:      number of heads
            d_ffn:        dimension of feed-forward network
            dropout:      probability of dropout occurring
        """
        super().__init__()
        # masked multi-head attention sublayer
        self.masked_attention = MultiHeadAttention(d_model, n_heads, dropout)
        # layer norm for masked multi-head attention
        self.masked_attn_layer_norm = nn.LayerNorm(d_model)

        # multi-head attention sublayer
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        # layer norm for multi-head attention
        self.attn_layer_norm = nn.LayerNorm(d_model)

        # position-wise feed-forward network
        self.positionwise_ffn = PositionwiseFeedForward(d_model, d_ffn, dropout)
        # layer norm for position-wise fnn
        self.fnn_layer_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, trg: Tensor, src: Tensor, trg_mask: Tensor, src_mask: Tensor):
        """
        Args:
            trg:          embedded sequences                (batch_size, trg_seq_length, d_model)
            src:          embedded sequences                (batch_size, src_seq_length, d_model)
            trg_mask:     mask for the sequences            (batch_size, 1, trg_seq_length, trg_seq_length)
            src_mask:     mask for the sequences            (batch_size, 1, 1, src_seq_length)

        Returns:
            trg:          sequences after self-attention    (batch_size, trg_seq_length, d_model)
            attn_probs:   self-attention softmax scores     (batch_size, n_heads, trg_seq_length, src_seq_length)
        """
        # pass trg embeddings through masked multi-head attention
        _trg, attn_probs = self.masked_attention(trg, trg, trg, trg_mask)

        # residual add and norm
        trg = self.masked_attn_layer_norm(trg + self.dropout(_trg))

        # pass trg and src embeddings multi-head attention
        _trg, attn_probs = self.attention(trg, src, src, src_mask)

        # residual add and norm
        trg = self.attn_layer_norm(trg + self.dropout(_trg))

        # position-wise feed-forward network
        _trg = self.positionwise_ffn(trg)

        # residual add and norm
        trg = self.fnn_layer_norm(trg + self.dropout(_trg))
        return trg, attn_probs