import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout, max_length):
        """
        Args:
          d_model:      dimension of embeddings
          dropout:      randomly zeroes-out some of the input
          max_length:   max sequence length
        """
        # inherit from Module
        super().__init__()

        # initialize dropout
        self.dropout = nn.Dropout(p=dropout)

        # create tensor of 0s
        pe = torch.zeros(max_length, d_model)

        # create position column
        k = torch.arange(0, max_length).unsqueeze(1)

        # calc divisor for positional encoding
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        # calc sine on even indices
        pe[:, 0::2] = torch.sin(k * div_term)

        # calc cosine on odd indices
        pe[:, 1::2] = torch.cos(k * div_term)

        # add dimension
        pe = pe.unsqueeze(0)

        # buffer are saved in state_dict but not trained by the optimizer
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor):
        """
        Args:
          x:        embeddings (batch_size, seq_length, d_model)

        Returns:
                    embeddings + positional encodings (batch_size, seq_length, d_model)
        """
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)

        # perform dropout
        return self.dropout(x)

