import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math


class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        """
        Args:
          vocab_size:     size of vocabulary
          d_model:        dimension of embeddings
        """
        # inherit from nn.Module
        super().__init__()

        # embedding look-up table
        self.lut = nn.Embedding(vocab_size, d_model)

        # dimension of embeddings
        self.d_model = d_model

    def forward(self, x: Tensor):
        """
        Args:
          x:              input Tensor (batch_size, seq_length)
        Returns:
                          embedding vector
        """
        # embeddings by constant sqrt(d_model)
        return self.lut(x) * math.sqrt(self.d_model)