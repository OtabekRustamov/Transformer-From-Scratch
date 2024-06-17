# torch packages
import torch.nn as nn
from torch import Tensor
import torch
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k
import io
from Embeddings import Embeddings
from PositionalEncoding import PositionalEncoding
from EncoderLayer import EncoderLayer
from DecoderLayer import DecoderLayer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super().__init__()
        self.encoder_embeddings = Embeddings(src_vocab_size, d_model)
        self.decoder_embeddings = Embeddings(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_length)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for layer in range(num_layers)])
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for layer in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def make_mask(self, src, tgt):
        # assign 1 to tokens that need attended to and 0 to padding tokens, then add 2 dimensions
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        # src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        # assign True to tokens that need attended to and False to padding tokens, then add 2 dimensions
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)

        # generate subsequent mask
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length, device=device), diagonal=1)).bool()

        # bitwise "and" operator | 0 & 0 = 0, 1 & 1 = 1, 1 & 0 = 0
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):

        # create source and target masks
        src_mask, tgt_mask = self.make_mask(src, tgt)

        # push the src through the encoder layers
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embeddings(src)))

        # decoder output and attention probabilities
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embeddings(tgt)))

        # pass the sequences through each encoder
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output, _ = enc_layer(enc_output, src_mask)

        # pass the sequences through each decoder
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output, _ = dec_layer(dec_output, enc_output, tgt_mask, src_mask)

        # set output layer
        output = self.fc(dec_output)
        return output

