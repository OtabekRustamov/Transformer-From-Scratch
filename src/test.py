# importing required libraries
import math
import copy
import numpy as np

# torch packages
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from sklearn.model_selection import train_test_split
from functools import partial
import math
from Transformer_Model import Transformer
# Tokenizers
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor

# Definitions for the Embeddings, PositionalEncoding, MultiHeadAttention, PositionwiseFeedForward, EncoderLayer, DecoderLayer, and Transformer classes remain the same
# Ensure you have these classes defined above or import them if they are in another module.

# Load the model and load the state dictionary
model_path = "../train model/transformer_state_dict_epoch_3"
state_dict = torch.load(model_path)

src_vocab_size = 10_000
tgt_vocab_size = 10_000
d_model = 256
num_heads = 4
num_layers = 4
d_ff = 1024
max_seq_length = 100
dropout = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Definitions for the Embeddings, PositionalEncoding, MultiHeadAttention, PositionwiseFeedForward, EncoderLayer, DecoderLayer, and Transformer classes remain the same
# Ensure you have these classes defined above or import them if they are in another module.

# Load the model and load the state dictionary
model_path = "../train model/transformer_state_dict_epoch_3"
state_dict = torch.load(model_path)

src_vocab_size = 10_000
tgt_vocab_size = 10_000
d_model = 256
num_heads = 4
num_layers = 4
d_ff = 1024
max_seq_length = 100
dropout = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transformer_loaded = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length,
                                 dropout).to(device)
transformer_loaded.load_state_dict(state_dict)
transformer_loaded.eval()

# Tokenizers and vocab (assuming these are defined somewhere in your code)
SRC_LANGUAGE = "en"
TGT_LANGUAGE = "es"
tokenizer = {
    SRC_LANGUAGE: get_tokenizer("spacy", "en_core_web_sm"),
    TGT_LANGUAGE: get_tokenizer("spacy", "es_core_news_sm")
}
special_symbols = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
PAD_IDX = 0
BOS_IDX = 2
EOS_IDX = 3


# Assuming the vocabularies were built similarly to how they were in your training code
def yield_tokens(data_iter, language):
    for tokens in data_iter:
        if language == SRC_LANGUAGE:
            yield tokens[0]
        else:
            yield tokens[1]


# Use the same data used for training to build the vocab again
with open('../dataset/spa.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

lines = [line.split('\t') for line in lines]
lines = ['\t'.join(line[:2]) for line in lines]
train_lines, val_test_lines = train_test_split(lines, test_size=0.2, random_state=42, shuffle=True)
val_lines, test_lines = train_test_split(val_test_lines, test_size=0.5, random_state=42, shuffle=True)

train_data = [(tokenizer[SRC_LANGUAGE](src), tokenizer[TGT_LANGUAGE](tgt)) for src, tgt in
              (line.split('\t') for line in train_lines)]
vocab = {
    SRC_LANGUAGE: build_vocab_from_iterator(yield_tokens(train_data, SRC_LANGUAGE), min_freq=1,
                                            specials=special_symbols, special_first=True, max_tokens=src_vocab_size),
    TGT_LANGUAGE: build_vocab_from_iterator(yield_tokens(train_data, TGT_LANGUAGE), min_freq=1,
                                            specials=special_symbols, special_first=True, max_tokens=tgt_vocab_size)
}

vocab[SRC_LANGUAGE].set_default_index(1)  # UNK_IDX
vocab[TGT_LANGUAGE].set_default_index(1)  # UNK_IDX


# Define the translate function
def translate(src):
    src_tokens = tokenizer[SRC_LANGUAGE](src)
    tgt_tokens = ["<BOS>"]

    src_indices = [BOS_IDX] + [vocab[SRC_LANGUAGE][token] for token in src_tokens] + [EOS_IDX]
    src_vectors = torch.tensor(src_indices + [PAD_IDX] * (max_seq_length - len(src_indices)), dtype=torch.long,
                               device=device).unsqueeze(0)

    for i in range(max_seq_length):
        tgt_indices = [vocab[TGT_LANGUAGE][token] for token in tgt_tokens]
        tgt_vectors = torch.tensor(tgt_indices + [PAD_IDX] * (max_seq_length - len(tgt_indices)), dtype=torch.long,
                                   device=device).unsqueeze(0)

        output = transformer_loaded(src_vectors, tgt_vectors)
        idx = torch.argmax(nn.functional.softmax(output, dim=2)[0][i]).item()

        tgt_token = vocab[TGT_LANGUAGE].lookup_token(idx)
        tgt_tokens.append(tgt_token)

        if idx == EOS_IDX:
            break

    return " ".join(tgt_tokens[1:-1])  # Skip <BOS> and <EOS> tokens


# Test the translation function
print(translate("My name is John."))
print(translate("I have three books and two pens."))
print(translate("I am learning Spanish."))
