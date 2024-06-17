# importing required libraries
from Transformer_Model import Transformer

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from sklearn.model_selection import train_test_split
from functools import partial
import math
import time

# Set random seed and device
random_seed = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
data_path = '../../PycharmProjects/Transformer-From-Sctatch/dataset/spa.txt'
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Preprocess data
lines = [line.split('\t') for line in lines]
lines = ['\t'.join(line[:2]) for line in lines]

# Create train, val, test split
train_lines, val_test_lines = train_test_split(lines, test_size=0.2, random_state=random_seed, shuffle=True)
val_lines, test_lines = train_test_split(val_test_lines, test_size=0.5, random_state=random_seed, shuffle=True)

# Tokenizers
SRC_LANGUAGE = "en"
TGT_LANGUAGE = "es"
tokenizer = {
    SRC_LANGUAGE: get_tokenizer("spacy", "en_core_web_sm"),
    TGT_LANGUAGE: get_tokenizer("spacy", "es_core_news_sm")
}


# Tokenize lines
def tokenize_lines(lines, src_tokenizer, tgt_tokenizer):
    tokenized_lines = []
    for line in lines:
        src, tgt = line.split('\t')
        src_tokens = src_tokenizer(src)
        tgt_tokens = tgt_tokenizer(tgt)
        tokenized_lines.append((src_tokens, tgt_tokens))
    return tokenized_lines


train_data = tokenize_lines(train_lines, tokenizer[SRC_LANGUAGE], tokenizer[TGT_LANGUAGE])
val_data = tokenize_lines(val_lines, tokenizer[SRC_LANGUAGE], tokenizer[TGT_LANGUAGE])
test_data = tokenize_lines(test_lines, tokenizer[SRC_LANGUAGE], tokenizer[TGT_LANGUAGE])

# Build vocabularies
special_symbols = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
src_vocab_size = 10_000
tgt_vocab_size = 10_000


def yield_tokens(data, lang_idx):
    for tokens in data:
        yield tokens[lang_idx]


vocab = {}
vocab[SRC_LANGUAGE] = build_vocab_from_iterator(yield_tokens(train_data, lang_idx=0), min_freq=1,
                                                specials=special_symbols, special_first=True, max_tokens=src_vocab_size)
vocab[TGT_LANGUAGE] = build_vocab_from_iterator(yield_tokens(train_data, lang_idx=1), min_freq=1,
                                                specials=special_symbols, special_first=True, max_tokens=tgt_vocab_size)
vocab[SRC_LANGUAGE].set_default_index(1)  # UNK_IDX
vocab[TGT_LANGUAGE].set_default_index(1)  # UNK_IDX

# Define collate function
PAD_IDX = 0
BOS_IDX = 2
EOS_IDX = 3
max_seq_len = 100


def collate_fn(batch, vocab):
    batch_size = len(batch)
    srcs, tgts = zip(*batch)
    src_vectors = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)
    tgt_vectors = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)

    for i in range(batch_size):
        src_indices = [BOS_IDX] + [vocab[SRC_LANGUAGE][token] for token in srcs[i]] + [EOS_IDX]
        tgt_indices = [BOS_IDX] + [vocab[TGT_LANGUAGE][token] for token in tgts[i]] + [EOS_IDX]
        src_len = len(src_indices)
        tgt_len = len(tgt_indices)
        src_vectors[i, :src_len] = torch.tensor(src_indices[:max_seq_len], dtype=torch.long)
        tgt_vectors[i, :tgt_len] = torch.tensor(tgt_indices[:max_seq_len], dtype=torch.long)

    return src_vectors, tgt_vectors


# Create DataLoaders
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=partial(collate_fn, vocab=vocab))
val_dataloader = DataLoader(val_data, batch_size=32, shuffle=True, collate_fn=partial(collate_fn, vocab=vocab))
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True, collate_fn=partial(collate_fn, vocab=vocab))

from torch.cuda.amp import GradScaler, autocast

from torch.cuda.amp import GradScaler, autocast


def calculate_accuracy(output, target, pad_idx):
    output_flat = output.argmax(dim=-1).reshape(-1)
    target_flat = target.reshape(-1)
    non_pad_elements = target_flat != pad_idx
    correct = output_flat.eq(target_flat) & non_pad_elements
    accuracy = correct.sum().float() / non_pad_elements.sum().float()
    return accuracy.item()


src_vocab_size = 10_000
tgt_vocab_size = 10_000
d_model = 256
num_heads = 4
num_layers = 4
d_ff = 1024
max_seq_length = 100
dropout = 0.1
num_epochs = 3
pad_idx = 0

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length,
                          dropout).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
scaler = GradScaler()

for epoch in range(num_epochs):
    print(f"Epoch: {epoch + 1}\n------------------------------")
    transformer.train()
    epoch_loss = 0
    epoch_accuracy = 0
    batch_count = 0

    for data in train_dataloader:
        src_data, tgt_data = data
        optimizer.zero_grad()
        with autocast():
            output = transformer(src_data, tgt_data[:, :-1])
            loss = criterion(output.contiguous().reshape(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().reshape(-1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        accuracy = calculate_accuracy(output, tgt_data[:, 1:], pad_idx)
        epoch_accuracy += accuracy
        batch_count += 1
        print(f"Batch: {batch_count}, Training Loss: {loss.item()}, Training Accuracy: {accuracy}")

        torch.cuda.empty_cache()

    print(
        f"Epoch: {epoch + 1}, Average Training Loss: {epoch_loss / batch_count}, Average Training Accuracy: {epoch_accuracy / batch_count}")

    transformer.eval()
    epoch_val_loss = 0
    epoch_val_accuracy = 0
    batch_count = 0
    with torch.no_grad():
        for data in val_dataloader:
            src_data, tgt_data = data
            with autocast():
                output = transformer(src_data, tgt_data[:, :-1])
                loss = criterion(output.contiguous().reshape(-1, tgt_vocab_size),
                                 tgt_data[:, 1:].contiguous().reshape(-1))

            epoch_val_loss += loss.item()
            accuracy = calculate_accuracy(output, tgt_data[:, 1:], pad_idx)
            epoch_val_accuracy += accuracy
            batch_count += 1
            print(f"Batch: {batch_count}, Validation Loss: {loss.item()}, Validation Accuracy: {accuracy}")

    print(
        f"Epoch: {epoch + 1}, Average Validation Loss: {epoch_val_loss / batch_count}, Average Validation Accuracy: {epoch_val_accuracy / batch_count}")

    torch.save(transformer.state_dict(), f'./transformer_state_dict_epoch_{epoch + 1}')

