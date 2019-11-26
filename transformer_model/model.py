import math
from einops import rearrange

import torch
from torch import nn


class IMDBTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, seq_length, pos_dropout, fc_dropout):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size + 1, d_model)
        self.pos_enc = PositionalEncoding(d_model, pos_dropout, seq_length)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(d_model, 2)

    def forward(self, src, mask):
        x = rearrange(src, 'n s -> s n')
        x = self.pos_enc(self.embed(x) * math.sqrt(self.d_model))
        x = self.encoder(x, src_key_padding_mask=mask)
        x = rearrange(x, 's n e -> n s e')
        x = torch.sum(x, dim=1)  # Batch size, d_model
        return self.dropout(self.fc(x))


# Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
