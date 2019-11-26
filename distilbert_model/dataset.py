import pandas as pd
import numpy as np
import ast

from torch.utils.data import Dataset
from transformers import DistilBertTokenizer


class IMDBDataset(Dataset):
    def __init__(self, data_path, seq_length):
        self.data = pd.read_csv(data_path).astype('object')
        self.seq_length = seq_length
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def __getitem__(self, idx):
        review = ast.literal_eval(self.data.at[idx, 'review'])
        review = ['[CLF]'] + review[:self.seq_length - 1]
        review = review + ['[PAD]' for _ in range(self.seq_length - len(review))]
        sentiment = self.data.at[idx, 'sentiment']

        if sentiment == 'positive':
            sentiment = 1
        else:
            sentiment = 0
        return np.array(self.tokenizer.convert_tokens_to_ids(review)), sentiment

    def __len__(self):
        return self.data.shape[0]

