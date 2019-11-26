import pandas as pd
import numpy as np
import ast

from torch.utils.data import Dataset


class IMDBDataset(Dataset):
    def __init__(self, data_path, seq_length):
        self.data = pd.read_csv(data_path).astype('object')
        self.seq_length = seq_length

    def __getitem__(self, idx):
        review = ast.literal_eval(self.data.at[idx, 'review'])
        review = review[:self.seq_length]

        review = [0 for _ in range(self.seq_length - len(review))] + review
        sentiment = self.data.at[idx, 'sentiment']

        if sentiment == 'positive':
            sentiment = 1
        else:
            sentiment = 0
        return np.array(review), sentiment

    def __len__(self):
        return self.data.shape[0]

