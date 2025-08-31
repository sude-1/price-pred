import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

class PriceDataset(Dataset):
    def __init__(self, file_path, window_size=10):
        df = pd.read_csv(file_path, skiprows=2)
        prices = df.iloc[:, 1].astype(float).values  # ikinci kolon fiyat

        prices = pd.Series(prices).ffill().bfill().values

        # Ölçekleyici
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        prices = self.scaler.fit_transform(prices.reshape(-1, 1)).flatten()

        self.prices = prices
        self.window_size = window_size
        self.X, self.y = self.create_sequences()

    def create_sequences(self):
        x, y = [], []
        for i in range(len(self.prices) - self.window_size):
            x.append(self.prices[i:i+self.window_size])
            y.append(self.prices[i+self.window_size])
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
