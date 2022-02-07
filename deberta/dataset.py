from torch.utils.data import Dataset
from tqdm import tqdm
import torch


class DepresionDataset(Dataset):
    def __init__(self, df, mode='train'):
        super().__init__()

        self.data = {}
        self.labels = {}
        self.mode = mode
        if mode == 'test':
            for n, row in df.iterrows():
                self.data[n] = (row['Text_data'])
        else:
            for n, row in df.iterrows():
                self.labels[n] = row['Label']
                self.data[n] = (row['Text_data'], row['Label'])

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        return self.labels

    def __getitem__(self, idx):
        if self.mode == 'test':
            text = self.data[idx]
            return (text)
        else:
            text, label = self.data[idx]
            return (text, torch.tensor(label, dtype=torch.long))