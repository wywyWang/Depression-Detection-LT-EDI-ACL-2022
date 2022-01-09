from torch.utils.data import Dataset
from tqdm import tqdm
import torch


class DepresionDataset(Dataset):
    def __init__(self, df, mode='train'):
        super().__init__()

        self.data = {}
        self.mode = mode
        if mode == 'test':
            for n, row in df.iterrows():
                self.data[n] = (row['Text_data'], row['summarized_text'])
        else:
            for n, row in df.iterrows():
                self.data[n] = (row['Text_data'], row['summarized_text'], row['Label'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode == 'test':
            text, summarized_text = self.data[idx]
            return (text, summarized_text)
        else:
            text, summarized_text, label = self.data[idx]
            return (text, summarized_text, torch.tensor(label, dtype=torch.long))