import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset

class DepressDataset(Dataset):
    def __init__(self, file_path, mode):
        super().__init__()
        self.mode = mode
        df = pd.read_csv(file_path, sep='\t')
        dic = {'not depression': 0, 'moderate': 1, 'severe': 2}
        if mode != 'test':
            df['label'] = df['label'].map(dic)
            self.labels = df['label'].tolist()
            self.data = {}
            for idx, row in df.iterrows():
                self.data[idx] = (row['PID'], row['Text_data'], row['label'])
        else:
            self.data = {}
            for idx, row in df.iterrows():
                self.data[idx] = (row['PID'], row['Text_data'])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.mode != 'test':
            pid, text, label = self.data[idx]
            return (pid, text, torch.tensor(label, dtype=torch.long))
        else:
            pid, text = self.data[idx]
            return (pid, text)

    def get_labels(self):
        return self.labels
