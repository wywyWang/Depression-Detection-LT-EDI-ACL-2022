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
        df['Label'] = df['Label'].map(dic)
        self.data = {}
        for idx, row in df.iterrows():
            self.data[idx] = (row['Text_data'], row['Label'])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text, label = self.data[idx]
        return (text, torch.tensor(label, dtype=torch.long))