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
            if mode != 'test':
                self.data[idx] = (row['Text_data'], row['neg'], row['neu'], row['pos'], row['compound'], row['label'])
            else:
                self.data[idx] = (row['Text_data'], row['neg'], row['neu'], row['pos'], row['compound'])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.mode != 'test':
            text, neg, neu, pos, compound, label = self.data[idx]
            vad_score = [neg, neu, pos, compound]
            return (text, torch.tensor(vad_score), torch.tensor(label, dtype=torch.long))
        else:
            text, neg, neu, pos, compound = self.data[idx]
            vad_score = [neg, neu, pos, compound]
            return (text, torch.tensor(vad_score))

    def get_labels(self):
        return self.labels