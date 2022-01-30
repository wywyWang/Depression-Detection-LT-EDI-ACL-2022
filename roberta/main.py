from transformers import AutoTokenizer, AutoModelForSequenceClassification,\
                         AdamW, get_scheduler  
from datasets import Dataset, Features, ClassLabel, Value, features
from dataset import DepressDataset
from sklearn.metrics import f1_score
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

MODEL = 'cardiffnlp/twitter-roberta-base-sentiment'
MODEL_NAME = 'twitter-roberta-base-sentiment'
EPOCHS = 5
LR = 2e-5
BATCH_SIZE = 2

def prepare_data(train_path, dev_path):
    train_data = DepressDataset(train_path, mode='train')
    dev_data = DepressDataset(dev_path, mode='test')
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    dev_dataloader = DataLoader(dev_data, batch_size=1, shuffle=False)
    return train_dataloader, dev_dataloader

def train(train_path, dev_path):
    train_dataloader, dev_dataloader = prepare_data(train_path, dev_path)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader)*EPOCHS)
    criterion = nn.CrossEntropyLoss() 
    # check trained parameters
    print("Parameters to train:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    best_f1 = 0
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        for data in train_dataloader:
            text, label = list(data[0]), data[1].to(device)
            input_text = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            # print(input_text)
            # raise Exception("stop")
            outputs = model(**input_text).logits
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    
        model.eval()
        pred = []
        labels = []
        for data in dev_dataloader:
            text, label = list(data[0]), data[1].to(device)
            input_text = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**input_text).logits
            pred.append(torch.argmax(outputs, dim=-1).cpu().numpy())
            labels.append(label.cpu().numpy())
        f1 = round(f1_score(labels, pred, average='macro'), 4)
        print(f"Epoch: {epoch}, F1 score: {f1}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), f"../model/{MODEL_NAME}_{f1}.pt")


if __name__ == '__main__':
    train(train_path='../data/train.tsv', dev_path='../data/dev_with_labels.tsv')