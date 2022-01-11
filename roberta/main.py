from transformers import AutoTokenizer, AutoModelForSequenceClassification,\
                         AdamW, get_scheduler  
from dataset import DepressDataset
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import RAdam
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler
import wandb
import sys

MODEL = {
    "roberta":{
        "pretrain": "cardiffnlp/twitter-roberta-base-sentiment",
        "name": "twitter-roberta-base-sentiment"
    },
    "electra":{
        "pretrain": "google/electra-base-discriminator",
        "name": "electra-base-discriminator"
    }
}
EPOCHS = 30
LR = 2e-5
BATCH_SIZE = 4
SEED = 17
WARM_UP = 5

def set_wandb(model_type):
    wandb.init(project="depression-challenge", entity="nycu_adsl_depression_ycw", tags=[model_type])
    wandb.config = {
        "MODEL": MODEL[model_type]["pretrain"],
        "MODEL_NAME": MODEL[model_type]["name"],
        "EPOCHS": EPOCHS,
        "LR": LR,
        "BATCH_SIZE": BATCH_SIZE,
        "SEED": SEED,
        "WARM_UP": WARM_UP
    }

def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

def prepare_data(train_path, dev_path):
    train_data = DepressDataset(train_path, mode='train')
    dev_data = DepressDataset(dev_path, mode='test')
    # train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=ImbalancedDatasetSampler(train_data))
    dev_dataloader = DataLoader(dev_data, batch_size=1, shuffle=False)
    return train_dataloader, dev_dataloader

def train(model_type, train_path, dev_path):
    set_wandb(model_type)
    set_seed()
    train_dataloader, dev_dataloader = prepare_data(train_path, dev_path)
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL[model_type]["pretrain"], num_labels=3).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL[model_type]["pretrain"])
    optimizer = AdamW(model.parameters(), lr=LR)
    # optimizer = RAdam(model.parameters(), lr=LR)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=WARM_UP, num_training_steps=len(train_dataloader)*EPOCHS)
    criterion = nn.CrossEntropyLoss() 
    # check trained parameters
    print("Parameters to train:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    best_f1 = 0
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        total_loss = 0
        for data in train_dataloader:
            optimizer.zero_grad()
            text, label = list(data[0]), data[1].to(device)
            input_text = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            outputs = model(**input_text).logits
            loss = criterion(outputs, label)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        scheduler.step()
    
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
        precision, recall, f1, support = precision_recall_fscore_support(labels, pred, average='macro', zero_division=1)
        precision = round(precision, 4)
        recall = round(recall, 4)
        f1 = round(f1, 4)
        avg_loss = round(total_loss/len(train_dataloader), 4)
        print(f"Epoch: {epoch}, F1 score: {f1}, Loss: {avg_loss}")
        wandb.log({"epoch": epoch, "f1": f1, "train loss": avg_loss, "precision": precision, "recall": recall, "support": support})
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), f"../model/{MODEL[model_type]['name']}_{f1}.pt")


if __name__ == '__main__':
    model_type = sys.argv[1]
    if model_type not in MODEL.keys():
        raise ValueError(f"{model_type} is not a valid model type [roberta, electra]")
    train(model_type, train_path='../data/train.tsv', dev_path='../data/dev_with_labels.tsv')