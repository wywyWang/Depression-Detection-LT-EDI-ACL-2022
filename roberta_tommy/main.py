from transformers import AutoTokenizer, AutoModelForSequenceClassification,\
                         AdamW, get_scheduler  
from dataset import DepressDataset
from model import Model
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler
from pytorch_metric_learning import losses
import wandb
import sys
import os

MODEL = {
    "roberta":{
        "pretrain": "cardiffnlp/twitter-roberta-base-sentiment",
        "name": "twitter-roberta-base-sentiment"
    },
    "electra":{
        "pretrain": "google/electra-base-discriminator",
        "name": "google-electra-base-discriminator"
        # "pretrain": "google/electra-large-discriminator",
        # "name": "google-electra-large-discriminator"
    },
    "deberta":{
        "pretrain": "microsoft/deberta-base",
        "name": "deberta-base"
    },
    "longformer":{
        "pretrain": "allenai/longformer-base-4096",
        "name": "longformer-base"
    }
}

# EPOCHS = 30
# LR = 4e-5
# BATCH_SIZE = 8
# SEED = 17
# WARM_UP = 5
# HIDDEN = 512
# DROPOUT = 0.1
# LAMBDA = 0.7

# EPOCHS = 30
# LR = wandb.config['lr']
# BATCH_SIZE = wandb.config['batch_size']
# SEED = wandb.config['seed']
# WARM_UP = wandb.config['warm_up']
# HIDDEN = wandb.config['hidden']
# DROPOUT = wandb.config['dropout']
# LAMBDA = wandb.config['lambda']

GPU_NUM = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_NUM
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)

def set_wandb(model_type):
    wandb.init(project="depression-challenge", entity="nycu_adsl_depression_ycw", tags=[model_type])
    # wandb.init(project="depression-challenge", entity="nycu_adsl_depression_ycw")

    global EPOCHS, LR, BATCH_SIZE, SEED, WARM_UP, HIDDEN, DROPOUT, LAMBDA, LAMBDA2
    EPOCHS = 20
    LR = 4e-5
    BATCH_SIZE = 8
    SEED = 17
    WARM_UP = 5
    HIDDEN = 512
    DROPOUT = 0.1
    LAMBDA = 0.7
    LAMBDA2 = 0.3

    wandb.config.update({
        "model": MODEL[model_type]["pretrain"],
        "model_name": MODEL[model_type]["name"],
        "epochs": EPOCHS,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "warm_up": WARM_UP,
        "hidden": HIDDEN,
        "dropout": DROPOUT,
        "lambda": LAMBDA,
        "lambda2": LAMBDA2
    })

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
    config = {
        'dropout': DROPOUT,
        'hidden': HIDDEN
    }
    train_dataloader, dev_dataloader = prepare_data(train_path, dev_path)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # model = AutoModelForSequenceClassification.from_pretrained(MODEL[model_type]["pretrain"], num_labels=3)
    # model.to(device)
    model = Model(MODEL[model_type]["pretrain"], config).to(device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL[model_type]["pretrain"])
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=WARM_UP, num_training_steps=len(train_dataloader)*EPOCHS)
    criterion = nn.CrossEntropyLoss() 
    loss_func = losses.SupConLoss().to(device)
    # check trained parameters
    print("Parameters to train:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    best_f1 = 0
    pbar = tqdm(range(EPOCHS), desc='Epoch: ')
    for epoch in pbar:
        model.train()
        total_loss = 0
        for data in train_dataloader:
            optimizer.zero_grad()
            # text, label = list(data[0]), data[1].to(device)
            text, vad_score, label = list(data[0]), data[1].to(device), data[2].to(device)
            input_text = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            # logits = model(**input_text).logits

            logits, pretrained_output, vad_embedding = model(vad_score=vad_score, **input_text)
            # logits, pretrained_output = model(**input_text)

            ce_loss = criterion(logits, label)
            scl_pretrained_loss = loss_func(pretrained_output, label)
            scl_vad_loss = loss_func(vad_embedding, label)    # not change variable before!!!
            # loss = LAMBDA * ce_loss + (1-LAMBDA) * scl_pretrained_loss
            loss = LAMBDA * ce_loss + (1-LAMBDA) * scl_pretrained_loss + (LAMBDA2) * scl_vad_loss

            # loss = criterion(logits, label)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        scheduler.step()
    
        model.eval()
        pred = []
        labels = []
        for data in dev_dataloader:
            # text, label = list(data[0]), data[1].to(device)
            text, vad_score, label = list(data[0]), data[1].to(device), data[2].to(device)
            input_text = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            with torch.no_grad():
                # logits = model(**input_text).logits

                # logits, pretrained_output = model(**input_text)
                logits, pretrained_output, vad_embedding = model(vad_score=vad_score, **input_text)

            pred.append(torch.argmax(logits, dim=-1).cpu().numpy())
            labels.append(label.cpu().numpy())
        precision, recall, f1, support = precision_recall_fscore_support(labels, pred, average='macro', zero_division=1)
        precision = round(precision, 4)
        recall = round(recall, 4)
        f1 = round(f1, 4)
        avg_loss = round(total_loss/len(train_dataloader), 4)
        pbar.set_description(f"Epoch: {epoch}, F1 score: {f1}, Loss: {avg_loss}", refresh=True)
        wandb.log({"epoch": epoch, "f1": f1, "train loss": avg_loss, "precision": precision, "recall": recall, "support": support})
        if f1 > best_f1:
            wandb.run.summary["best_f1_macro"] = f1
            wandb.run.summary["best_precision_macro"] = precision
            wandb.run.summary["best_recall_macro"] = recall
            best_f1 = f1
            if f1 >= 0.5:
                torch.save(model.state_dict(), f"../model/{MODEL[model_type]['name']}_{f1}.pt")


if __name__ == '__main__':
    model_type = sys.argv[1]
    if model_type not in MODEL.keys():
        raise ValueError(f"{model_type} is not a valid model type [roberta, electra, deberta]")
    # train(model_type, train_path='../data/train.tsv', dev_path='../data/dev_with_labels.tsv')
    train(model_type, train_path='../data/train_np.tsv', dev_path='../data/valid_np.tsv')