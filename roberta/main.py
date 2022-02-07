from transformers import AutoTokenizer, AutoModelForSequenceClassification,\
                         AdamW, get_scheduler  
from dataset import DepressDataset
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import RAdam
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler
from pytorch_metric_learning import losses
import wandb
import argparse

MODEL = {
    "roberta":{
        "pretrain": "cardiffnlp/twitter-roberta-base-sentiment",
        "name": "twitter-roberta-base-sentiment"
    },
    "electra":{
        "pretrain": "google/electra-base-discriminator",
        "name": "electra-base-discriminator"
    },
    "deberta":{
        "pretrain": "microsoft/deberta-v3-base",
        "name": "deberta-v3-base"
    }
}
EPOCHS = 20
LR = 2e-5
BATCH_SIZE = 2
SEED = 17
WARM_UP = 5
LOSS_LAMBDA = 0
WEIGHT_DECAY = 0.01
LLR_DECAY = 0.9

def set_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='roberta', help='model type', choices=MODEL.keys())
    parser.add_argument('--train_path', type=str, default='../data/train.tsv', help='train path')
    parser.add_argument('--dev_path', type=str, default='../data/dev_with_labels.tsv', help='dev path')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='epochs')
    parser.add_argument('--lr', type=float, default=LR, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='batch size')
    parser.add_argument('--seed', type=int, default=SEED, help='seed')
    parser.add_argument('--warm_up', type=int, default=WARM_UP, help='warm up')
    return parser.parse_args()

def set_wandb(args):
    wandb.init(project="depression-challenge", entity="nycu_adsl_depression_ycw", tags=[args.model_type])
    wandb.config = {
        "MODEL": MODEL[args.model_type]["pretrain"],
        "MODEL_NAME": MODEL[args.model_type]["name"],
        "EPOCHS": args.epochs,
        "LR": args.lr,
        "BATCH_SIZE": args.batch_size,
        "SEED": args.seed,
        "WARM_UP": WARM_UP,
        "LOSS_LAMBDA": LOSS_LAMBDA,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "LLR_DECAY": LLR_DECAY
    }

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def prepare_data(train_path, dev_path, batch_size):
    train_data = DepressDataset(train_path, mode='train')
    dev_data = DepressDataset(dev_path, mode='test')
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=ImbalancedDatasetSampler(train_data))
    dev_dataloader = DataLoader(dev_data, batch_size=1, shuffle=False)
    return train_dataloader, dev_dataloader

def get_optimizer_grouped_parameters(
    model, model_type, 
    learning_rate, weight_decay, 
    layerwise_learning_rate_decay
):
    no_decay = ["bias", "LayerNorm.weight"]
    # initialize lr for task specific layer
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "classifier" in n or "pooler" in n],
            "weight_decay": 0.0,
            "lr": learning_rate,
        },
    ]
    # initialize lrs for every layer
    num_layers = model.config.num_hidden_layers
    layers = [getattr(model, model_type).embeddings] + list(getattr(model, model_type).encoder.layer)
    layers.reverse()
    lr = learning_rate
    for layer in layers:
        lr *= layerwise_learning_rate_decay
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr,
            },
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]
    return optimizer_grouped_parameters

def train(args):
    set_wandb(args)
    set_seed(args.seed)
    train_dataloader, dev_dataloader = prepare_data(args.train_path, args.dev_path, args.batch_size)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL[args.model_type]["pretrain"], num_labels=3).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL[args.model_type]["pretrain"])
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # grouped_optimizer_params = get_optimizer_grouped_parameters(
    #     model, model_type, 
    #     LR, WEIGHT_DECAY, 
    #     LLR_DECAY
    # )
    optimizer = AdamW(model.parameters(), lr=args.lr)
    # optimizer = AdamW(grouped_optimizer_params, lr=args.lr)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=WARM_UP, num_training_steps=len(train_dataloader)*args.epochs)
    CE_loss = nn.CrossEntropyLoss().to(device)
    Sup_loss = losses.SupConLoss().to(device)
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
            # seq = model.deberta(**input_text)[0]
            # cls = model.classifier
            # emb = cls.dropout(torch.tanh(cls.dense(cls.dropout(seq[:, 0, :]))))
            # loss = (1-LOSS_LAMBDA) * CE_loss(outputs, label) + LOSS_LAMBDA * Sup_loss(emb, label)
            loss = CE_loss(outputs, label)
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
            torch.save(model.state_dict(), f"../model/{MODEL[args.model_type]['name']}_{f1}.pt")
    wandb.log({"best_f1_macro": best_f1})


if __name__ == '__main__':
    args = set_arg()
    train(args)
