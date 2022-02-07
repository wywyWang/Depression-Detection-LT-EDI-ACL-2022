from transformers import AutoTokenizer, AutoModelForSequenceClassification,\
                         AdamW, get_scheduler  
from dataset import DepressDataset
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RAdam
from torch.utils.data import DataLoader
import argparse

MODEL1 = 'cardiffnlp/twitter-roberta-base-sentiment'
MODEL1_NAME = 'twitter-roberta-base-sentiment_0.5686'
MODEL2 = 'google/electra-base-discriminator'
MODEL2_NAME = 'electra-base-discriminator_0.5704'
MODEL3 = 'microsoft/deberta-v3-base'
MODEL3_NAME = 'deberta-v3-base_0.5579'

def set_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='../data/dev_with_labels.tsv')
    parser.add_argument('--mode', type=str, default='dev')
    return parser.parse_args()

def prepare_data(path, mode='dev'):
    data = DepressDataset(path, mode)
    dataloader = DataLoader(data, batch_size=1, shuffle=False)
    return dataloader

def ensemble(args):
    path, mode = args.path, args.mode
    dataloader = prepare_data(path, mode)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model1 = AutoModelForSequenceClassification.from_pretrained(MODEL1, num_labels=3).to(device)
    model2 = AutoModelForSequenceClassification.from_pretrained(MODEL2, num_labels=3).to(device)
    model3 = AutoModelForSequenceClassification.from_pretrained(MODEL3, num_labels=3).to(device)
    tokenizer1 = AutoTokenizer.from_pretrained(MODEL1)
    tokenizer2 = AutoTokenizer.from_pretrained(MODEL2)
    tokenizer3 = AutoTokenizer.from_pretrained(MODEL3)
    # check trained parameters

    model1.load_state_dict(torch.load(f"../model/{MODEL1_NAME}.pt"))
    model2.load_state_dict(torch.load(f"../model/{MODEL2_NAME}.pt"))
    model3.load_state_dict(torch.load(f"../model/{MODEL3_NAME}.pt"))

    model1.eval()
    model2.eval()
    model3.eval()
    pred, pred_dev = [], []
    pids, labels = [], []
    for data in dataloader:
        if mode == 'dev':
            pid, text, label = data[0][0], list(data[1]), data[2]
        else:
            pid, text = data[0][0], list(data[1])
        input_text1 = tokenizer1(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        input_text2 = tokenizer2(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        input_text3 = tokenizer3(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs1 = model1(**input_text1).logits
            outputs2 = model2(**input_text2).logits
            outputs3 = model3(**input_text3).logits
            tmp = torch.add(torch.add(outputs1, outputs2), outputs3) / 3
        if mode == 'dev':
            sf = F.softmax(tmp, dim=1)
            pred.append(torch.argmax(sf, dim=-1).cpu().numpy())
            pred_dev.append(sf.cpu().numpy())
            labels.append(label.cpu().numpy())
            pids.append(pid)
        else:
            pred.append(F.softmax(tmp, dim=1).cpu().numpy())
            pids.append(pid)
    
    if mode == 'dev':
        precision, recall, f1, support = precision_recall_fscore_support(labels, pred, average='macro', zero_division=1)
        precision = round(precision, 4)
        recall = round(recall, 4)
        f1 = round(f1, 4)
        print(f"F1 score: {f1}, Precision: {precision}, Recall: {recall}")
        pred = np.array(pred_dev).T
    else:
        pred = np.array(pred).T
    
    prob_df = pd.DataFrame({'PID': pids, 'not depression': pred[0][0], 'moderate': pred[1][0], 'severe': pred[2][0]})
    prob_df.to_csv(f"../result/ensemble_{mode}.csv", index=False)

if __name__ == '__main__':
    args = set_arg()
    ensemble(args)