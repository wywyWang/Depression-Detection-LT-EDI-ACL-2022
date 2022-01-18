from transformers import AutoTokenizer, AutoModelForSequenceClassification,\
                         AdamW, get_scheduler
from dataset import DepressDataset
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import RAdam
from torch.utils.data import DataLoader
import wandb
import jieba
import jieba.analyse

MODEL1 = 'cardiffnlp/twitter-roberta-base-sentiment'
# MODEL1_NAME = 'twitter-roberta-base-sentiment_0.5478'
MODEL1_NAME = 'twitter-roberta-base-sentiment_0.5579'
MODEL2 = 'google/electra-base-discriminator'
# MODEL2_NAME = 'electra-base-discriminator_0.5485'
MODEL2_NAME = 'electra-base-discriminator_0.5704'
MODEL3 = 'microsoft/deberta-v3-base'
MODEL3_NAME = 'deberta-v3-base_0.5579'
SEED = 17
MASK_TOKEN = '[MASK]'
BIAS = 0.1

def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

def prepare_data(dev_path):
    dev_data = DepressDataset(dev_path, mode='test')
    dev_dataloader = DataLoader(dev_data, batch_size=1, shuffle=False)
    return dev_dataloader

def test(dev_path):
    set_seed()
    dev_dataloader = prepare_data(dev_path)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model1 = AutoModelForSequenceClassification.from_pretrained(MODEL1, num_labels=3).to(device)
    model2 = AutoModelForSequenceClassification.from_pretrained(MODEL2, num_labels=3).to(device)
    # model3 = AutoModelForSequenceClassification.from_pretrained(MODEL3, num_labels=3).to(device)
    # tokenizer1 = AutoTokenizer.from_pretrained(MODEL1, return_special_tokens_mask=True)
    tokenizer2 = AutoTokenizer.from_pretrained(MODEL2, return_special_tokens_mask=True)
    # tokenizer3 = AutoTokenizer.from_pretrained(MODEL3, return_special_tokens_mask=True)

    # model1.load_state_dict(torch.load(f"../model/{MODEL1_NAME}.pt"))
    model2.load_state_dict(torch.load(f"../model/{MODEL2_NAME}.pt"))
    # model3.load_state_dict(torch.load(f"../model/{MODEL3_NAME}.pt"))

    # model1.eval()
    model2.eval()
    # model3.eval()
    pred = []
    labels = []
    for data in dev_dataloader:
        text, label = list(data[0]), data[1].to(device)

        sentence = text[0]
        keywords = jieba.analyse.extract_tags(sentence, topK=999999999, withWeight=True)
        keywords_map = {}
        fully_mask_text, partial_mask_text = [], []
        all_text = sentence.split(' ')
        all_text = [word.strip() for word in all_text if len(word.strip())>0]
        for item in keywords:
            keywords_map[item[0]] = item[1]
        for j in range(len(all_text)):
            fully_mask_text.append(MASK_TOKEN)
            if all_text[j] in keywords_map:
                partial_mask_text.append(MASK_TOKEN)
            else:
                partial_mask_text.append(all_text[j])
        
        fully_mask_text = ' '.join(fully_mask_text)
        partial_mask_text = ' '.join(partial_mask_text)
        # input_text1 = tokenizer1(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        # input_text12 = tokenizer1(fully_mask_text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        # input_text13 = tokenizer1(partial_mask_text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        input_text2 = tokenizer2(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        input_text22 = tokenizer2(fully_mask_text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        input_text23 = tokenizer2(partial_mask_text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        # input_text3 = tokenizer3(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        # input_text32 = tokenizer3(mask_text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            # outputs1 = model1(**input_text1).logits
            # outputs12 = model1(**input_text12).logits
            # outputs13 = model1(**input_text13).logits
            outputs2 = model2(**input_text2).logits
            outputs22 = model2(**input_text22).logits
            outputs23 = model2(**input_text23).logits
            # outputs3 = model3(**input_text3).logits
            # outputs32 = model3(**input_text32).logits
            # tmp1 = torch.subtract(outputs1, outputs12 * BIAS)
            # tmp1 = torch.subtract(outputs1, outputs13 * BIAS)
            # tmp1[tmp1 < 0] = 0
            tmp2 = torch.subtract(outputs2, outputs22 * BIAS)
            tmp2 = torch.subtract(tmp2, outputs23 * BIAS)
            tmp2[tmp2 < 0] = 0
            # tmp3 = torch.subtract(outputs3, outputs32 * BIAS)
            # tmp3[tmp3 < 0] = 0
            # tmp = torch.add(torch.add(tmp1, tmp2), tmp3)
            tmp = tmp2
        pred.append(torch.argmax(tmp, dim=-1).cpu().numpy())
        labels.append(label.cpu().numpy())
    
    precision, recall, f1, support = precision_recall_fscore_support(labels, pred, average='macro', zero_division=1)
    precision = round(precision, 4)
    recall = round(recall, 4)
    f1 = round(f1, 4)
    print(f"F1 score: {f1}, Precision: {precision}, Recall: {recall}")


if __name__ == '__main__':
    test(dev_path='../data/dev_with_labels.tsv')