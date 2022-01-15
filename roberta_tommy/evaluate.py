from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import torch
import torch.nn as nn
from dataset import DepressDataset
from sklearn.metrics import precision_recall_fscore_support
import os
import sys
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Model

MODEL = 'google/electra-base-discriminator'
MODEL_NAME = 'google-electra-base-discriminator_0.5706'
EPOCHS = 30
LR = 2e-5
BATCH_SIZE = 4
SEED = 17
WARM_UP = 5
HIDDEN = 768
DROPOUT = 0.1
LAMBDA = 0.5

GPU_NUM = '0'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_NUM
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)

def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

def prepare_data(dev_path):
    dev_data = DepressDataset(dev_path, mode='test')
    dev_dataloader = DataLoader(dev_data, batch_size=1, shuffle=False)
    return dev_dataloader

def test(dev_path):
    config = {
        'dropout': DROPOUT,
        'hidden': HIDDEN
    }
    set_seed()
    val_dataloader = prepare_data(dev_path)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model1 = AutoModelForSequenceClassification.from_pretrained(MODEL1, num_labels=3).to(device)
    model = Model(MODEL, config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    # check trained parameters

    category = {
        'moderate': 1,
        'severe': 2,
        'not depression': 0
    }

    inverse_category = {
        1: 'moderate',
        2: 'severe',
        0: 'not depression'
    }

    model.load_state_dict(torch.load(f"../model/{MODEL_NAME}.pt"))

    model.eval()
    y_pred = []
    labels = []
    pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
    for idx, data in pbar:
        text, label = list(data[0]), data[1].to(device)
        input_text = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            # predicted_output = model1(**input_text1).logits
            predicted_output, pooled_output = model(**input_text)

        softmax = nn.Softmax(dim=1)
        predicted_output = softmax(predicted_output)

        if len(y_pred) == 0:
            y_pred = predicted_output.cpu().detach().tolist()
        else:
            y_pred += predicted_output.cpu().detach().tolist()

        pbar.set_description("y_pred len: {}".format(len(y_pred)), refresh=True)

    answer = pd.DataFrame(y_pred, columns=category.keys())
    answer.to_csv('../prediction/{}answer.csv'.format(MODEL_NAME), index=False)


if __name__ == '__main__':
    test(dev_path='../data/dev_with_labels.tsv')