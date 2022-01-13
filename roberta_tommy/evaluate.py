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

MODEL1 = 'google/electra-base-discriminator'
# MODEL2_NAME = 'electra-base-discriminator_0.5485'
MODEL1_NAME = 'electra-base-discriminator_0.5711'
BATCH_SIZE = 4
SEED = 17

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
    set_seed()
    val_dataloader = prepare_data(dev_path)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model1 = AutoModelForSequenceClassification.from_pretrained(MODEL1, num_labels=3).to(device)
    tokenizer1 = AutoTokenizer.from_pretrained(MODEL1)
    # check trained parameters

    category = {
        'moderate': 1,
        'severe': 2,
        'not depression': 0
    }

    inverse_category = {
        0: 'moderate',
        1: 'severe',
        2: 'not depression'
    }

    model1.load_state_dict(torch.load(f"../model/{MODEL1_NAME}.pt"))

    model1.eval()
    y_pred = []
    labels = []
    pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
    for idx, data in pbar:
        text, label = list(data[0]), data[1].to(device)
        input_text1 = tokenizer1(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            predicted_output = model1(**input_text1).logits

        softmax = nn.Softmax(dim=1)
        predicted_output = softmax(predicted_output)

        if len(y_pred) == 0:
            y_pred = predicted_output.cpu().detach().tolist()
        else:
            y_pred += predicted_output.cpu().detach().tolist()

        pbar.set_description("y_pred len: {}".format(len(y_pred)), refresh=True)

    answer = pd.DataFrame(y_pred, columns=category.keys())
    answer.to_csv('../prediction/{}answer.csv'.format(MODEL1_NAME), index=False)


if __name__ == '__main__':
    test(dev_path='../data/dev_with_labels.tsv')