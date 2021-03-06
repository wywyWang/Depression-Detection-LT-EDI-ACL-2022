from transformers import AutoTokenizer, AutoModel, AdamW, get_scheduler  
import pandas as pd
import logging
import os
import sys
import ast
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import DeBERTaBaseline
from dataset import DepresionDataset

import wandb
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils.inference import InferenceModel, MatchFinder
from torchsampler import ImbalancedDatasetSampler

transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)

GPU_NUM = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_NUM

MODEL_TYPE = "deberta"
PRETRAINED_PATH = 'microsoft/deberta-base'
MAX_SEQUENCE_LENGTH = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)    # gpu vars


if __name__ == '__main__':
    '''
    1st argument: model_path, 2nd: best model
    '''
    model_path, best_model_epoch = sys.argv[1], sys.argv[2]
    config = ast.literal_eval(open(model_path + '{}config'.format(best_model_epoch)).readline())
    set_seed(config['seed_value'])

    # read tsv data
    # NOTE: should change the column name in dev_with_labels.tsv to 'Text_data'
    df_val = pd.read_csv('../data/val_summarized.csv')

    category = {
        'moderate': 0,
        'severe': 1,
        'not depression': 2
    }

    inverse_category = {
        0: 'moderate',
        1: 'severe',
        2: 'not depression'
    }

    df_val['Label'] = df_val['Label'].map(category)

    # prepare to dataloader
    val_dataset = DepresionDataset(df_val, mode='test')
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=16)

    # load pretrained model
    pretrained_model = AutoModel.from_pretrained(PRETRAINED_PATH).to(device)
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_PATH)
    for param in pretrained_model.parameters():
        param.requires_grad = False

    net = DeBERTaBaseline(config).to(device)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.RAdam(net.parameters(), lr=config['lr'])
    
    net.load_state_dict(torch.load(model_path + '{}model'.format(best_model_epoch)))

    # check trained parameters
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))

    # testing
    y_pred = []
    net.eval()
    pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
    for loader_idx, item in pbar:
        text = list(item)

        # transform sentences to embeddings via DeBERTa
        input_text = tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=MAX_SEQUENCE_LENGTH).to(device)
        output_text = pretrained_model(**input_text)

        predicted_output, embeddings = net(output_text.last_hidden_state)

        # generate probabilities
        softmax = nn.Softmax(dim=1)
        predicted_output = softmax(predicted_output)

        # generate label
        # _, predicted_label = torch.topk(predicted_output, 1)

        if len(y_pred) == 0:
            y_pred = predicted_output.cpu().detach().tolist()
        else:
            y_pred += predicted_output.cpu().detach().tolist()

        pbar.set_description("y_pred len: {}".format(len(y_pred)), refresh=True)

    answer = pd.DataFrame(y_pred, columns=category.keys())
    answer.to_csv('{}answer.csv'.format(model_path), index=False)