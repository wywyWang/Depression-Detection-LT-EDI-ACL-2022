from transformers import DebertaTokenizer, DebertaModel
import pandas as pd
import logging
import argparse
import os
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import DeBERTaBaseline
from dataset import DepresionDataset


transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)

MODEL_TYPE = "deberta"
PRETRAINED_PATH = 'microsoft/deberta-base'
MAX_SEQUENCE_LENGTH = 512
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_argument():
    opt = argparse.ArgumentParser()
    opt.add_argument("--output_folder_name",
                        type=str,
                        help="path to save model")
    opt.add_argument("--seed_value",
                        type=int,
                        default=42,
                        help="seed value")
    opt.add_argument("--batch_size",
                        type=int,
                        default=32,
                        help="batch size")
    opt.add_argument("--lr",
                        type=int,
                        default=2e-5,
                        help="learning rate")
    opt.add_argument("--epochs",
                        type=int,
                        default=3,
                        help="epochs")
    config = vars(opt.parse_args())
    return config


def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)    # gpu vars


def save(model, config, epoch=None):
    output_folder_name = config['output_folder_name']
    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)

    if epoch is None:
        model_name = output_folder_name + 'model'
        config_name = output_folder_name + 'config'
    else:
        model_name = output_folder_name + str(epoch) + 'model'
        config_name = output_folder_name + str(epoch) + 'config'
    
    torch.save(model.state_dict(), model_name)
    with open(config_name, 'w') as config_file:
        config_file.write(str(config))


if __name__ == '__main__':
    config = get_argument()
    set_seed(config['seed_value'])

    # read tsv data
    # NOTE: should change the column name in dev_with_labels.tsv to 'Text_data'
    df_train = pd.read_csv('../data/train.tsv', sep='\t')
    df_val = pd.read_csv('../data/dev_with_labels.tsv', sep='\t')

    category = {
        'moderate': 0,
        'severe': 1,
        'not depression': 2
    }

    df_train['Label'] = df_train['Label'].map(category)
    df_val['Label'] = df_val['Label'].map(category)

    # load pretrained model
    deberta_tokenizer = DebertaTokenizer.from_pretrained(PRETRAINED_PATH)
    deberta = DebertaModel.from_pretrained(PRETRAINED_PATH)
    for param in deberta.parameters():
        param.requires_grad = False

    deberta_classifier = DeBERTaBaseline()
    criterion = nn.CrossEntropyLoss()
    deberta_classifier_optimizer = torch.optim.Adam(deberta_classifier.parameters(), lr=config['lr'])

    deberta.to(device), deberta_classifier.to(device), criterion.to(device)

    # prepare to dataloader
    train_dataset = DepresionDataset(df_train, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=16)
    val_dataset = DepresionDataset(df_val, mode='train')
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=16)

    # check trained parameters
    print(sum(p.numel() for p in deberta_classifier.parameters() if p.requires_grad))

    # training
    pbar = tqdm(range(config['epochs']), desc='Epoch: ')
    for epoch in pbar:
        deberta_classifier.train()
        total_loss, best_val_f1 = 0, 0
        for loader_idx, item in enumerate(train_dataloader):
            deberta_classifier_optimizer.zero_grad()
            text, label = item[0], item[1].to(device)

            # transform sentences to embeddings via DeBERTa
            input_text = deberta_tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=MAX_SEQUENCE_LENGTH).to(device)
            output_text = deberta(**input_text)
            output_text = output_text.last_hidden_state

            predicted_output = deberta_classifier(output_text)
            
            loss = criterion(predicted_output, label)
            loss.backward()
            deberta_classifier_optimizer.step()

            current_loss = loss.item()
            total_loss += current_loss
            pbar.set_description("Loss: {}".format(round(current_loss, 3)), refresh=True)

        # testing
        y_pred, y_true = [], []
        deberta_classifier.eval()
        for loader_idx, item in enumerate(val_dataloader):
            deberta_classifier_optimizer.zero_grad()
            text, label = item[0], item[1].to(device)

            # transform sentences to embeddings via DeBERTa
            input_text = deberta_tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=MAX_SEQUENCE_LENGTH).to(device)
            output_text = deberta(**input_text)
            output_text = output_text.last_hidden_state

            predicted_output = deberta_classifier(output_text)

            _, predicted_label = torch.topk(predicted_output, 1)

            if len(y_pred) == 0:
                y_pred = predicted_label.cpu().detach().flatten().tolist()
                y_true = label.tolist()
            else:
                y_pred += predicted_label.cpu().detach().flatten().tolist()
                y_true += label.tolist()

        f1 = round(f1_score(y_true, y_pred, average='macro'), 5)

        if f1 >= best_val_f1:
            best_val_f1 = f1
            save(deberta_classifier, config, epoch=epoch)

        with open(config['output_folder_name'] + 'record', 'a') as config_file:
            config_file.write(str(epoch) + ',' + str(round(total_loss/len(train_dataloader), 5)) + ',' + str(f1))
            config_file.write('\n')

    config['total_loss'] = total_loss
    config['val_f1'] = best_val_f1
    save(deberta_classifier, config)