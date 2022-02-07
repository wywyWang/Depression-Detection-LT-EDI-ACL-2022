from transformers import AutoTokenizer, AutoModel
import pandas as pd
import logging
import argparse
import os
from sklearn.metrics import precision_recall_fscore_support
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


wandb.init(project="depression-challenge", entity="nycu_adsl_depression_ycw", tags=["deberta"])
# artifact = wandb.use_artifact('nycu_adsl_depression_ycw/depression-challenge/dataset:latest', type='dataset')
# artifact_dir = artifact.download()
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
                        default=1e-4,
                        help="learning rate")
    opt.add_argument("--lambda",
                        type=int,
                        default=0.5,
                        help="learning rate")
    opt.add_argument("--epochs",
                        type=int,
                        default=30,
                        help="epochs")
    opt.add_argument("--hidden",
                        type=int,
                        default=256,
                        help="dimension of hidden state")
    opt.add_argument("--dropout",
                        type=int,
                        default=0.1,
                        help="dropout rate")
    opt.add_argument("--head",
                        type=int,
                        default=4,
                        help="number of heads")
    opt.add_argument("--optimizer",
                        type=str,
                        default='adam',
                        help="optimizer")
    opt.add_argument("--n_layers",
                        type=int,
                        default=2,
                        help="number of layers")
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
    wandb.config.update(config.copy())
    set_seed(config['seed_value'])

    # read tsv data
    # NOTE: should change the column name in dev_with_labels.tsv to 'Text_data'
    # df_train = pd.read_csv('../data/train.tsv', sep='\t')
    # df_val = pd.read_csv('../data/dev_with_labels.tsv', sep='\t')
    df_train = pd.read_csv('../data/train_summarized.csv')
    df_val = pd.read_csv('../data/val_summarized.csv')

    category = {
        'moderate': 0,
        'severe': 1,
        'not depression': 2
    }

    df_train['Label'] = df_train['Label'].map(category)
    df_val['Label'] = df_val['Label'].map(category)

    # prepare to dataloader
    train_dataset = DepresionDataset(df_train, mode='train')
    train_dataloader = DataLoader(train_dataset, sampler=ImbalancedDatasetSampler(train_dataset), batch_size=config['batch_size'], shuffle=False, num_workers=16)
    val_dataset = DepresionDataset(df_val, mode='train')
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=16)

    # load pretrained model
    pretrained_model = AutoModel.from_pretrained(PRETRAINED_PATH).to(device)
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_PATH)
    for param in pretrained_model.parameters():
        param.requires_grad = False

    net = DeBERTaBaseline(config).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    loss_func = losses.SupConLoss().to(device)
    if config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'])
    elif config['optimizer'] == 'radam':
        optimizer = torch.optim.RAdam(net.parameters(), lr=config['lr'])
    # scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader)*config['epochs'])

    # check trained parameters
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))

    # training
    pbar = tqdm(range(config['epochs']), desc='Epoch: ')
    best_val_f1 = 0
    for epoch in pbar:
        net.train()
        total_loss = 0
        for loader_idx, item in enumerate(train_dataloader):
            optimizer.zero_grad()
            text, label = list(item[0]), item[1].to(device)

            # transform sentences to embeddings via DeBERTa
            input_text = tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=MAX_SEQUENCE_LENGTH).to(device)
            output_text = pretrained_model(**input_text)

            predicted_output, embeddings = net(output_text.last_hidden_state)
            ce_loss = criterion(predicted_output, label)

            # embeddings = net(output_text.last_hidden_state)
            scl_loss = loss_func(embeddings, label)
            loss = config['lambda'] * ce_loss + (1 - config['lambda']) * scl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            # scheduler.step()

            current_loss = loss.item()
            total_loss += current_loss
            pbar.set_description("Status: Train, Loss: {}".format(round(current_loss, 3)), refresh=True)


        # testing
        pbar.set_description("Status: Val", refresh=True)
        y_pred, y_true = [], []
        net.eval()

        # # metric learning
        # match_finder = MatchFinder(distance=CosineSimilarity(), threshold=0.7)
        # inference_model = InferenceModel(net, match_finder=match_finder)
        # knn_labels = []

        # for loader_idx, item in enumerate(train_dataloader):
        #     text, label = list(item[0]), item[1].to(device)

        #     knn_labels.append(label.cpu().tolist())

        #     # transform sentences to embeddings via DeBERTa
        #     input_text = tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=MAX_SEQUENCE_LENGTH).to(device)
        #     output_text = pretrained_model(**input_text)

        #     if loader_idx == 0:
        #         inference_model.train_knn(output_text.last_hidden_state)
        #     else:
        #         inference_model.add_to_knn(output_text.last_hidden_state)

        # knn_labels = [item for sublist in knn_labels for item in sublist]

        # for loader_idx, item in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
        #     text, label = list(item[0]), item[1].to(device)

        #     # transform sentences to embeddings via DeBERTa
        #     input_text = tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=MAX_SEQUENCE_LENGTH).to(device)
        #     output_text = pretrained_model(**input_text)

        #     # predicted_output = net(output_text.last_hidden_state)
        #     # _, predicted_label = torch.topk(embeddings, 1)

        #     distances, indices = inference_model.get_nearest_neighbors(output_text.last_hidden_state, k=10)

        #     neighbor_labels = [0 for _ in range(3)]
        #     for knn_index in indices[0]:
        #         neighbor_labels[knn_labels[knn_index]] += 1
        #     predicted_label = neighbor_labels.index(max(neighbor_labels))

        #     if len(y_pred) == 0:
        #         y_pred = [predicted_label]
        #         y_true = label.tolist()
        #     else:
        #         y_pred += [predicted_label]
        #         y_true += label.tolist()

        # original ce
        for loader_idx, item in enumerate(val_dataloader):
            text, label = list(item[0]), item[1].to(device)

            # transform sentences to embeddings via DeBERTa
            input_text = tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=MAX_SEQUENCE_LENGTH).to(device)
            output_text = pretrained_model(**input_text)

            predicted_output, embeddings = net(output_text.last_hidden_state)
            _, predicted_label = torch.topk(predicted_output, 1)

            if len(y_pred) == 0:
                y_pred = predicted_label.cpu().detach().flatten().tolist()
                y_true = label.tolist()
            else:
                y_pred += predicted_label.cpu().detach().flatten().tolist()
                y_true += label.tolist()

        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=1)
        f1 = round(f1, 5)
        precision = round(precision, 5)
        recall = round(recall, 5)

        wandb.log({'epoch': epoch, 'train loss': round(total_loss/len(train_dataloader), 5), 'f1': f1, 'precision': precision, 'recall': recall, 'support': support})

        if f1 >= best_val_f1:
            wandb.run.summary["best_f1_macro"] = f1
            wandb.run.summary["best_precision_macro"] = precision
            wandb.run.summary["best_recall_macro"] = recall
            best_val_f1 = f1
            save(net, config, epoch=epoch)

        with open(config['output_folder_name'] + 'record', 'a') as config_file:
            config_file.write(str(epoch) + ',' + str(round(total_loss/len(train_dataloader), 5)) + ',' + str(f1))
            config_file.write('\n')

    config['total_loss'] = total_loss
    config['val_f1'] = best_val_f1
    save(net, config)
