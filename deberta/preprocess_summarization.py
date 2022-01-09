from transformers import pipeline
import pandas as pd
import logging
import argparse
import os
from tqdm import tqdm


transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)


if __name__ == '__main__':
    df_train = pd.read_csv('../data/train_preprocess.tsv', sep='\t')
    df_val = pd.read_csv('../data/val_preprocess.tsv', sep='\t')

    summarizer = pipeline("summarization", model='facebook/bart-base', tokenizer='facebook/bart-base')

    summarized_list = []
    pbar = tqdm(df_val.iterrows(), total=len(df_val), desc='Processing train data')
    for n, row in tqdm(df_train.iterrows(), total=len(df_train)):
        pbar.set_description("Current pid: {}".format(row['PID']), refresh=True)
        summarized_text = summarizer(row['Text_data'], min_length=5, max_length=64, do_sample=True, truncation=True)
        summarized_list.append(summarized_text[0]['summary_text'])

    df_train['summarized_text'] = summarized_list
    df_train.to_csv('../data/train_preprocess_summarized.csv', index=False)

    summarized_list = []
    pbar = tqdm(df_val.iterrows(), total=len(df_val), desc='Processing validation data')
    for n, row in pbar:
        pbar.set_description("Current pid: {}".format(row['PID']), refresh=True)
        summarized_text = summarizer(row['Text_data'], min_length=5, max_length=64, do_sample=True, truncation=True)
        summarized_list.append(summarized_text[0]['summary_text'])

    df_val['summarized_text'] = summarized_list
    df_val.to_csv('../data/val_preprocess_summarized.csv', index=False)