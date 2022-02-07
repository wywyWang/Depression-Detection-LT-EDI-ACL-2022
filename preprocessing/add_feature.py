from xmlrpc.client import Boolean
import argparse
import re
import pandas as pd
import emoji
from tqdm import tqdm
from nltk.sentiment import vader
from textblob import TextBlob
from flair.models import TextClassifier
from flair.data import Sentence

# import nltk
# nltk.download('vader_lexicon')

def read_input():
    '''
    Read tsv data
    NOTE: should change the column name in dev_with_labels.tsv to 'Text_data'    
    '''
    df_train = pd.read_csv('./train.tsv', sep='\t')
    df_val = pd.read_csv('./dev_with_labels.tsv', sep='\t')

    return df_train, df_val

def preprocessing(df_train, df_val):
    ## Transform emoji
    df_train['Text_data'] = df_train['Text_data'].astype(str).apply(lambda x: emoji.demojize(x, delimiters=(' ', ' ')))
    df_val['Text data'] = df_val['Text data'].astype(str).apply(lambda x: emoji.demojize(x, delimiters=(' ', ' ')))
    
    ## Remove URL
    df_train['Text_data'] = df_train['Text_data'].astype(str).apply(lambda x: re.sub(r'http\S+', '', x))
    df_val['Text data'] = df_val['Text data'].astype(str).apply(lambda x: re.sub(r'http\S+', '', x))
    return df_train, df_val

def add_vader(data):
    '''
    Natural Language Toolkit: vader
    VADER (AAAI'14)
    '''
    df = pd.DataFrame(columns=['neg', 'neu', 'pos', 'compound'])
    vad = vader.SentimentIntensityAnalyzer()

    for i in tqdm(range(data.shape[0])):
        result = vad.polarity_scores(data.iloc[i, 1])
        result_list = []
        result_list.append(result.get('neg'))
        result_list.append(result.get('neu'))
        result_list.append(result.get('pos'))
        result_list.append(result.get('compound'))
        a_series = pd.Series(result_list, index = df.columns)
        df = df.append(a_series, ignore_index=True)

    return data.join(df)

def add_polarity(x):
    blob = TextBlob(x)
    polarity = blob.sentences[0].sentiment.polarity
    return polarity

def add_subjectivity(x):
    blob = TextBlob(x)
    subjectivity = blob.sentences[0].sentiment.subjectivity
    return subjectivity

def flair_prediction(x):
    sentence = Sentence(x)
    sia.predict(sentence)
    score = sentence.labels[0]
    return score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessing', type = Boolean, default = False)
    
    args = parser.parse_args()
    isPreprocessing = args.preprocessing

    df_train, df_val = read_input()
    if isPreprocessing:
        df_train, df_val = preprocessing(df_train, df_val)


    ## train
    df_train = add_vader(df_train)

    df_train["polarity"] = df_train['Text_data'].apply(add_polarity)
    df_train["subjectivity"] = df_train['Text_data'].apply(add_subjectivity)

    sia = TextClassifier.load('en-sentiment')
    df_train["flair_sentiment"] = df_train['Text_data'].apply(flair_prediction)

    df_train.to_csv('train_np.tsv', sep='\t')


    ## val
    df_val = add_vader(df_val)

    df_val["polarity"] = df_val['Text data'].apply(add_polarity)
    df_val["subjectivity"] = df_val['Text data'].apply(add_subjectivity)

    sia = TextClassifier.load('en-sentiment')
    df_val["flair_sentiment"] = df_val['Text data'].apply(flair_prediction)

    df_val.to_csv('val_np.tsv', sep='\t')