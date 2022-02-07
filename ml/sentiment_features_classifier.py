import numpy as np
import pandas as pd
import time
import argparse
import logging
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier

from nltk.tokenize import TweetTokenizer 
from sentence_transformers import SentenceTransformer, LoggingHandler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import CondensedNearestNeighbour

def read_input():
    '''
    Read tsv data
    NOTE: should change the column name in dev_with_labels.tsv to 'Text_data'    
    '''
    train = pd.read_csv('../data/train_np.tsv', sep='\t')
    val = pd.read_csv('../data/val_np.tsv', sep='\t')
    test = pd.read_csv('../data/test_np.tsv', sep='\t')

    return train, val, test

def change_format(train, val, test):
    ## label
    category = {
        'moderate': 0,
        'severe': 1,
        'not depression': 2
    }
    train['Label'] = train['Label'].map(category)
    val['Label'] = val['Label'].map(category)

    ## drop some columns
    train = train.drop(columns = ['Unnamed: 0', 'PID', 'flair_sentiment'])
    val = val.drop(columns = ['Unnamed: 0', 'PID', 'flair_sentiment'])
    test = test.drop(columns = ['Pid', 'flair_sentiment'])

    return train, val, test

def f1_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    err = 1-f1_score(y_true, np.round(y_pred))
    return 'f1_err', err

def tokenizer(text):
    tk = TweetTokenizer() 
    token = tk.tokenize(text)
    return token

def analysis(pred, groundtruth):
    print(classification_report(groundtruth, pred))

def sentence_transformer(train, val, test):
    '''
    all-MiniLM-L6-v2: 384 dim
    roberta-large-nli-stsb-mean-tokens: 1024 dim
    average_word_embeddings_glove.6B.300d: 300d
    all-mpnet-base-v2: 768d

    SentenceTransformer maps sentences & paragraphs 
    to a xxx dimensional dense vector space
    '''
    model = SentenceTransformer('all-mpnet-base-v2')

    def encode(text):
        sentence_embeddings = model.encode(text)
        return sentence_embeddings

    train['Text_data'] = train['Text_data'].apply(encode)
    val['Text data'] = val['Text data'].apply(encode)
    test['Text_data'] = test['Text_data'].apply(encode)

    train_embedding = pd.DataFrame(index = range(len(train['Text_data'])), columns = range(768))
    for i in range(len(train['Text_data'])):
        embeds = train.iloc[i,0]
        for j in range(len(embeds)):
            train_embedding.iloc[i, j] = float(embeds[j])

    val_embedding = pd.DataFrame(index = range(len(val['Text data'])), columns = range(768))
    for i in range(len(val['Text data'])):
        embeds = val.iloc[i,0]
        for j in range(len(embeds)):
            val_embedding.iloc[i, j] = float(embeds[j])

    test_embedding = pd.DataFrame(index = range(len(test['Text_data'])), columns = range(768))
    for i in range(len(test['Text_data'])):
        embeds = test.iloc[i,0]
        for j in range(len(embeds)):
            test_embedding.iloc[i, j] = float(embeds[j])

    train_embedding.astype('float').to_csv('train_mpnet.csv', index = False)
    val_embedding.astype('float').to_csv('val_mpnet.csv', index = False)
    test_embedding.astype('float').to_csv('test_mpnet.csv', index = False)

def smote(X, y):
    print('Before: ', Counter(y))
    n_samples = 1971 # 2964

    def sampling_strategy(X, y, n_samples, t = 'majority'):
        target_classes = ''
        if t == 'majority':
            target_classes = y.value_counts() > n_samples
        elif t == 'minority':
            target_classes = y.value_counts() < n_samples
        tc = target_classes[target_classes == True].index
        sampling_strategy = {}
        for target in tc:
            sampling_strategy[target] = n_samples
        return sampling_strategy

    # under_sampler = ClusterCentroids(sampling_strategy=sampling_strategy(X,y,n_samples,t='majority'))
    under_sampler = CondensedNearestNeighbour(sampling_strategy = 'majority')
    X_under, y_under = under_sampler.fit_resample(X, y)
    print('After under sampling: ', Counter(y_under))

    over_sampler = SMOTE(sampling_strategy=  sampling_strategy(X_under, y_under, n_samples, t = 'minority'), k_neighbors=2)
    X, y = over_sampler.fit_resample(X_under, y_under)
    print('After over sampling: ', Counter(y))

    return X, y

def embedding_and_sent(train, val, test, model):
    ## Sentence embedding
    sentence_transformer(train, val, test)
    train_add_embedding = pd.read_csv('train_mpnet.csv').join(train.iloc[:,2:])
    val_add_embedding = pd.read_csv('val_mpnet.csv').join(val.iloc[:,2:])
    test_add_embedding = pd.read_csv('test_mpnet.csv').join(test.iloc[:,1:])

    ## Data imbalance
    X_train, y_train = smote(train_add_embedding, train['Label'])

    ## Split training data
    X_real_train, X_train_val, y_real_train, y_train_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
    # X_real_train = X_train
    # y_real_train = y_train

    X_val = val_add_embedding
    y_val = val['Label']

    if model == 'LGBM':
        ## roberta-large-nli-stsb-mean-tokens
        # lgbmModel = lgb.LGBMClassifier(objective = 'multiclass', learning_rate = 0.4, 
        #                                 n_estimators = 81, max_depth = 3, random_state = 0)
        ## all-MiniLM-L6-v2, add sentiment, F1=0.528
        # lgbmModel = lgb.LGBMClassifier(objective = 'multiclass', learning_rate = 0.345, 
        #                                 n_estimators = 81, max_depth = 9, random_state = 0, num_leaves = 32)
        lgbmModel = lgb.LGBMClassifier(objective = 'multiclass', learning_rate = 0.5, 
                                    n_estimators = 70, max_depth = 9, random_state = 0, num_leaves = 64)
        # param_test = {
        #     'boosting_type' :['gbdt'],#, 'dart', 'goss', 'rf'],
        #     'n_estimators': range(1, 100, 10),
        #     'num_leaves' : [32, 64, 128, 256, 512],
        #     'learning_rate' : np.linspace(0.01, 0.6, 12), 
        #     'max_depth' : [9, 10, 11, 12]
        # }
        # gsearch = GridSearchCV(estimator = lgbmModel, 
        #                     param_grid = param_test, 
        #                     scoring='f1_macro', cv=5, verbose=10)

        lgbmModel.fit(X_real_train, y_real_train)
        # print('LightGBM best parameters: ', gsearch.best_params_)
        train_preds = lgbmModel.predict(X_real_train)
        print('LightGBM (add embedding) training F1:', f1_score(y_real_train, train_preds, average = 'macro'))
        analysis(train_preds, y_real_train)

        train_preds = lgbmModel.predict(X_train_val)
        print('LightGBM (add embedding) train_val F1:', f1_score(y_train_val, train_preds, average = 'macro'))
        analysis(train_preds, y_train_val)

        preds = lgbmModel.predict(X_val) #_proba(X_val)
        print('LightGBM (add embedding) validation F1:', f1_score(y_val, preds, average = 'macro'))
        analysis(preds, y_val)

        # ans = pd.DataFrame(preds, columns = ['0','1','2'])
        # ans.to_csv('./result/LGBM_val.csv', index = False)

        test_preds = lgbmModel.predict_proba(test_add_embedding)
        print('Testing data prediction: ', test_preds)
        # counter = Counter(test_preds)
        # print('Test preds: ', counter)

        ans = pd.DataFrame(test_preds, columns = ['0','1','2'])
        ans.to_csv('./result/LGBM.csv', index = False)

    elif model == 'XGBoost':
        xgboostModel = XGBClassifier(n_estimators = 100, gamma = 0.02, max_depth = 6, 
                                        subsample = 0.98, learning_rate = 0.1)

        # param_test = {
        #     # 'n_estimators': range(1, 100, 11),
        #     'gamma' : np.linspace(0, 0.1, 11),
        #     'max_depth' : [2, 3, 4, 5, 6],
        #     'subsample' : np.linspace(0.9, 1, 11)
        # }
        # gsearch = GridSearchCV(estimator = xgboostModel, 
        #                         param_grid = param_test, 
        #                         scoring='f1_macro', cv=5)

        xgboostModel.fit(X_train, y_train, eval_metric = f1_eval)
        # print('XGBoost best parameters: ', gsearch.best_params_)

        train_preds = xgboostModel.predict(X_train)
        print('XGBoost (add embedding) training F1:', f1_score(y_train, train_preds, average = 'macro'))
        analysis(train_preds, y_train)

        preds = xgboostModel.predict_proba(X_val)
        # print('XGBoost (add embedding) validation F1:', f1_score(y_val, preds, average = 'macro'))
        # analysis(preds, y_val)

        ans = pd.DataFrame(preds, columns = ['0','1','2'])
        ans.to_csv('./result/xgboost_val.csv', index = False)        

        test_preds = xgboostModel.predict_proba(test_add_embedding)
        print('Testing data prediction: ', test_preds)
        # counter = Counter(test_preds)
        # print('Test preds: ', counter)

        ans = pd.DataFrame(test_preds, columns = ['0','1','2'])
        ans.to_csv('./result/xgboost.csv', index = False)

## Only Sentiment features
def xgboost(train, val):
    xgboostModel = XGBClassifier(n_estimators=100, gamma = 0.1, max_depth = 4, subsample = 1, learning_rate = 0.1)
    
    param_test = {
        'n_estimators': range(1, 100, 11),
        'gamma' : np.linspace(0, 0.1, 11),
        'max_depth' : [2, 3, 4, 5, 6],
        'subsample' : np.linspace(0.9, 1, 11)
    }
    gsearch = GridSearchCV(estimator = xgboostModel, 
                            param_grid = param_test, 
                            scoring='f1_macro', cv=5)
    
    gsearch.fit(train.iloc[:,1:], train['Label'], eval_metric=f1_eval)
    print('XGBoost best parameters: ', gsearch.best_params_)
    preds = gsearch.predict(val.iloc[:,1:])
    print('XGBoost (only sentiment features) validation F1:', f1_score(val['Label'], preds, average='macro'))

## Only Sentiment features
def lgbm(train, val, printFeatureImportance):
    lgbmModel = lgb.LGBMClassifier(objective = 'multiclass', 
                                    learning_rate = 0.451, 
                                    n_estimators = 89,
                                    max_depth = 7, 
                                    random_state = 0)
    # param_test = {
    #     'n_estimators': range(1, 100, 11),
    #     'learning_rate' : np.linspace(0.01, 0.5, 11), 
    #     'max_depth' : [2, 3, 4, 5, 6, 7, 8, 9, 10]
    # }
    # gsearch = GridSearchCV(estimator = lgbmModel, 
    #                     param_grid = param_test, 
    #                     scoring='f1_macro', cv=5)

    lgbmModel.fit(train.iloc[:,1:], train['Label'])
    preds = lgbmModel.predict(val.iloc[:,1:])
    # print('LigntGBM best parameters: ', gsearch.best_params_)
    print('LightGBM (only sentiment features) validation F1:', f1_score(val['Label'], preds, average = 'macro'))
    analysis(preds, val['Label'])
    print(confusion_matrix(val['Label'], preds))

    if printFeatureImportance:
        predictors = train.iloc[:,1:].columns
        feat_imp = pd.Series(lgbmModel.feature_importances_, predictors).sort_values(ascending = False)
        feat_imp.plot(kind = 'bar', title = 'Feature Importances')
        plt.ylabel('LGBM Feature Importance Score')
        plt.savefig('Lgbm_feature_importance.jpg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--printFeatureImportance', default = True)
    parser.add_argument('--embedding', default = 'glove')
    parser.add_argument('--model', default = 'LGBM')

    args = parser.parse_args()
    printFeatureImportance = args.printFeatureImportance
    embedding_type = args.embedding
    model = args.model
    
    train, val, test = read_input()
    train, val, test = change_format(train, val, test)

    # xgboost(train, val)
    # lgbm(train, val, printFeatureImportance)
    embedding_and_sent(train, val, test, model)