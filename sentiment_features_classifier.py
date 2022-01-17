import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import wandb
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import lightgbm as lgb
from lightgbm import LGBMClassifier

def read_input():
    '''
    Read tsv data
    NOTE: should change the column name in dev_with_labels.tsv to 'Text_data'    
    '''
    df_train = pd.read_csv('./data/train_np.tsv', sep='\t')
    df_val = pd.read_csv('./data/val_np.tsv', sep='\t')

    return df_train, df_val

def change_format(df_train, df_val):
    ## label
    category = {
        'moderate': 0,
        'severe': 1,
        'not depression': 2
    }
    df_train['Label'] = df_train['Label'].map(category)
    df_val['Label'] = df_val['Label'].map(category)

    ## drop some columns
    df_train = df_train.drop(columns=['Unnamed: 0', 'PID', 'Text_data', 'flair_sentiment'])
    df_val = df_val.drop(columns=['Unnamed: 0', 'PID', 'Text data', 'flair_sentiment'])

    return df_train, df_val

def f1_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    err = 1-f1_score(y_true, np.round(y_pred))
    return 'f1_err', err

def xgboost(df_train, df_val):
    xgboostModel = XGBClassifier(n_estimators=100, gamma = 0.1, max_depth = 4, subsample = 1, learning_rate = 0.1)
    
    param_test = {
        # 'n_estimators': range(1, 100, 11),
        'gamma' : np.linspace(0, 0.1, 11),
        'max_depth' : [2, 3, 4, 5, 6],
        'subsample' : np.linspace(0.9, 1, 11)
    }
    gsearch = GridSearchCV(estimator = xgboostModel, 
                            param_grid = param_test, 
                            scoring='f1_macro', cv=5)
    
    gsearch.fit(df_train.iloc[:,1:], df_train['Label'], eval_metric=f1_eval)
    preds = gsearch.predict(df_val.iloc[:,1:])
    print('xgboost validation F1:', f1_score(df_val['Label'], preds, average='macro'))

def lgbm(df_train, df_val, printFeatureImportance):
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

    lgbmModel.fit(df_train.iloc[:,1:], df_train['Label'])
    preds = lgbmModel.predict(df_val.iloc[:,1:])

    # print(gsearch.best_params_)

    print('lgbm validation F1:', f1_score(df_val['Label'], preds, average='macro'))
    analysis(preds, df_val['Label'])
    print(confusion_matrix(df_val['Label'], preds))

    if printFeatureImportance:
        predictors = df_train.iloc[:,1:].columns
        feat_imp = pd.Series(lgbmModel.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        plt.savefig('lgbm_feature_importance.jpg')

def analysis(pred, groundtruth):
    print(classification_report(groundtruth, pred))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--printFeatureImportance', default = True)
    
    args = parser.parse_args()
    printFeatureImportance = args.printFeatureImportance
    
    df_train, df_val = read_input()
    df_train, df_val = change_format(df_train, df_val)

    # xgboost(df_train, df_val)
    lgbm(df_train, df_val, printFeatureImportance)
