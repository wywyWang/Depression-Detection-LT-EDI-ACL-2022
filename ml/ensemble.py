import argparse
import numpy as np
from numpy import mean
from numpy import std
import pandas as pd
from sklearn.metrics import f1_score

def read_file(filename):
    XGB = pd.read_csv('./result/xgboost' + filename)
    LGBM = pd.read_csv('./result/LGBM' + filename)
    return XGB, LGBM

def validation(xgb, lgbm):
    category = {
        'moderate': 0,
        'severe': 1,
        'not depression': 2
    }
    val = pd.read_csv('../data/val_np.tsv', sep='\t')
    ground_truth = val['Label'].map(category)

    ensemble_val = pd.read_csv('./ml-based_val.csv')

    def get_results():
        results_dict = dict()
        results_dict['lgbm'] = np.argmax(lgbm.to_numpy(), axis=1)
        results_dict['xgb'] = np.argmax(xgb.to_numpy(), axis=1)
        results_dict['soft_voting'] = np.argmax(ensemble_val.to_numpy(), axis=1)
        return results_dict

    def evaluate(preds):
        scores = f1_score(ground_truth, preds, average = 'macro')
        return scores

    results = get_results()
    for name, preds in results.items():
        print('preds: ', preds)
        scores = evaluate(preds)
        print('>%s %.3f' % (name, mean(scores)))


def ensemble(XGB, LGBM, filename, val):
    ans = pd.DataFrame(columns = ['moderate', 'severe', 'not depression'])
    ans['moderate'] = (XGB.iloc[:, 0] + LGBM.iloc[:, 0]) / 2.0
    ans['severe'] = (XGB.iloc[:, 1] + LGBM.iloc[:, 1]) / 2.0
    ans['not depression'] = (XGB.iloc[:, 2] + LGBM.iloc[:, 2]) / 2.0
    output_filename = 'ml-based' + filename
    print('output: ', output_filename)
    ans.to_csv(output_filename, index = False)

    if val == 'True':
        validation(XGB, LGBM)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default = '_val.csv')
    parser.add_argument('--val', default = 'True')

    args = parser.parse_args()
    filename = args.filename
    val = args.val

    XGB, LGBM = read_file(filename)
    ensemble(XGB, LGBM, filename, val)