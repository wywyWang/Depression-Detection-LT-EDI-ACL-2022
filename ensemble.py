import pandas as pd
import sys
from sklearn.metrics import f1_score

if sys.argv[1] == 'test':
    df_electra = pd.read_csv('prediction/google-electra-base-discriminator_0.5775test_answer.csv')
    df_roberta = pd.read_csv('prediction/twitter-roberta-base-sentiment_0.5479test_answer.csv')
    df_deberta = pd.read_csv('prediction/deberta-base_0.5521test_answer.csv')
    df_val = pd.read_csv('data/test_np.tsv', sep='\t')
else:
    df_electra = pd.read_csv('prediction/google-electra-base-discriminator_0.5775answer.csv')
    df_roberta = pd.read_csv('prediction/twitter-roberta-base-sentiment_0.5479answer.csv')
    df_deberta = pd.read_csv('prediction/deberta-base_0.5521answer.csv')
    df_val = pd.read_csv('data/valid_np.tsv', sep='\t')

assert df_electra.shape == df_roberta.shape and df_roberta.shape == df_deberta.shape

category = {
    'not depression': 0,
    'moderate': 1,
    'severe': 2
}

inverse_category = {
    1: 'moderate',
    2: 'severe',
    0: 'not depression'
}

# 0.6 * electra + 0.4 * roberta reaches: 0.5834
# 0.5 * electra + 0.35 * deberta + 0.15 * roberta: 0.5955
pid, label, y_pred = [], [], []
e_proportion = 0.5
d_proportion = 0.35
r_proportion = 0.15
for i in range(len(df_electra)):
    # moderate, severe, not depression
    electra_prob = df_electra.iloc[i].values.tolist()[1:]
    roberta_prob = df_roberta.iloc[i].values.tolist()[1:]
    deberta_prob = df_deberta.iloc[i].values.tolist()[1:]

    pid.append(df_val['PID'][i])

    prob = [(e_proportion*e_prob+d_proportion*d_prob+r_proportion*r_prob) for e_prob, d_prob, r_prob in zip(electra_prob, deberta_prob, roberta_prob)]
    class_index = prob.index(max(prob))

    label.append(prob)
    y_pred.append(inverse_category[class_index])

answer = pd.DataFrame(label, columns=category.keys())
answer['PID'] = df_val['PID'].values
answer = answer[['PID', 'moderate', 'severe', 'not depression']]
if sys.argv[1] == 'test':
    answer.to_csv('./prediction/test_ensemble.csv', index=False)
else:
    answer.to_csv('./prediction/dev_ensemble.csv', index=False)
    f1 = f1_score(df_val['label'], y_pred, average='macro')
    print(f1)

# # 0.6 * electra + 0.4 * roberta reaches: 0.5834
# # 0.5 * electra + 0.35 * deberta + 0.15 * roberta: 0.5955
# current_best = 0
# for p1 in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#     for p in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#         pid, label = [], []
#         e_proportion = p1/(p1+p+(1-p))
#         d_proportion = p/(p1+p+(1-p))
#         r_proportion = (1-p)/(p1+p+(1-p))
#         for i in range(len(df_electra)):
#             # moderate, severe, not depression
#             electra_prob = df_electra.iloc[i].values.tolist()[1:]
#             roberta_prob = df_roberta.iloc[i].values.tolist()[1:]
#             deberta_prob = df_deberta.iloc[i].values.tolist()[1:]

#             pid.append(df_val['PID'][i])

#             prob = [(e_proportion*e_prob+d_proportion*d_prob+r_proportion*r_prob) for e_prob, d_prob, r_prob in zip(electra_prob, deberta_prob, roberta_prob)]
#             class_index = prob.index(max(prob))

#             label.append(inverse_category[class_index])

#         f1 = f1_score(df_val['label'], label, average='macro')
#         if f1 > current_best:
#             current_best = f1
#             print("{} * electra + {} * deberta + {} * roberta".format(e_proportion, d_proportion, r_proportion))
#             print(f1_score(df_val['label'], label, average='macro'))