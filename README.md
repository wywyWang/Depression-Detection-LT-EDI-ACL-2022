# NYCU-TWD in Depression-Detection-LT-EDI-ACL-2022

A shared task on [Detecting Signs of Depression from Social Media Text at LT-EDI 2022, ACL 2022 Workshop](https://sites.google.com/view/lt-edi-2022/home?authuser=0). We won the **second place** and the paper will be published in the near future.

## Challenge Overview
Given social media postings in English, the system should classify the signs of depression into three labels namely “not depressed”, “moderately depressed”, and “severely depressed”.

## Usage
- Method 1: Gradient Boosting Models + VAD Score
  - Add sentiment features by VADER (preprocessing/)
    ```=bash
    python add_feature.py --preprocessing {boolean}
    ```  
  - Train model (ml/)
    ```=bash
    python sentiment_features_classifier.py --embedding {name} --model {name}
    ```  
- Method 2: Pre-trained Language Models
  - Train model
    ```=bash
    python3 main.py --model_type [roberta/electra/deberta]
    ```
  - Ensemble and evaluate (for dev and test)
    ```=bash
    python3 ensemble.py --path [file path] --mode [dev/test]
    ```
- Method 3: Pre-trained Language Models + VAD Score + Supervised Contrastive Learning (plm_scl/)
  - Train model
    ```=bash
    python main.py {pre-trained name}
    ```
  - Evaluate model
    ```=bash
    python evaluate.py
    ```
    You need to modify {MODEL} and {MODEL_NAME} to your pre-trained model and corresponding path in `evaluate.py`.
    
- Power Weighted Sum
    ```=bash
    python ensemble.py
    ```

## Dataset
The dataset comprises training, development and test set. The data files are in Tab Separated Values (tsv) format with three columns namely posting_id (pid), text data and label. 

|   | Tran | Dev | Test |
|:---:|:---:|:---:|:---:|
| Not depressed | 1,971 | 1,830 |  |
| Moderate | 6,019 | 2,306 |  |
| Severe | 901 | 360 |  |
| **Total** | 8,891 | 4,496 | 3,245 |

## Metric
Performance will be measured in terms of macro averaged Precision, macro averaged Recall and macro averaged F1-Score across all the classes.

## Implementation Details
We report the hyper-parameters of each method as follows.
- Method 1: Gradient Boosting Models + VAD Score
  - General
    - Pretrained Sentence Embedding: MPNet
  - LightGBM
    | LR | num\_leaves | n\_estimators | max\_depth |
    |:---:|:---:|:---:|:---:|
    | 0.5 | 64 | 70 | 9 |
  - XGBoost
    | LR | gamma | n\_estimators | max\_depth | subsample |
    |:---:|:---:|:---:|:---:|:---:|
    | 0.1 | 0.02 | 100 | 6 | 0.98 |
- Method 2: Pre-trained Language Models
  - General
    | LR | Epochs |
    |:---:|:---:|
    | 2e-5 | 20 |
  - RoBERTa
    | Seed | Warm Up | Batch Size |
    |:---:|:---:|:---:|
    | 13 | 4 | 3 |
  - DeBERTa
    | Seed | Warm Up | Batch Size |
    |:---:|:---:|:---:|
    | 49 | 8 | 6 |
  - ELECTRA
    | Seed | Warm Up | Batch Size |
    |:---:|:---:|:---:|
    | 17 | 5 | 2 |
- Method 3: Pre-trained Language Models + VAD Score + Supervised Contrastive Learning
  | Epochs | LR | Batch Size | Seed | Warmup Steps | Hidden Dimension | Dropout | Lambda_{ce} | Lambda_{scl} |
  |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
  | 20 | 4e-5 | 8 | 17 | 5 | 512 | 0.1 | 0.7 | 0.3 |
- Power Weighted Sum
  - ensemble_weight: [1, 0.67, 0.69]
  - power: 4

## Leaderboard
<img width="624" alt="Leaderboard" src="https://user-images.githubusercontent.com/44032506/153540392-2ff8fd40-5500-4b55-9fb8-eba898babeed.png">
