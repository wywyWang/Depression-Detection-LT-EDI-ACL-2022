# NYCU-TWD in Depression-Detection-LT-EDI-ACL-2022

A shared task on [Detecting Signs of Depression from Social Media Text at LT-EDI 2022, ACL 2022 Workshop](https://sites.google.com/view/lt-edi-2022/home?authuser=0). We won the **second place** and the technical report will be published in the near future.

## Challenge Overview
Given social media postings in English, the system should classify the signs of depression into three labels namely “not depressed”, “moderately depressed”, and “severely depressed”.

## Usage
- Method 1: Machine Learning Models
  - Add sentiment features by VADER (preprocessing/)
    ```=bash
    python add_feature.py --preprocessing {boolean}
    ```  
  - Train model (ml/)
    ```=bash
    python sentiment_features_classifier.py --embedding {name} --model {name}
    ```  
- Method 2: Pre-trained Language Models
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
Performance will be measured in terms of macro averaged Precision, macro averaged Recall and macro averaged F-Score across all the classes.

## Leaderboard
<img width="624" alt="Leaderboard" src="https://user-images.githubusercontent.com/44032506/153540392-2ff8fd40-5500-4b55-9fb8-eba898babeed.png">
