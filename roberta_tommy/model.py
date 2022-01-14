import torch.nn as nn
import torch
from transformers import AutoModel

class Model(nn.Module):
    def __init__(self, pretrained_type, config):
        super().__init__()

        num_labels = 3
        self.pretrained_model = AutoModel.from_pretrained(pretrained_type, num_labels=num_labels)
        self.dense = nn.Linear(config['hidden'], config['hidden'])
        self.dropout = nn.Dropout(config['dropout'])
        self.classifier = nn.Linear(config['hidden'], num_labels)
        # self.classifier = nn.Sequential(
        #     nn.Linear(config['hidden'], 128),
        #     nn.GELU(),
        #     nn.Dropout(config['dropout']),
        #     nn.Linear(128, num_labels),
        # )

        self.gelu = nn.GELU()

        self.classifier.apply(self.init_weights)
        torch.nn.init.orthogonal_(self.dense.weight)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight)

    def forward(self, **pretrained_text):
        outputs = self.pretrained_model(**pretrained_text).last_hidden_state
        pooled_output = outputs[:, 0, :]
        pooled_output = self.gelu(self.dense(pooled_output))
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits, pooled_output