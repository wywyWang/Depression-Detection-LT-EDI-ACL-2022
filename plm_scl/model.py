import torch.nn as nn
import torch
from transformers import AutoModel


class Model(nn.Module):
    def __init__(self, pretrained_type, config):
        super().__init__()

        num_labels = 3
        self.pretrained_model = AutoModel.from_pretrained(pretrained_type, num_labels=num_labels)
        self.dense = nn.Linear(768, config['hidden'])
        self.dropout = nn.Dropout(config['dropout'])
        self.classifier = nn.Linear(config['hidden'], num_labels)

        vad_dim = 128
        self.vad_embedding = nn.Sequential(
            nn.Linear(4, vad_dim),
            nn.GELU()
        )

        self.pool_embedding = nn.Sequential(
            nn.Linear(config['hidden']+vad_dim, config['hidden']),
            nn.GELU()
        )

        self.gelu = nn.GELU()

        torch.nn.init.orthogonal_(self.dense.weight)
        torch.nn.init.orthogonal_(self.classifier.weight)

    def forward(self, vad_score, **pretrained_text):
        vad_embedding = self.vad_embedding(vad_score)
        outputs = self.pretrained_model(**pretrained_text).last_hidden_state
        pretrained_output = outputs[:, 0, :]
        pretrained_output = self.gelu(self.dense(pretrained_output))

        pooled_output = torch.cat((vad_embedding, pretrained_output), dim=1)
        pooled_output = self.pool_embedding(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits, pretrained_output, vad_embedding