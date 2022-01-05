import torch.nn as nn
import torch


class DeBERTaBaseline(nn.Module):
    def __init__(self):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, text_embedding):
        # aggregate word embeddings to a sentence embedding
        text_embedding = torch.mean(text_embedding, dim=1)
        predicted_output = self.classifier(text_embedding)
        return predicted_output