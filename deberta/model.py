import torch.nn as nn
import torch
from layers import MultiHeadAttention, PositionwiseFeedForward


class DeBERTaBaseline(nn.Module):
    def __init__(self):
        super().__init__()

        dim = 256
        dropout = 0.1
        head = 4

        self.text_embedding = nn.Sequential(
            nn.Linear(768, dim),
            nn.ReLU()
        )

        self.gru = nn.GRU(input_size=dim, hidden_size=dim, num_layers=1, dropout=dropout, batch_first=False, bidirectional=False)

        self.text_attention = MultiHeadAttention(head, dim, dim, dim, dropout=dropout)
        self.text_pos_ffn = PositionwiseFeedForward(dim, dim*2, dropout=dropout)

        self.classifier = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, pretrained_text):
        text_embedding = self.text_embedding(pretrained_text)

        gru_output, hidden = self.gru(text_embedding)
        attention_output, _ = self.text_attention(gru_output, gru_output, gru_output)
        attention_output = self.text_pos_ffn(attention_output)

        # aggregate word embeddings to a sentence embedding
        attention_output = torch.mean(attention_output, dim=1)
        predicted_output = self.classifier(attention_output)
        return predicted_output