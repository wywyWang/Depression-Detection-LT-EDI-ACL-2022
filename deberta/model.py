import torch.nn as nn
import torch
from layers import MultiHeadAttention, PositionwiseFeedForward


class DeBERTaBaseline(nn.Module):
    def __init__(self, config):
        super().__init__()

        dim = config['hidden']
        dropout = config['dropout']
        head = config['head']
        n_layers = config['n_layers']

        self.text_embedding = nn.Sequential(
            nn.Linear(768, dim),
            nn.ReLU()
        )

        # self.cnn = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1)
        self.gru = nn.GRU(input_size=dim, hidden_size=dim, num_layers=n_layers, dropout=dropout, batch_first=False, bidirectional=True)

        self.text_attention = MultiHeadAttention(head, dim*2, dim*2, dim*2, dropout=dropout)
        self.text_pos_ffn = PositionwiseFeedForward(dim*2, dim*2*2, dropout=dropout)

        self.classifier = nn.Sequential(
            nn.Linear(dim*2, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, pretrained_text):
        text_embedding = self.text_embedding(pretrained_text)

        # cnn_output = self.cnn(text_embedding.transpose(1, 2)).transpose(1, 2)
        gru_output, hidden = self.gru(text_embedding)
        attention_output, _ = self.text_attention(gru_output, gru_output, gru_output)
        attention_output = self.text_pos_ffn(attention_output)

        # aggregate word embeddings to a sentence embedding
        attention_output = torch.mean(attention_output, dim=1)

        predicted_output = self.classifier(attention_output)
        return predicted_output, attention_output
