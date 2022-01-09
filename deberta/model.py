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

        self.gru = nn.GRU(input_size=dim, hidden_size=dim, num_layers=n_layers, dropout=dropout, batch_first=False, bidirectional=True)

        self.text_attention = MultiHeadAttention(head, dim*2, dim*2, dim*2, dropout=dropout)
        self.text_pos_ffn = PositionwiseFeedForward(dim*2, dim*2*2, dropout=dropout)

        # for summarized text
        self.summarized_text_embedding = nn.Sequential(
            nn.Linear(768, dim),
            nn.ReLU()
        )

        self.summarized_gru = nn.GRU(input_size=dim, hidden_size=dim, num_layers=n_layers, dropout=dropout, batch_first=False, bidirectional=True)

        self.summarized_text_attention = MultiHeadAttention(head, dim*2, dim*2, dim*2, dropout=dropout)
        self.summarized_text_pos_ffn = PositionwiseFeedForward(dim*2, dim*2*2, dropout=dropout)

        self.classifier = nn.Sequential(
            nn.Linear(dim*2*2, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, pretrained_text, summarized_pretrained_text):
        text_embedding = self.text_embedding(pretrained_text)

        gru_output, hidden = self.gru(text_embedding)
        attention_output, _ = self.text_attention(gru_output, gru_output, gru_output)
        attention_output = self.text_pos_ffn(attention_output)

        # aggregate word embeddings to a sentence embedding
        attention_output = torch.mean(attention_output, dim=1)

        # summarized text
        summarized_text_embedding = self.summarized_text_embedding(summarized_pretrained_text)

        summarized_gru_output, hidden = self.summarized_gru(summarized_text_embedding)
        summarized_attention_output, _ = self.summarized_text_attention(summarized_gru_output, summarized_gru_output, summarized_gru_output)
        summarized_attention_output = self.summarized_text_pos_ffn(summarized_attention_output)

        # aggregate word embeddings to a sentence embedding
        summarized_attention_output = torch.mean(summarized_attention_output, dim=1)


        concat_output = torch.cat((attention_output, summarized_attention_output), dim=1)
        predicted_output = self.classifier(concat_output)
        return predicted_output
