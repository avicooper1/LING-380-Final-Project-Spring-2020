import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext.data as tt

class LanguageModel(nn.Module):
    def __init__(self, text_field: tt.Field, embedding_dim: int, hidden_dim: int, rnn_type: str):
        super(LanguageModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.text_field = text_field
        
        self.embedding = nn.Embedding(len(text_field.vocab), embedding_dim)
        
        if(rnn_type == "SRN"):
            self.rnn = nn.RNN(embedding_dim, hidden_dim)
        elif(rnn_type == "GRU"):
            self.rnn = nn.GRU(embedding_dim, hidden_dim)
        elif(rnn_type == "LSTM"):
            self.rnn = nn.LSTM(embedding_dim, hidden_dim)
        else:
            raise ValueError("Please choose either SRN, GRU, or LSTM as RNN type")
        
        self.out = nn.Linear(hidden_dim, len(text_field.vocab))
        
    def forward(self, input):
        batch_size = input.shape[0]
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded)
        
        output = self.out(output)
        
        return F.softmax(output, dim = 1), hidden