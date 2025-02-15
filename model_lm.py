import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext.data as tt
from spinn import SPINN

from torch.autograd import Variable
class Bottle(nn.Module):

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0] * size[1], -1))
        return out.view(*size, -1)


class Linear(Bottle, nn.Linear):
    pass


class LanguageModel(nn.Module):
    def __init__(self, text_field: tt.Field, embedding_dim: int, hidden_dim: int, rnn_type: str, glove_obj=None, stored_model=None):
        super(LanguageModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.text_field = text_field
        self.rnn_type = rnn_type

        self.embedding = nn.Embedding.from_pretrained(glove_obj.vectors, freeze=True) if glove_obj is not None else nn.Embedding(len(text_field.vocab), embedding_dim)
        
        if(self.rnn_type == "SRN"):
            self.rnn = nn.RNN(embedding_dim, hidden_dim)
        elif(self.rnn_type == "GRU"):
            self.rnn = nn.GRU(embedding_dim, hidden_dim)
        elif(self.rnn_type == "LSTM"):
            self.rnn = nn.LSTM(embedding_dim, hidden_dim)
        elif(self.rnn_type == "SPINN"):
            self.linear_encoder = Linear(self.embedding_dim, self.embedding_dim * 2)
            self.rnn = SPINN(embedding_dim)
        else:
            raise ValueError("Please choose either SRN, GRU, LSTM or SPINN as RNN type")
        
        self.out = nn.Linear(hidden_dim, len(text_field.vocab))

        if stored_model is not None: self.load_state_dict(torch.load(stored_model))
        
    def forward(self, input):
        if hasattr(input, "premise_transitions"):
            transitions = input.premise_transitions
        else:
            transitions = None
        
        if hasattr(input, "premise"):
            embedded = self.embedding(input.premise[0])
        elif type(input) == tuple:
            embedded = self.embedding(input[0])
            transitions = input[1]
        else:
            embedded = self.embedding(input)
            
        if self.rnn_type != "SPINN":
            output, hidden = self.rnn(embedded)
        else:
            prem_embed = self.linear_encoder(embedded)
            output, hidden = self.rnn(prem_embed, transitions if transitions is not None else None)

        output = self.out(output)
        
        return torch.sigmoid(output), hidden
