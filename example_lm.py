import torch
import torch.nn as nn
import torch.optim as optim
from model_lm import LanguageModel
import train_lm
from tools import load_snli
from torchtext.vocab import GloVe
from argparse import ArgumentParser
import os

parser = ArgumentParser(description='LING 380 Final Project')
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--model', type=str, default='SPINN')
parser.add_argument('--glove_set', type=str, default='6B')

args = parser.parse_args()

#GloVe vector set to use:
glove_obj = GloVe(args.glove_set, dim=100)

# load data
train_iter, val_iter, test_iter, text_field, label_field = load_snli(batch_size=128, trees=True, glove_obj=glove_obj)

# define model params, loss function, and optimizer
embedding_dim = 100
hidden_dim = 100

pad_idx = text_field.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
model = LanguageModel(text_field, embedding_dim, hidden_dim, args.model, glove_obj=glove_obj) # can use GRU or LSTM instead of SRN
optimizer = optim.Adam(model.parameters())

# if GPU is available, change to make model run on GPU and make all tensors run there by default
if torch.cuda.is_available():
    model.cuda()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# train model
n_epochs = args.epochs
train_lm.train(model, train_iter, val_iter, test_iter, optimizer, criterion, args.model + '_checkpoint.pt', short_train=False, n_epochs=n_epochs, patience=3)