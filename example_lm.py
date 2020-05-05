import torch
import torch.nn as nn
import torch.optim as optim
from model_lm import LanguageModel
import train_lm
from tools import load_snli

# load data
train_iter, val_iter, test_iter, text_field, label_field = load_snli(batch_size=32, trees=True)

# define model params, loss function, and optimizer
embedding_dim = 300
hidden_dim = 300

pad_idx = text_field.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
model = LanguageModel(text_field, embedding_dim, hidden_dim, "SPINN") # can use GRU or LSTM instead of SRN
optimizer = optim.Adam(model.parameters())

# if GPU is available, change to make model run on GPU and make all tensors run there by default
if torch.cuda.is_available():
    model.cuda()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# train model
n_epochs = 1
train_lm.train(model, train_iter, val_iter, test_iter, optimizer, criterion, short_train=True, n_epochs=n_epochs, patience=3)