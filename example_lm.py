import torch
import torch.nn as nn
import torch.optim as optim
from model_lm import LanguageModel
import train_lm
from tools import load_snli
from torchtext.vocab import GloVe
from argparse import ArgumentParser

parser = ArgumentParser(description='LING 380 Final Project')
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--model', type=str, default='SRN')
parser.add_argument('--glove_set', type=str, default='6B')
parser.add_argument('--device', type=str, default='cuda')

args = parser.parse_args()

#GloVe vector set to use:
glove_obj = GloVe(args.glove_set, dim=50)

if not (torch.cuda.is_available() or args.device == 'cpu'):
    print("Either command line arg must be cpu or cuda must be available. Setting device to cpu")
    args.device = 'cpu'

# load data
train_iter, val_iter, test_iter, text_field, label_field = load_snli(batch_size=32, device=args.device, trees=True, glove_obj=glove_obj)

# define model params, loss function, and optimizer
embedding_dim = 50
hidden_dim = 50

pad_idx = text_field.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
model = LanguageModel(text_field, embedding_dim, hidden_dim, args.model, glove_obj=glove_obj, stored_model=None)#args.model + '_checkpoint.pt') # can use GRU or LSTM instead of SRN
# if GPU is available, change to make model run on GPU and make all tensors run there by default
if torch.cuda.is_available() and not args.device == 'cpu':
    model.cuda(args.device)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

optimizer = optim.Adam(model.parameters())


# train model
n_epochs = args.epochs
train_lm.train(model, train_iter, val_iter, test_iter, optimizer, criterion, args.model + '_checkpoint.pt', device=args.device, short_train=True, n_epochs=n_epochs, patience=3)