import torch
from model_lm import LanguageModel
from tools import *
from torchtext.vocab import GloVe
from argparse import ArgumentParser
import json

parser = ArgumentParser(description='LING 380 Final Project')
parser.add_argument('--model', type=str, default='SRN')
parser.add_argument('--glove_set', type=str, default='6B')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--load_model', action='store_true', dest='load_model')

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
model = LanguageModel(text_field, embedding_dim, hidden_dim, args.model, glove_obj=glove_obj, stored_model=args.model + '_checkpoint.pt') # can use GRU or LSTM instead of SRN
# if GPU is available, change to make model run on GPU and make all tensors run there by default
if torch.cuda.is_available() and not args.device == 'cpu':
    model.cuda(args.device)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
# load model from checkpoint
model.load_state_dict(torch.load('model_checkpoints/' + args.model + "_checkpoint.pt"))
model = model.to(args.device)

# extract data from BLiMP
result, sent_bad, sent_good, pre_bad, pre_good = get_blimp_data('principle_A_c_command')
context, gprefix = blimp_to_tensor(result, sent_good, model)
_, bprefix = blimp_to_tensor(result, sent_bad, model)

# if running on GPU, you may need to move the above tensors to the GPU by uncommenting this code:
# context, gprefix, bprefix = context.cuda(), gprefix.cuda(), bprefix.cuda()

# if you are testing an SRN, GRU, or LSTM, uncomment and run the next line. then, ignore the rest of this code
# correct, gp_count, bp_count, total = blimp_accuracy_context(model, gcontext, sent_good, sent_bad) 

# if you are testing SPINN, continue here
with open('blimp_transitions/good_sent_parses.json') as f:
  good_parses = json.load(f)

with open('blimp_transitions/bad_sent_parses.json') as f:
  bad_parses = json.load(f)

gparse_list = list(good_parses.values())
parses = parse_to_tensor(good_parse_list, model)

# as before, if running on GPU, you may need to uncomment this code
# parses = parses.cuda()

cont_parse_input = (context, parses)

blimp_accuracy_parse(model, cont_parse_input, good_prefix, bad_prefix)
import pdb; pdb.set_trace()
