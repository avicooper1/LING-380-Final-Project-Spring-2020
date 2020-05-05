from typing import List, Tuple

import numpy as np
import torch
import torchtext.data as tt
from torchtext import datasets
from torchtext.vocab import GloVe
import torch.nn as nn

import os
import urllib.request
import json

def load_snli(batch_size: int, trees = False) -> \
        Tuple[tt.Iterator, tt.Iterator, tt.Iterator, tt.Field, tt.Field]:
    """
    Loads the SNLI data from torchtext

    :param batch_size: The size of the mini-batches
    :param trees: Whether or not data should be created with parse trees (false by default)
    :return: Train, validation, and test set iterators, text and label fields
    """

    # prepare fields
    TEXT = datasets.nli.ParsedTextField()
    LABEL = tt.LabelField()
    
    # load data dependent on whether data should include parse trees    
    if trees:
        TREE = datasets.nli.ShiftReduceField()
        train, val, test = datasets.SNLI.splits(TEXT, LABEL, TREE)
        
    else:
        train, val, test = datasets.SNLI.splits(TEXT, LABEL)
    
    # build vocab
    TEXT.build_vocab(train, vectors=GloVe('6B'))
    LABEL.build_vocab(train, vectors=GloVe('6B'))

    # print vocab information
    print(f"Size of vocabulary: {len(TEXT.vocab)}")
    
    # iterator    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    iters = tt.BucketIterator.splits((train, val, test),
                                     batch_size=batch_size, device=device)

    return iters + (TEXT, LABEL)

def get_blimp_data(fname: str) -> \
        Tuple[List, List, List]:
        
    """
    Downloads and prepares necessary BLiMP data
    
    :param fname: file name - must be the exact file name of any .jsonl file found here:
    https://github.com/alexwarstadt/blimp/tree/master/data
    :return: list of dicts containing all attributes of the .json file, list of strings containing the bad and good sentences
    """
    
    git_dir = "https://raw.githubusercontent.com/alexwarstadt/blimp/master/data/"
    
    if not os.path.exists('blimp_data'):
        os.makedirs('blimp_data')
        
    if not os.path.exists('blimp_data/' + fname + '.jsonl'):
        urllib.request.urlretrieve(git_dir + fname + ".jsonl", "blimp_data/" + fname + ".jsonl")
        
    with open('blimp_data/' + fname + ".jsonl", "r") as json_file:
        json_list = list(json_file)
    
    result = []
    for json_str in json_list:
        result.append(json.loads(json_str))
        
    sentence_bad, sentence_good = [], []
    prefix_bad, prefix_good = [], []
    for i in result:
        sentence_bad.append(i['sentence_bad']), sentence_good.append(i['sentence_good'])
        prefix_bad.append(i['one_prefix_word_bad']), prefix_good.append(i['one_prefix_word_good'])
        
    return result, sentence_bad, sentence_good, prefix_bad, prefix_good

def blimp_to_tensor(sentence_list: List, prefix_list: List, model: nn.Module) -> torch.Tensor:
    """
    Converts list of sentences and prefixes to a tensor containing all the words in the sentence leading up to the prefix
    The tensor will be of shape (max_sentence_length, 1000), meaning column i contains the context for the ith prefix
    Also returns a tensor of the tokenized prefixes of shape (1000)

    :param sentence_list: List of sentences of BLiMP data (as returned by get_blimp_data)
    :param prefix_list: List of prefixes of BLiMP data
    :param model: Model being used (parameter is here to use the model's text field for tokenization)
    :return: Context and prefix tensors
    """
    
    prefix_context = []
    for sentence, prefix in zip(sentence_list, prefix_list):
        prefix_context.append(sentence.split()[:sentence.replace(".", "").split().index(prefix)])
        
    index_list = []
    max_len = 0    
    for sentence in prefix_context:
        if(len(sentence) > max_len):
            max_len = len(sentence)
        
        curr_sentence = []  
        for word in sentence:
            curr_sentence.append(model.text_field.vocab.stoi[word])
        index_list.append(curr_sentence)
        
    pad_index = model.text_field.vocab.stoi['<pad>']
    for sentence_index in index_list:
        while(len(sentence_index) < max_len):
            sentence_index.insert(0, pad_index)

    context_index_tensor = torch.FloatTensor(index_list).T
    prefix_index_tensor = torch.FloatTensor([model.text_field.vocab.stoi[word] for word in prefix_list])
    
    return context_index_tensor.long(), prefix_index_tensor.long()

def blimp_accuracy(model: nn.Module, context: torch.Tensor, good_prefix: torch.Tensor, bad_prefix: torch.Tensor) -> \
        Tuple[int, int, int, int]:
        
    """
    Evalutes model accuracy on a given BLiMP dataset
    
    :param model: The model used to predict the next word
    :param context: Tensor of sentence context to the desired prefix, as returned from blimp_to_tensor
    :param good_prefix, bad_prefix: Tensors of indices of the good (correct) and bad (incorrect) prefixes, also from blimp_to_tensor
    :return correct: Number of times where the model probability for good prefix > probability for bad prefix
    :return gp_count: Number of times where the most probable word was the good prefix
    :return bp_count: Number of times where the most probable word was the bad prefix
    :return total: Total guesses
    """
    output = model(context)[0].permute(1, 0, 2) # permute to be (n_batches, max_sentence_len, n_words)

    correct = 0
    gp_count = 0
    bp_count = 0
    total = 0
    for i, gp_index, bp_index in zip(range(output.shape[0]), good_prefix, bad_prefix):
        if(output[i, -1, gp_index] > output[i, -1, bp_index]):
            correct += 1
        total += 1

        _, max_index = torch.max(output[i, -1, :], dim = 0)
        if(max_index == gp_index):
            gp_count += 1
        elif(max_index == bp_index):
            bp_count += 1

    return correct, gp_count, bp_count, total
