from typing import List, Tuple

import numpy as np
import torch
import torchtext.data as tt
from torchtext import datasets
from torchtext.vocab import GloVe

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
    TEXT.build_vocab(train)
    LABEL.build_vocab(train)

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
    for i in result:
        sentence_bad.append(i['sentence_bad']), sentence_good.append(i['sentence_good'])
        
    return result, sentence_bad, sentence_good
