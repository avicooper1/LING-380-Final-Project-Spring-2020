from typing import List, Tuple

import numpy as np
import torch
import torchtext.data as tt
from torchtext import datasets
from torchtext.vocab import GloVe

import os
import urllib.request
import json

def load_wikitext_103(batch_size: int) -> \
        Tuple[tt.Iterator, tt.Iterator, tt.Iterator, tt.Field]:
    """
    Loads the WikiText103 data from torchtext

    :param batch_size: The size of the mini-batches
    :return: Iterators for the three datasets, text field
    """
    # Prepare fields
    text_field = tt.Field(lower=True)
    
    # Load data
    train_data, valid_data, test_data = datasets.WikiText103.splits(text_field)

    # build vocab
    text_field.build_vocab(train_data, valid_data, test_data, vectors=GloVe(name='6B', dim=300))

    # print vocab information
    print(f"Size of vocabulary: {len(text_field.vocab)}")
    
    # iterator    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    iters = tt.BucketIterator.splits((train_data, valid_data, test_data),
                                     batch_size=batch_size, device=device)

    return iters + (text_field,)

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
