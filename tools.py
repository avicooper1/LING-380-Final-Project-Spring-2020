from typing import List, Tuple

import numpy as np
import torch
import torchtext.data as tt
from torchtext import datasets
from torchtext.vocab import GloVe

def load_wikitext_103(batch_size: int) -> \
        Tuple[tt.Iterator, tt.Iterator, tt.Iterator, tt.Field]:
    """
    Loads the WikiText103 data from torchtext

    :param batch_size: The size of the mini-batches
    :return: Iterators for the three datasets, text field
    """
    # Prepare fields
    text_field = tt.Field(lower=True, batch_first=True)
    
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