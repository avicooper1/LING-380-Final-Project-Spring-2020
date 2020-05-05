import torchtext.data as tt
from torchtext.data import Field, BucketIterator
import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
import model_lm
from early_stopping import EarlyStopping
from tqdm import tqdm

def train_epoch(model: nn.Module, iterator: BucketIterator, optimizer: optim.Optimizer, criterion: nn.Module,
                clip: float, short_train: bool = False,):
    
    model.train()
    epoch_loss = 0.
    n_batches = len(iterator)
    for n, batch in enumerate(tqdm(iterator)):
        if short_train and n % 20 != 0:
            continue       
        batch_loss = torch.tensor(0., requires_grad=True)
        optimizer.zero_grad()    
        
        output, _ = model(batch)
        text = batch.premise[0]
        text = text.permute(1, 0) # permute text to be of size (n_batches, seq_len)
        output = output.permute(1, 0, 2) # permute output to be of size (n_batches, seq_len, n_words)
        for i in range(1, text.shape[1]):
            pred_words = output[:, i - 1]
            target = text[:, i]
            batch_loss = batch_loss + criterion(pred_words, target)
            
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += batch_loss.item()      
    return epoch_loss / len(iterator)


def evaluate(model: nn.Module, iterator: BucketIterator, criterion: nn.Module):
    
    model.eval()
    epoch_loss = 0.
    
    pad_index = model.text_field.vocab.stoi['<pad>']
    with torch.no_grad():
        for batch in iterator:
            output, _ = model(batch)
            text = batch.premise[0]
            text = text.permute(1, 0)
            output = output.permute(1, 0, 2)
            for i in range(1, text.shape[1]):
                pred_words = output[:, i - 1]
                target = text[:, i]
                epoch_loss = epoch_loss + criterion(pred_words, target).item()           
    return epoch_loss / len(iterator)


def epoch_time(start_time: int, end_time: int):
    
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, train_iterator, valid_iterator, test_iterator, optimizer, criterion,
          clip=1, short_train=True, n_epochs=50, patience=3):
    
    early_stopping = EarlyStopping(patience=patience, verbose=False, filename='checkpoint.pt')
    for epoch in range(n_epochs):
        start_time = time.time()
        train_loss = train_epoch(model, train_iterator, optimizer, criterion, clip, short_train)
        valid_loss = evaluate(model, valid_iterator, criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3E}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3E}')
        
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping, reloading checkpoint model")
            model.load_state_dict(torch.load('checkpoint.pt'))
            break

    test_loss = evaluate(model, test_iterator, criterion)
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3E} |')
