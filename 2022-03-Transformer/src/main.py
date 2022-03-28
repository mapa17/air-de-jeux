from pathlib import Path
from typing import Tuple, Dict, Callable
import copy
import time
import math

from numpy import ndarray
import pandas as pd
from pandas import DataFrame
#from tqdm import tqdm


import typer

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from torchtext.vocab import vocab, build_vocab_from_iterator

from Transformer import TransformerModel, generate_square_subsequent_mask
from Names import Names, NamesDataset


app = typer.Typer()

def train(model: nn.Module, loss_fun : _Loss, optimizer : Optimizer, train_data : Names, device : torch.device) -> float:
    """Train given Transformer model applying causality masking.

    Args:
        model (nn.Module): _description_
        train_data (Names): _description_
        valid_data (Names): _description_
    """
    model.train()  # turn on train mode
    total_loss = 0.

    batch_size = train_data.batch_size
    ntokens = len(train_data.names_dataset.vocab)
    # Create a additive mask that is used to exclude future sequence elements from
    # the self attention mechanism
    src_mask = generate_square_subsequent_mask(batch_size).to(device)
    num_batches = len(train_data)

    for i, (data, targets) in enumerate(train_data):
        this_batch_size = data.size(0)
        if this_batch_size != batch_size:  # only on last batch
            src_mask = src_mask[:this_batch_size, :this_batch_size]
        output = model(data, src_mask)
        # Transform the output and targets, show they work with the normal CrossEntropyLoss
        # function that expects an input [BS, output] and targets in shaped [BS]
        # from shape [BS, SeqLength, Vocab] -> [BS*SeqLength, Vocab]
        # targets from [BS, SeqLength] -> [BS*SeqLength]
        loss = loss_fun(output.view(-1, ntokens), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        #print(f"Training minibatch {i+1}/{num_batches}, Training Loss: {loss.item()}")

        total_loss += loss.item()

    return total_loss / num_batches


def evaluate(model: nn.Module, loss_fun : _Loss, eval_data: Names, device : torch.device) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    batch_size = eval_data.batch_size 
    ntokens = len(eval_data.names_dataset.vocab)
    num_batches = len(eval_data)
    src_mask = generate_square_subsequent_mask(batch_size).to(device)

    with torch.no_grad():
        for (data, targets) in eval_data:
            this_batch_size = data.size(0)
            if batch_size != this_batch_size:
                src_mask = src_mask[:this_batch_size, :this_batch_size]
            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += loss_fun(output_flat, targets.reshape(-1)).item()
    return total_loss / num_batches 

def save_model(path : Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, loss: float, device : str) :
    torch.save({
            'model_state_dict': model.state_dict(),
            'model_kwargs' : model.kwargs,
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss,
            'device': device,
            }, path)


def load_model(path : Path, model_cls : Callable, optimizer_cls : Callable, device : str) -> Dict[str, None] :
    """Load a stored model checkpoint including the optimizer. Makes sure to load
    the model onto the given device correctly.

    Args:
        path (str): _description_
        model (nn.Module): _description_
        optimizer (torch.optim.Optimizer): _description_
        state_dict (Dict[str, None]): _description_
        device (str): _description_

    Raises:
        ValueError: _description_

    Returns:
        Dict[str, None]: _description_
    """

    checkpoint = torch.load(path)
    src_dev = checkpoint['device']
    dest_dev = device

    model_kwargs = checkpoint['model_kwargs']
    model = model_cls(**model_kwargs)

    if src_dev == 'cuda' and dest_dev == 'cpu':
        model.load_state_dict(checkpoint['model_state_dict'], map_location = torch.device('cpu'))
    elif src_dev == 'cuda' and dest_dev == 'cuda':
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(torch.device('cuda'))
    elif src_dev == 'cpu' and dest_dev == 'cuda':
        model.load_state_dict(checkpoint['model_state_dict'], map_location = torch.device('cuda:0'))
    elif src_dev == 'cpu' and dest_dev == 'cpu':
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise ValueError(f"Can not load model from {src_dev} to {dest_dev} ...")

    optimizer = optimizer_cls(model.parameters(), lr=0.0)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']


    loaded_model = {
        'model': model,
        'optimizer': optimizer,
        'epoch': epoch,
        'loss': loss
    }

    return loaded_model

@app.command()
def train_model(data : Path, epochs : int, storage : Path) :

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trn_data = Names(data / "training.csv", 4, device)
    # Make sure to use the same vocab for validation as for training
    val_data = Names(data / "validation.csv", 8, device, vocab=trn_data.names_dataset.vocab)

    # Transformer configuration
    ntokens = len(trn_data.names_dataset.vocab)  # size of vocabulary
    emsize = 80  # embedding dimension
    d_hid = 80  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 4  # number of heads in nn.MultiheadAttention
    dropout = 0.2  # dropout probability
    model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

    loss_fun = nn.CrossEntropyLoss()
    lr = 1.0  # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_epoch = 0
    best_optimizer = None
    best_model = None

    print('-' * 89)
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        trn_loss = train(model, loss_fun, optimizer, trn_data, device)
        val_loss = evaluate(model, loss_fun, val_data, device)
        #val_ppl = math.exp(val_loss) # Validation perplexity
        elapsed = time.time() - epoch_start_time
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            f'training loss {trn_loss:5.2f} | valid loss {val_loss:5.2f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            best_optimizer = copy.deepcopy(optimizer)
            best_epoch = epoch

    storage = storage / "latest.pt"
    print(f"Storing model to {storage} ...")
    save_model(storage, best_model, best_optimizer, best_epoch, best_val_loss, str(device))


@app.command()
def predict(storage : Path) :
    lm = load_model(storage, TransformerModel, torch.optim.SGD, 'cpu')
    model = lm['model'].eval()

    start_sequence = NamesDataset.start_token
    

 
if __name__ == "__main__":
    app()