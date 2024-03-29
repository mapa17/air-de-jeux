from locale import T_FMT_AMPM
from pathlib import Path
from re import S
from typing import Tuple, Dict, Callable
import copy
import time
import math

from numpy import ndarray
import pandas as pd
from pandas import DataFrame, Series
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

    ntokens = len(train_data.names_dataset.vocab)
    
    # Remove one from the padded sequence length because we have the shift in the training/target data. We want to predict the next character.
    sequence_length = train_data.get_padded_sequence_length()-1  

    # Create a additive mask that is used to exclude future sequence elements from
    # the self attention mechanism
    src_mask = generate_square_subsequent_mask(sequence_length).to(device)
    num_batches = len(train_data)

    for i, (data, targets) in enumerate(train_data):

        # Transpose the data in order to get it into the shape [sequence_length, batch_size]
        data = data.T
        targets = targets.T

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

    # Remove one from the padded sequence length because we have the shift in the training/target data. We want to predict the next character.
    sequence_length = eval_data.get_padded_sequence_length()-1  

    ntokens = len(eval_data.names_dataset.vocab)
    num_batches = len(eval_data)
    src_mask = generate_square_subsequent_mask(sequence_length).to(device)

    with torch.no_grad():
        for (data, targets) in eval_data:

            # Transpose the data in order to get it into the shape [sequence_length, batch_size]
            data = data.T
            targets = targets.T

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
def create_vocab(data : Path, storage : Path):
    data_loader = Names(data, 8, torch.device('cpu'))
    vocab = data_loader.names_dataset.vocab

    print(f"Storing vocab in {storage}")
    torch.save(vocab, storage)

@app.command()
def train_model(data : Path, vocab_storage : Path, epochs : int, model_storage : Path) :

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab = torch.load(vocab_storage)
    trn_data = Names(data / "training.csv", 8, device, vocab=vocab)
    # Make sure to use the same vocab for validation as for training
    val_data = Names(data / "validation.csv", 8, device, vocab=vocab)

    # Transformer configuration
    ntokens = len(vocab)  # size of vocabulary
    emsize = 20  # embedding dimension (output of the encoder)
    d_hid = 40  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 4  # number of heads in nn.MultiheadAttention
    dropout = 0.2  # dropout probability
    model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

    loss_fun = nn.CrossEntropyLoss()
    lr = 0.5  # learning rate
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

    print(f"Storing model to {model_storage} ...")
    save_model(model_storage, best_model, best_optimizer, best_epoch, best_val_loss, str(device))


@app.command()
def predict(
    model_storage : Path = typer.Argument(..., exists=True, readable=True),
    vocab_storage : Path = typer.Argument(..., exists=True, readable=True),
    num_names : int = typer.Argument(..., min=1),
    names_storage : Path = typer.Argument(..., writable=True),
    max_iterations : int = 100) :
    """Predict names using a previously trained model and vocab.

    Args:
        model_storage (Path, optional): _description_. Defaults to typer.Argument(..., exists=True, readable=True).
        vocab_storage (Path, optional): _description_. Defaults to typer.Argument(..., readable=True).
        num_names (int, optional): _description_. Defaults to typer.Argument(..., min=1).
        names_storage (Path, optional): _description_. Defaults to typer.Argument(..., writable=True).
        max_iterations (int, optional): _description_. Defaults to 100.
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lm = load_model(model_storage, TransformerModel, torch.optim.SGD, str(device))
    vocab = torch.load(vocab_storage)
    model = lm['model'].eval()

    batch_size=32
    start_tk = vocab[NamesDataset.start_token]
    stop_tk = vocab[NamesDataset.stop_token]

    src_mask = generate_square_subsequent_mask(max_iterations).to(device)
    names = []
    num_batches = num_names // batch_size
    last_batch_num_names = num_names - (num_batches*batch_size)
    batch_names =[]
    for batch_idx in range(num_batches+1):
        names.extend(batch_names)
        print(f"{batch_idx+1} Generating batch of names ...")
        seqs = torch.tensor([start_tk, ], dtype=torch.long).repeat(1, batch_size)
        for input_seq_length in range(1, max_iterations+1):
            mask = src_mask[:input_seq_length, :input_seq_length]
            logits = model(seqs, mask)

            # Use the model output (token probability) to sample concrete tokens.
            token_dist = torch.nn.functional.softmax(logits[-1, :])
            next_token = torch.multinomial(token_dist, 1).T

            # Accumulate tokens
            seqs = torch.cat([seqs, next_token], dim=0)
            
            # Test if there is a stop token in each batch element
            if min((seqs == stop_tk).sum(dim=0)) == 1:
                break

        # Position of the first stop token for each name
        seq_length = torch.argmax((seqs == stop_tk).int(), dim=0)
        token_idx = seqs.T.tolist()
        batch_names = [''.join(vocab.lookup_tokens(tokens[1:sl])) for tokens, sl in zip(token_idx, seq_length)]

    names.extend(batch_names[:last_batch_num_names])
    names.insert(0, 'name') # Add column name to have identical structure as the training data

    print(f"Writing names to {names_storage} ...")
    with open(names_storage, 'w') as f:
        f.writelines('\n'.join(names))


def name_comparison(tgt_names : Series, syn_names : Series) -> Dict[str, float] :
    unique_names = syn_names.nunique() / syn_names.shape[0]
    
    # Name overlap
    identical_names = sum([1 if n in list(syn_names) else 0 for n in list(tgt_names)]) / syn_names.shape[0]

    #
    split_syn_names = syn_names.str.split(' ', n=1, expand=True)
    split_tgt_names = tgt_names.str.split(' ', n=1, expand=True)
    
    sf = split_syn_names[0]
    sl = split_syn_names[1]
    tf = split_tgt_names[0]
    tl = split_tgt_names[1]

    identical_first_names = sum([1 if n in list(sf) else 0 for n in list(tf)]) / sf.shape[0]
    identical_last_names = sum([1 if n in list(sl) else 0 for n in list(tl)]) / sl.shape[0]

    avg_sf = sf.str.len().mean()
    avg_tf = tf.str.len().mean()
    avg_sl = sl.str.len().mean()
    avg_tl = tl.str.len().mean()

    metrics = {
        'unique_names': unique_names,
        'identical_names': identical_names,
        'identical_f': identical_first_names,
        'identical_l': identical_last_names,
        'avg_f_diff': avg_sf-avg_tf,
        'avg_l_diff': avg_sl-avg_tl,
    }
    return metrics

@app.command()
def compare(
    tgt_names_storage : Path = typer.Argument(..., exists=True, readable=True),
    syn_names_storage : Path = typer.Argument(..., exists=True, readable=True)
    ) :
    tgt_names = pd.read_csv(tgt_names_storage)
    syn_names = pd.read_csv(syn_names_storage)

    # Calculate metrics
    metrics = name_comparison(tgt_names['name'], syn_names['name'])
    print(metrics)
 
if __name__ == "__main__":
    app()