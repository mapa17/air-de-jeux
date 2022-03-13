from pathlib import Path
from typing import Tuple
import copy
import time
import math

from numpy import ndarray
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm


import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from torchtext.vocab import vocab, build_vocab_from_iterator

from Transformer import TransformerModel, generate_square_subsequent_mask
from Names import Names


def train(model: nn.Module, loss_fun : _Loss, optimizer : Optimizer, train_data : Names) -> float:
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


def evaluate(model: nn.Module, loss_fun : _Loss, eval_data: Names) -> float:
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
            total_loss += batch_size * loss_fun(output_flat, targets.reshape(-1)).item()
    return total_loss / num_batches 


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trn_data = Names("2022-03-Transformer/data/training.csv", 8, device)
    # Make sure to use the same vocab for validation as for training
    val_data = Names('2022-03-Transformer/data/validation.csv', 8, device, vocab=trn_data.names_dataset.vocab)

    # Transformer configuration
    ntokens = len(trn_data.names_dataset.vocab)  # size of vocabulary
    emsize = 20  # embedding dimension
    d_hid = 20  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2  # number of heads in nn.MultiheadAttention
    dropout = 0.2  # dropout probability
    model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

    loss_fun = nn.CrossEntropyLoss()
    lr = 1.0  # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    epochs = 50 
    best_model = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(model, loss_fun, optimizer, trn_data)
        val_loss = evaluate(model, loss_fun, val_data)
        val_ppl = math.exp(val_loss)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
