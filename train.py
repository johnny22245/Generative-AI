"""
TODO write your training loop here.
Things to take care with:
    - make sure to use the correct loss function for the task
    - make sure that the targets are correct (each token should predict the next token in the sequence)
    - there should be no loss for padding tokens.
"""

#initialize logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# load modules
import torch
from torch import nn
from model import LanguageModel
import math

# set random seed
import random
import numpy as np
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

"""
Hyper-parameters for training
"""

learning_rate = 3e-4
weight_decay = 0.1
epochs = 100

# assign GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu' # assign GPU if available

def train_model(model, dataloader, save_path='best_model.pt'):
    # train model
    print("Present device: ", device)
    model = model.to(device)
    
    # create pytorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay = weight_decay)
    
    # initialize loss -- ignore padding that we will set to -100 later
    loss_fn = nn.CrossEntropyLoss(reduction='sum', ignore_index=-100)
    
    # store data for early stopping
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_state = None
    
    train_losses = []
    val_losses = []
    
    # train model for certain epochs - training data
    for iter in range(epochs):
        total_train_loss = 0
        total_train_tokens = 0
        total_val_loss = 0
        total_val_tokens = 0
        # sample a batch of data
        train_data, val_data = dataloader['train'], dataloader['val']
        for batch in train_data:
            input_ids = batch['input_ids'].to(device)
            targets = input_ids[:, 1:]
            inputs = input_ids[:, :-1]

            padding_mask = batch['attention_mask'].to(device)
            target_padding_mask = padding_mask[:, 1:]
            input_padding_mask = padding_mask[:, :-1]
            
            # replace targets with -100 where it's padding
            targets = targets.masked_fill(target_padding_mask==0, -100).view(-1)
            
            # train model and zero the optimizer gradients
            model.train()
            optimizer.zero_grad()
            
            logits = model(inputs, input_padding_mask)
            B, S, V = logits.shape
            logits = logits.view(B*S, V)
            #print("Before train loss calc. :", logits.shape)
            loss = loss_fn(logits, targets)
            #print("Loss for batch: ", loss)
            
            # add up total loss and tokens - for perplexity calculation
            total_train_loss += loss_fn(logits, targets).item() 
            total_train_tokens += target_padding_mask.sum().item() 
            
            loss.backward()
            #apply gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
            optimizer.step()
            
        # evaluate model
        for batch in val_data:
            input_ids = batch['input_ids'].to(device)
            targets = input_ids[:, 1:]
            inputs = input_ids[:, :-1]

            padding_mask = batch['attention_mask'].to(device)
            target_padding_mask = padding_mask[:, 1:]
            input_padding_mask = padding_mask[:, :-1]
            
            # replace targets with -100 where it's padding
            targets = targets.masked_fill(target_padding_mask==0, -100).view(-1)
            
            #model evaluate
            model.eval()
            
            with torch.no_grad():
                val_outputs = model(inputs, input_padding_mask)
                B, S, V = val_outputs.shape
                logits = val_outputs.view(-1, V)
                #print("Before val loss calc. :", logits.shape, targets.shape)
                val_loss = loss_fn(logits, targets)
                
                # add up total loss and tokens - for perplexity calculation
                total_val_loss += loss_fn(logits, targets).item() 
                total_val_tokens += target_padding_mask.sum().item() 
            
        #print the epoch train loss and perplexity
        perplexity_train = math.exp(total_train_loss / total_train_tokens)
        perplexity_val = math.exp(total_val_loss / total_val_tokens)
        
        logger.info(f"Epoch {iter}: Train loss is {total_train_loss} and perplexity is {perplexity_train}.")
        logger.info(f"Epoch {iter}: Val loss is {total_val_loss} and perplexity is {perplexity_val}.")
        
        train_losses.append(total_train_loss)
        val_losses.append(total_val_loss)
        
        # Early stopping
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered at epoch {iter}")
            logger.info(f"Best val loss: {best_val_loss}")
            break
    
    # callback to the best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
            
    # save best model (so far)
    torch.save(model.state_dict(), save_path)
    
    #store train & val loss for plotting all graphs in 1 go
    import json
    train_val_loss = {"train_loss": train_losses, "val_loss": val_losses}
    with open("./train_val_loss.json", 'w')as f:
        json.dump(train_val_loss, f, indent = 4)
    
    return 0