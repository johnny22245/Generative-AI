from data import GPTTokenizedData
from model import LanguageModel
from util import count_trainable_parameters
from train import train_model
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    # get dataloaders (data.py)
    tokenized = GPTTokenizedData()
    dataloaders = tokenized.dataloaders # all 3 dataloaders in a dictionary with keys 'train', 'test', 'val
    vocab_size = tokenized.vocab_size
    
    # instantiate model (model.py)
    model = LanguageModel(n_vocab_size = vocab_size)
    model = model.to(device)
    
    # train model (train.py)
    print(count_trainable_parameters(model))
    train_model(model, dataloaders)
    #print(loss)
    
    # evaluate perplexity for all three splits (evaluate.py)
    """
    Taken care in train.py function for train and val loss & perplexity
    For test data loss and perplexity will run the evaluation.py separately post this.
    """
    print("Training complete run evaluation.py")

if __name__ == "__main__":
    main()
