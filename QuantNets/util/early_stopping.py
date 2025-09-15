import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                             Default: 0
            restore_best_weights (bool): Whether to restore model weights from the epoch with the best value.
                                       Default: True
            verbose (bool): If True, prints a message for each validation loss improvement. 
                          Default: False
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None
        
    def __call__(self, val_loss, model=None):
        """
        Call this method after each epoch with the validation loss.
        
        Args:
            val_loss (float): Current validation loss
            model (torch.nn.Module): Model to save weights from (if restore_best_weights=True)
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        score = -val_loss  # We want to maximize the negative loss (minimize loss)
        
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights and model is not None:
                self.best_weights = model.state_dict().copy()
            if self.verbose:
                print(f'Initial validation loss: {val_loss:.6f}')
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print('Early stopping triggered')
        else:
            self.best_score = score
            if self.restore_best_weights and model is not None:
                self.best_weights = model.state_dict().copy()
            self.counter = 0
            if self.verbose:
                print(f'Validation loss improved: {val_loss:.6f}')
                
        return self.early_stop
    
    def restore_weights(self, model):
        """Restore the best weights to the model."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            if self.verbose:
                print('Restored best model weights')