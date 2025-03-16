import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, Any, List, Tuple


def save_checkpoint(model: nn.Module, 
                   optimizer: torch.optim.Optimizer, 
                   epoch: int,
                   loss: float,
                   accuracy: float,
                   save_dir: str,
                   model_name: str):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        loss: Validation loss
        accuracy: Validation accuracy
        save_dir: Directory to save checkpoint
        model_name: Name of the model
    """
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    
    filename = os.path.join(save_dir, f'{model_name}_epoch_{epoch}.pt')
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")
    
    # Save best model separately
    best_model_path = os.path.join(save_dir, f'{model_name}_best.pt')
    if not os.path.exists(best_model_path) or loss < torch.load(best_model_path)['loss']:
        torch.save(checkpoint, best_model_path)
        print(f"New best model saved to {best_model_path}")


def load_checkpoint(model: nn.Module, 
                   optimizer: torch.optim.Optimizer = None, 
                   checkpoint_path: str = None) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer (optional)
        checkpoint_path: Path to checkpoint file
        
    Returns:
        model: Loaded model
        checkpoint: Checkpoint dictionary
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, checkpoint


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute classification metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted logits or probabilities
        threshold: Threshold for converting probabilities to binary predictions
        
    Returns:
        metrics: Dictionary of metrics
    """
    # Apply sigmoid to convert logits to probabilities if needed
    # If the predictions are already in range [0, 1], this won't change them much
    y_pred_probs = 1 / (1 + np.exp(-y_pred)) if np.any(y_pred > 1.0) or np.any(y_pred < 0.0) else y_pred
    
    y_pred_binary = (y_pred_probs >= threshold).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'precision': precision_score(y_true, y_pred_binary, zero_division=0),
        'recall': recall_score(y_true, y_pred_binary, zero_division=0),
        'f1': f1_score(y_true, y_pred_binary, zero_division=0),
        'auc': roc_auc_score(y_true, y_pred_probs) if len(np.unique(y_true)) > 1 else 0
    }
    
    return metrics


def visualize_training(train_losses: List[float], 
                      val_losses: List[float], 
                      train_accs: List[float],
                      val_accs: List[float],
                      save_path: str = None):
    """
    Visualize training and validation metrics
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        save_path: Path to save the plot (if None, plot is displayed)
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False