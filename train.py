import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Import project modules
from datasets.art_dataset import ArtDataset, get_data_loaders
from models.classifier import create_model
from utils.helper import (
    save_checkpoint, 
    compute_metrics, 
    visualize_training, 
    set_seed
)
from config import TRAIN_CONFIG


def train_one_epoch(model: nn.Module,
                   train_loader: DataLoader,
                   criterion: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   device: torch.device,
                   scaler: GradScaler = None) -> tuple:
    """
    Train model for one epoch
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run training on
        scaler: Gradient scaler for mixed precision training
        
    Returns:
        avg_loss, avg_acc: Average loss and accuracy for the epoch
    """
    model.train()
    epoch_loss = 0.0
    epoch_preds = []
    epoch_targets = []
    
    # Add tqdm progress bar
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device).view(-1, 1)  # Reshape to [batch_size, 1]
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass with optional mixed precision
        if scaler is not None:
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Record loss and predictions
        epoch_loss += loss.item() * images.size(0)
        epoch_preds.extend(outputs.detach().cpu().numpy())
        epoch_targets.extend(labels.detach().cpu().numpy())
    
    # Calculate average loss and metrics
    avg_loss = epoch_loss / len(train_loader.dataset)
    metrics = compute_metrics(np.array(epoch_targets), np.array(epoch_preds))
    
    return avg_loss, metrics


def validate(model: nn.Module,
            val_loader: DataLoader,
            criterion: nn.Module,
            device: torch.device) -> tuple:
    """
    Validate model performance
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to run validation on
        
    Returns:
        avg_loss, avg_acc: Average loss and accuracy
    """
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_targets = []
    
    # Add tqdm progress bar
    progress_bar = tqdm(val_loader, desc="Validating", leave=False)
    
    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device).view(-1, 1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            val_preds.extend(outputs.detach().cpu().numpy())
            val_targets.extend(labels.detach().cpu().numpy())
    
    # Calculate average loss and metrics
    avg_loss = val_loss / len(val_loader.dataset)
    metrics = compute_metrics(np.array(val_targets), np.array(val_preds))
    
    return avg_loss, metrics


def create_train_val_loaders(data_dir: str, config: dict) -> tuple:
    """
    Create train and validation data loaders with proper stratification
    
    Args:
        data_dir: Directory with data
        config: Training configuration
        
    Returns:
        train_loader, val_loader: Data loaders
    """
    # Read CSV file with string type for file names
    train_csv_path = os.path.join(data_dir, 'train.csv')
    df = pd.read_csv(train_csv_path, dtype={'file_name': str})
    
    # Ensure column names are as expected
    required_columns = ['file_name', 'label']
    for col in required_columns:
        if col not in df.columns:
            # Check if using default pandas index column naming
            if df.columns[0] == 'Unnamed: 0' and col == 'file_name' and 'label' in df.columns:
                # First column might be index, second is file_name
                df['file_name'] = df.iloc[:, 1].astype(str)
            else:
                raise ValueError(f"Required column '{col}' not found in CSV. Available columns: {df.columns.tolist()}")
    
    # Split data into train and validation sets with stratification
    train_df, val_df = train_test_split(
        df, 
        test_size=config['val_split'],
        stratify=df['label'],
        random_state=config['seed']
    )
    
    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    # Save temporary CSV files
    train_temp_csv = os.path.join(data_dir, '_temp_train.csv')
    val_temp_csv = os.path.join(data_dir, '_temp_val.csv')
    train_df.to_csv(train_temp_csv, index=True)
    val_df.to_csv(val_temp_csv, index=True)
    
    # Create data transforms
    from torchvision import transforms
    
    train_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = ArtDataset(
        csv_file=train_temp_csv,
        root_dir=data_dir,
        transform=train_transform,
        is_test=False
    )
    
    val_dataset = ArtDataset(
        csv_file=val_temp_csv,
        root_dir=data_dir,
        transform=val_transform,
        is_test=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader


def main():
    """Main training function"""
    config = TRAIN_CONFIG
    
    # Set random seed for reproducibility
    set_seed(config['seed'])
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader = create_train_val_loaders(config['data_dir'], config)
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    
    # Create model
    model = create_model(config)
    print(f"Created model: {config['model_name']}")
    
    # Define loss function (BCEWithLogitsLoss for binary classification)
    # This is safer with mixed precision training as it combines sigmoid and BCE
    criterion = nn.BCEWithLogitsLoss()
    
    # Define optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Define learning rate scheduler
    if config['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config['epochs']
        )
    elif config['scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=5,
            gamma=0.1
        )
    else:
        scheduler = None
    
    # Initialize gradient scaler for mixed precision training
    scaler = torch.amp.GradScaler() if config['mixed_precision'] and torch.cuda.is_available() else None
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    # Training loop
    print(f"Starting training for {config['epochs']} epochs...")
    start_time = time.time()
    
    # Add tqdm for epoch tracking
    epochs_range = tqdm(range(1, config['epochs'] + 1), desc="Training Progress")
    
    for epoch in epochs_range:
        epoch_start = time.time()
        
        # Train for one epoch
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_metrics['accuracy'])
        val_accs.append(val_metrics['accuracy'])
        
        # Print progress
        epoch_time = time.time() - epoch_start
        epochs_range.set_postfix({
            'train_loss': f"{train_loss:.4f}",
            'val_loss': f"{val_loss:.4f}",
            'train_acc': f"{train_metrics['accuracy']:.4f}",
            'val_acc': f"{val_metrics['accuracy']:.4f}",
            'time': f"{epoch_time:.2f}s"
        })
        
        # Save checkpoint
        if epoch % config['save_interval'] == 0:
            save_checkpoint(
                model, 
                optimizer, 
                epoch,
                val_loss,
                val_metrics['accuracy'],
                config['save_dir'],
                f"{config['model_name_prefix']}_{config['model_name']}"
            )
        
        # Early stopping
        if config['early_stopping']:
            if val_loss < best_val_loss - config['min_delta']:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                
            if epochs_without_improvement >= config['patience']:
                print(f"Early stopping triggered after {epoch} epochs")
                break
    
    # Training complete
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    
    # Visualize training history
    visualize_training(
        train_losses, 
        val_losses, 
        train_accs,
        val_accs,
        save_path=os.path.join(TRAIN_CONFIG['save_dir'], 'training_history.png')
    )
    
    # Clean up temporary files
    temp_files = [
        os.path.join(config['data_dir'], '_temp_train.csv'),
        os.path.join(config['data_dir'], '_temp_val.csv')
    ]
    for file in temp_files:
        if os.path.exists(file):
            os.remove(file)


if __name__ == "__main__":
    main()