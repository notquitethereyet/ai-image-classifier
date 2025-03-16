import os
from typing import Dict, Any

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Create directories if they don't exist
for directory in [MODEL_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Training configuration
TRAIN_CONFIG: Dict[str, Any] = {
    # Dataset settings
    'data_dir': DATA_DIR,
    'img_size': 224,
    'batch_size': 32,
    'num_workers': 4,
    
    # Model settings
    'model_name': 'resnet50',  # Options: 'resnet50', 'efficientnet_b2', 'vit_b_16'
    'num_classes': 1,          # 1 for binary classification with sigmoid
    'pretrained': True,
    
    # Training settings
    'epochs': 20,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'scheduler': 'cosine',     # Options: 'cosine', 'step', or None
    'seed': 42,
    
    # Validation settings
    'val_split': 0.1,         # 10% of training data for validation
    
    # Checkpoint settings
    'save_dir': MODEL_DIR,
    'model_name_prefix': 'art_classifier',
    'save_interval': 1,        # Save checkpoint every n epochs
    
    # Mixed precision training (for faster training on compatible GPUs)
    'mixed_precision': True,
    
    # Early stopping
    'early_stopping': True,
    'patience': 5,             # Number of epochs to wait for improvement
    'min_delta': 0.001,        # Minimum change to qualify as improvement
}

# Inference configuration
INFERENCE_CONFIG: Dict[str, Any] = {
    'checkpoint_path': os.path.join(MODEL_DIR, f"art_classifier_resnet50_best.pt"),
    'threshold': 0.5,          # Threshold for binary classification
    'output_csv': os.path.join(BASE_DIR, 'predictions.csv')
}