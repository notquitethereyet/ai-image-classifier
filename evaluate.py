import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay

# Import project modules
from datasets.art_dataset import ArtDataset
from models.classifier import create_model
from utils.helper import load_checkpoint, compute_metrics
from config import TRAIN_CONFIG, INFERENCE_CONFIG


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate AI Art Classifier")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=os.path.join(TRAIN_CONFIG['save_dir'], f"{TRAIN_CONFIG['model_name_prefix']}_{TRAIN_CONFIG['model_name']}_best.pt"),
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default=TRAIN_CONFIG['data_dir'],
        help="Directory containing dataset"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="evaluation",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32,
        help="Batch size for evaluation"
    )
    return parser.parse_args()


def create_test_loader(data_dir, csv_file="_temp_val.csv", batch_size=32):
    """Create a data loader for validation/test data"""
    from torchvision import transforms
    
    # Define transform
    test_transform = transforms.Compose([
        transforms.Resize((TRAIN_CONFIG['img_size'], TRAIN_CONFIG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    test_dataset = ArtDataset(
        csv_file=os.path.join(data_dir, csv_file),
        root_dir=data_dir,
        transform=test_transform,
        is_test=False
    )
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=TRAIN_CONFIG['num_workers'],
        pin_memory=True
    )
    
    return test_loader


def evaluate_model(model, data_loader, device):
    """Evaluate model on a dataset"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.numpy()
            
            outputs = model(images)
            predictions = outputs.detach().cpu().numpy()
            
            all_preds.extend(predictions)
            all_labels.extend(labels)
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    
    return all_labels, all_preds


def plot_roc_curve(y_true, y_pred, output_dir):
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()


def plot_precision_recall_curve(y_true, y_pred, output_dir):
    """Plot precision-recall curve"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
    plt.close()


def plot_confusion_matrix(y_true, y_pred_binary, output_dir):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred_binary)
    
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Human", "AI"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()


def plot_threshold_metrics(y_true, y_pred, output_dir):
    """Plot metrics across different thresholds"""
    thresholds = np.arange(0.1, 1.0, 0.05)
    metrics = []
    
    for threshold in thresholds:
        y_pred_binary = (y_pred >= threshold).astype(int)
        results = compute_metrics(y_true, y_pred, threshold)
        results['threshold'] = threshold
        metrics.append(results)
    
    metrics_df = pd.DataFrame(metrics)
    
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['threshold'], metrics_df['accuracy'], label='Accuracy')
    plt.plot(metrics_df['threshold'], metrics_df['precision'], label='Precision')
    plt.plot(metrics_df['threshold'], metrics_df['recall'], label='Recall')
    plt.plot(metrics_df['threshold'], metrics_df['f1'], label='F1 Score')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'threshold_metrics.png'))
    plt.close()
    
    # Save metrics to CSV
    metrics_df.to_csv(os.path.join(output_dir, 'threshold_metrics.csv'), index=False)
    
    # Find optimal threshold for F1 score
    best_f1_idx = metrics_df['f1'].idxmax()
    best_threshold = metrics_df.iloc[best_f1_idx]['threshold']
    best_metrics = metrics_df.iloc[best_f1_idx].to_dict()
    
    return best_threshold, best_metrics


def main():
    """Main evaluation function"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(TRAIN_CONFIG)
    
    # Load checkpoint
    model, checkpoint = load_checkpoint(model, checkpoint_path=args.checkpoint)
    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}, loss: {checkpoint['loss']:.4f}, accuracy: {checkpoint['accuracy']:.4f}")
    
    # Move model to device
    model = model.to(device)
    
    # Create validation/test data loader
    # First check if validation split is saved, if not, create one
    val_csv_path = os.path.join(args.data_dir, '_temp_val.csv')
    if not os.path.exists(val_csv_path):
        print("Creating validation split...")
        from sklearn.model_selection import train_test_split
        
        # Read train CSV
        train_df = pd.read_csv(os.path.join(args.data_dir, 'train.csv'))
        
        # Split data
        train_df, val_df = train_test_split(
            train_df,
            test_size=TRAIN_CONFIG['val_split'],
            stratify=train_df['label'],
            random_state=TRAIN_CONFIG['seed']
        )
        
        # Save validation CSV
        val_df.reset_index(drop=True).to_csv(val_csv_path, index=True)
    
    # Create data loader
    test_loader = create_test_loader(args.data_dir, csv_file="_temp_val.csv", batch_size=args.batch_size)
    print(f"Test dataset size: {len(test_loader.dataset)}")
    
    # Evaluate model
    y_true, y_pred = evaluate_model(model, test_loader, device)
    
    # Compute metrics with default threshold
    default_metrics = compute_metrics(y_true, y_pred, threshold=INFERENCE_CONFIG['threshold'])
    print("\nEvaluation results with default threshold:")
    for metric, value in default_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot ROC curve
    plot_roc_curve(y_true, y_pred, args.output_dir)
    
    # Plot precision-recall curve
    plot_precision_recall_curve(y_true, y_pred, args.output_dir)
    
    # Plot confusion matrix with default threshold
    y_pred_binary = (y_pred >= INFERENCE_CONFIG['threshold']).astype(int)
    plot_confusion_matrix(y_true, y_pred_binary, args.output_dir)
    
    # Find optimal threshold and plot metrics
    best_threshold, best_metrics = plot_threshold_metrics(y_true, y_pred, args.output_dir)
    
    print(f"\nOptimal threshold: {best_threshold:.4f}")
    print("Metrics with optimal threshold:")
    for metric, value in best_metrics.items():
        if metric != 'threshold':
            print(f"{metric}: {value:.4f}")
    
    # Save evaluation summary
    summary = {
        "model": TRAIN_CONFIG['model_name'],
        "checkpoint_epoch": checkpoint['epoch'],
        "default_threshold": INFERENCE_CONFIG['threshold'],
        "optimal_threshold": best_threshold,
        "default_metrics": default_metrics,
        "optimal_metrics": best_metrics
    }
    
    summary_df = pd.DataFrame({
        "Metric": list(default_metrics.keys()),
        f"Default (t={INFERENCE_CONFIG['threshold']})": list(default_metrics.values()),
        f"Optimal (t={best_threshold:.2f})": [best_metrics[k] for k in default_metrics.keys()]
    })
    
    summary_df.to_csv(os.path.join(args.output_dir, 'evaluation_summary.csv'), index=False)
    
    print(f"\nEvaluation results saved to {args.output_dir}")
    
    # Clean up
    if os.path.exists(val_csv_path):
        os.remove(val_csv_path)


if __name__ == "__main__":
    main()